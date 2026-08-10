"""Shared training/inference plumbing.

Small on purpose: pre-training and fine-tuning have different enough loops that
sharing them behind one abstraction would obscure both. What *is* shared —
device selection, seeding, dataloaders, the inference pass — lives here.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(*, prefer_bf16: bool = True) -> tuple[torch.device, torch.dtype | None]:
    """Return ``(device, autocast_dtype)``.

    ``autocast_dtype`` is None on CPU, bf16 on any recent NVIDIA card (H200
    included), fp16 otherwise. bf16 needs no loss scaling, which keeps the
    training loop simpler.
    """
    if not torch.cuda.is_available():
        print("WARNING: no CUDA device visible; running on CPU will be very slow")
        return torch.device("cpu"), None
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        print(
            "WARNING: CUDA_VISIBLE_DEVICES is unset, so this process can see every GPU\n"
            "         on a shared machine. Pin it explicitly, e.g. "
            "`export CUDA_VISIBLE_DEVICES=7`."
        )
    dev = torch.device("cuda:0")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info(0)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    print(
        f"device: {name} (CUDA_VISIBLE_DEVICES={visible}), "
        f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free"
    )
    if prefer_bf16 and torch.cuda.is_bf16_supported():
        return dev, torch.bfloat16
    return dev, torch.float16


def make_loader(
    dataset,
    collator,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int = 4,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


@torch.no_grad()
def predict(
    model,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype | None,
    *,
    num_labels: int,
    progress_every: int = 200,
) -> dict[str, np.ndarray]:
    """Run the classifier over a loader.

    Returns arrays aligned to the loader's ``positions`` field, re-sorted into
    sequence-index order so they line up with the :class:`SequenceIndex` used to
    build the dataset. That alignment is what makes majority voting correct.
    """
    model.eval()
    n = len(loader.dataset)
    preds = np.zeros(n, dtype=np.int64)
    probs = np.zeros((n, num_labels), dtype=np.float32)
    truth = np.full(n, -1, dtype=np.int64)
    t0 = time.time()
    seen = 0

    for step, batch in enumerate(loader):
        ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        types = batch["token_type_ids"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None and device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=attn, token_type_ids=types)["logits"]
        p = logits.float().softmax(-1).cpu().numpy()
        pos = batch["positions"].numpy()
        probs[pos] = p
        preds[pos] = p.argmax(-1)
        truth[pos] = batch["labels"].numpy()
        seen += len(pos)
        if progress_every and step and step % progress_every == 0:
            rate = seen / (time.time() - t0)
            print(f"    {seen}/{n} sequences ({rate:.0f}/s)", flush=True)

    return {"pred": preds, "prob": probs, "true": truth}


class Timer:
    """Elapsed time plus a crude ETA, for long unattended runs."""

    def __init__(self, total_steps: int):
        self.total = max(1, total_steps)
        self.t0 = time.time()

    def eta(self, step: int) -> str:
        elapsed = time.time() - self.t0
        if step <= 0:
            return "?"
        remaining = elapsed * (self.total - step) / step
        return _hms(remaining)

    def elapsed(self) -> str:
        return _hms(time.time() - self.t0)


def _hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


class JsonLogger:
    """Appends one JSON object per line, and echoes a readable form to stdout."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a")
        else:
            self._fh = None

    def log(self, **fields) -> None:
        if self._fh:
            self._fh.write(json.dumps(fields, default=str) + "\n")
            self._fh.flush()
        parts = []
        for k, v in fields.items():
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
        print("  " + " ".join(parts), flush=True)

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def save_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
# resume
# ---------------------------------------------------------------------------

STATE_FILE = "training_state.pt"


def save_training_state(
    out_dir: str | Path,
    *,
    model,
    optimizer,
    scheduler=None,
    epoch: int,
    step: int,
    best: float | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a resumable snapshot: weights, optimizer, schedule, RNG.

    The GPUs here are shared with no scheduler, so a multi-hour run can lose its
    device at any moment. Without the optimizer moments and the schedule
    position, restarting is not resuming — Adam would rebuild its second-moment
    estimates from scratch and the learning rate would jump back up the warmup
    ramp.

    Written to a temporary file and renamed, because the snapshot is ~1 GB for
    the paper's 86M-parameter model and a process killed mid-write would
    otherwise leave a truncated file that fails to load.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best": best,
        "extra": extra or {},
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    final = out / STATE_FILE
    tmp = out / (STATE_FILE + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(final)
    return final


def load_training_state(
    out_dir: str | Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    map_location="cpu",
) -> dict | None:
    """Restore a snapshot written by :func:`save_training_state`.

    Returns the bookkeeping fields (``epoch``, ``step``, ``best``, ``extra``), or
    None when there is nothing to resume from — so callers can treat "no
    checkpoint yet" as a normal first run rather than an error.
    """
    path = Path(out_dir) / STATE_FILE
    if not path.exists():
        return None
    # weights_only=False: the payload carries RNG state objects, not just
    # tensors. Only ever point this at a checkpoint this code wrote.
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    rng = payload.get("rng") or {}
    try:
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"].cpu() if hasattr(rng["torch"], "cpu") else rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        if rng.get("python") is not None:
            random.setstate(rng["python"])
    except (RuntimeError, TypeError, ValueError) as exc:
        # A different GPU count or torch version can make RNG state unusable.
        # Losing exact reproducibility is not worth aborting a resume for.
        print(f"  note: could not restore RNG state ({exc}); continuing")
    return {
        "epoch": payload.get("epoch", 0),
        "step": payload.get("step", 0),
        "best": payload.get("best"),
        "extra": payload.get("extra", {}),
    }
