#!/usr/bin/env python3
"""Minimal BinProv inference for a packaged export directory.

This file is copied verbatim into every ``export_hf.py`` output and is the
supported way to run a released checkpoint. It loads the BinProv-native model
from ``model/`` next to this file, reads an ELF's ``.text`` with BinProv's own
dependency-free reader (or raw ``.text`` bytes), replicates the training cut into
``seq_bytes``-byte windows, predicts each window, and soft-votes per class over
the windows.

The end-to-end model is NOT a transformers AutoModel (the head is BinProv's), so
this script needs the BinProv package installed::

    pip install -r requirements.txt

No remote code is executed anywhere; there is no ``trust_remote_code``.

Example::

    python inference.py --model model --elf /path/to/binary.elf
    python inference.py --model model --text-bytes raw_text.bin
    python inference.py --model model --elf a.elf --json   # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPORT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = EXPORT_ROOT / "model"


def load_model(model_dir: Path):
    from binprov.model import BinProvForProvenance

    return BinProvForProvenance.load(model_dir)


def cut_windows(data: bytes, seq_bytes: int, stride: int, min_bytes: int = 16):
    """Cut overlapping model inputs at the evaluation stride."""
    out = []
    pos = 0
    n = len(data)
    i = 0
    while pos < n:
        ln = min(seq_bytes, n - pos)
        if ln < min_bytes and pos > 0:
            break
        out.append((np.frombuffer(data[pos : pos + ln], dtype=np.uint8), i))
        i += 1
        if ln < seq_bytes:
            break
        pos += stride
    return out


def predict(model, data: bytes, *, stride: int, batch_size: int, device):
    """Per-class mean probability over the binary's windows.

    Returns ``(probabilities, num_windows)`` with probabilities the soft vote of
    every non-overlapping ``seq_bytes`` window of ``data``.
    """
    from binprov.data import ClassificationCollator

    seq = model.cfg.seq_bytes
    coll = ClassificationCollator(model.cfg.seq_tokens)
    windows = cut_windows(data, seq, stride)
    if not windows:
        raise ValueError("no usable windows: input is empty or shorter than 16 bytes")
    model.to(device).eval()
    total = np.zeros(model.num_labels, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            items = [(chunk, -1, idx) for chunk, idx in windows[start:start + batch_size]]
            out = coll(items)
            ids = out["input_ids"].to(device)
            attn = out["attention_mask"].to(device)
            types = out["token_type_ids"].to(device)
            logits = model(ids, attn, types)["logits"]
            total += logits.float().softmax(-1).sum(0).cpu().numpy()
    return total / len(windows), len(windows)


def choose_device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="BinProv inference on an ELF or raw .text bytes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="model dir (model.pt + configs); default: ./model")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--elf", default=None, help="ELF binary to classify")
    src.add_argument("--text-bytes", default=None,
                     help="file containing raw .text bytes")
    ap.add_argument("--json", action="store_true", help="print JSON only")
    ap.add_argument("--stride", type=int, default=None,
                    help="window stride in bytes; default: packaged evaluation stride")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="inference batch size; default 1 limits memory use")
    ap.add_argument("--device", default="auto",
                    help="torch device (auto, cuda, mps, cpu); default: auto")
    args = ap.parse_args()

    if args.elf:
        from binprov.elf import parse

        buf = Path(args.elf).read_bytes()
        text = parse(buf).data
        source = f"ELF {args.elf} (.text, {len(text)} bytes)"
    else:
        text = Path(args.text_bytes).read_bytes()
        source = f"{args.text_bytes} ({len(text)} bytes)"

    model_dir = Path(args.model)
    model, head = load_model(model_dir)
    labels_file = model_dir / "labels.json"
    labels = json.loads(labels_file.read_text()) if labels_file.is_file() else {}
    classes = labels.get("classes") or []
    inference_cfg_file = model_dir / "inference_config.json"
    inference_cfg = json.loads(inference_cfg_file.read_text()) if inference_cfg_file.is_file() else {}
    stride = args.stride or int(inference_cfg.get("stride", model.cfg.seq_bytes))
    if stride <= 0 or args.batch_size <= 0:
        ap.error("--stride and --batch-size must be positive")
    device = choose_device(args.device)
    prob, n_windows = predict(
        model, text, stride=stride, batch_size=args.batch_size, device=device
    )
    pred = int(prob.argmax())
    result = {
        "source": source,
        "num_windows": n_windows,
        "seq_bytes": model.cfg.seq_bytes,
        "stride": stride,
        "device": str(device),
        "classes": classes,
        "probabilities": [round(float(x), 6) for x in prob],
        "prediction": pred,
        "predicted_label": classes[pred] if pred < len(classes) else str(pred),
        "task": labels.get("task"),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"input: {source}")
        print(f"task: {result['task']}   classes: {classes}")
        print(f"windows: {n_windows}   input: {model.cfg.seq_bytes} B   stride: {stride} B")
        print(f"device: {device}")
        for i, p in enumerate(prob):
            name = classes[i] if i < len(classes) else str(i)
            print(f"  {name:>6}: {100 * p:.2f}%")
        print(f"=> {result['predicted_label']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
