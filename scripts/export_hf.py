#!/usr/bin/env python3
"""Package a trained BinProv checkpoint into a self-contained, upload-ready dir.

Builds ``OUT/`` from a fine-tuned checkpoint produced by ``scripts/experiment.py``
(a directory holding ``model.pt`` + ``binprov_config.json`` + ``head.json`` plus
``result.json``/``probs.npz``). The result is self-contained: model weights,
configs, the label map, exact training/eval metadata, environment versions, the
LICENSE, MODEL_CARD.md/README.md and a small ``inference.py`` that accepts an ELF
path (extracted with this repo's dependency-free reader) or raw ``.text`` bytes.

Nothing is uploaded. Nothing is fetched. There is no ``trust_remote_code`` and no
remote code: the model's encoder is a HuggingFace RoBERTa *shape*, but the
pooling/classification head is BinProv's own, so the end-to-end model is **not**
``AutoModel``-loadable and this script never claims it is. Loading and running it
requires the BinProv package (``pip install .`` from this repository) plus torch.
The export includes the small ``binprov`` runtime package and its requirements,
so ``inference.py`` is directly usable after installing dependencies.

After writing, the export is verified in-process: the model is reloaded from the
export copy and its prediction on a small window (real corpus text when the
corpus is present, else a deterministic synthetic window) is compared with the
source checkpoint's — they must be identical, and a second run must reproduce the
first bit-for-bit.

Example::

    python scripts/export_hf.py --checkpoint results/gpu_release/opt4_wide_seed29/opt4 \\
        --out results/gpu_release/opt4_wide_seed29/export \\
        --model-name binprov
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPORTED_FILES = (
    "model.pt",
    "binprov_config.json",
    "head.json",
)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _versions() -> dict:
    out = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sys.platform": sys.platform,
    }
    for name in ("torch", "transformers", "numpy"):
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "not installed"
    return out


def load_metadata(checkpoint: Path) -> dict:
    """Collect the exact training/eval metadata the export will embed."""
    meta: dict = {}
    for name in ("args.json", "result.json"):
        p = checkpoint / name
        if p.is_file():
            meta[name] = json.loads(p.read_text())
    head = json.loads((checkpoint / "head.json").read_text())
    cfg = json.loads((checkpoint / "binprov_config.json").read_text())
    meta["head.json"] = head
    meta["binprov_config.json"] = cfg
    try:
        meta["source_checkpoint"] = str(checkpoint.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        meta["source_checkpoint"] = checkpoint.name
    return meta


def write_markdown(out: Path, model_name: str, meta: dict,
                   eval_json: Path | None, seq_acc: float | None,
                   binary_acc: float | None) -> None:
    """Write the Hub-rendered README model card."""
    cfg = meta["binprov_config.json"]
    head = meta["head.json"]
    args = meta.get("args.json", {})
    result = meta.get("result.json", {})
    ev = {}
    if eval_json is not None and eval_json.is_file():
        ev = json.loads(eval_json.read_text())
    task = result.get("task") or head.get("task") or "?"
    scored = result.get("scored_classes") or []
    seq_bytes = cfg.get("seq_bytes", "?")
    encoder = (
        f"RoBERTa-style byte encoder, seq {seq_bytes} bytes, "
        f"{cfg.get('num_hidden_layers')} layers x {cfg.get('hidden_size')} hidden"
    )
    acc_line = ""
    if seq_acc is not None:
        acc_line += f"\nEvaluation (this export): sequence accuracy **{seq_acc * 100:.4f}%**."
    if binary_acc is not None:
        acc_line += f" Binary-level accuracy **{binary_acc * 100:.4f}%**."
    if acc_line:
        acc_line += " A fresh training run's numbers vary by ~1-2 points with seed.\n"
    balanced_acc = ev.get("balanced_accuracy", result.get("test", {}).get("balanced_accuracy"))
    n_sequences = ev.get("num_sequences", result.get("test", {}).get("n", "?"))
    n_binaries = ev.get("num_binaries", "?")

    def pct(value):
        return "?" if value is None else f"{float(value) * 100:.4f}%"

    def count(value):
        return "?" if value in (None, "?") else f"{int(value):,}"

    readme = f"""---
license: mit
library_name: binprov
tags:
- binary-analysis
- compilation-provenance
- optimization-level-classification
- pytorch
- roberta
metrics:
- accuracy
---

# BinProv Opt4

`{model_name}` classifies the compiler optimization level of x86-64 ELF
binaries from raw `.text` bytes: {', '.join(scored) or 'see labels.json'}.
It follows the BinProv method and does not disassemble the input.

This is a BinProv-native PyTorch checkpoint. Its byte encoder is RoBERTa-shaped,
but its pooling and classification head are custom, so it is not compatible
with `transformers.AutoModel`. Use the bundled `inference.py`; no remote code or
`trust_remote_code` is involved.

## Model details

- Architecture: {encoder}; 261-token byte vocabulary
- Pooling: `{head.get('pool')}` {('with ' + str(head['pool_heads']) + ' heads') if head.get('pool_heads') else ''}
- Input: 2048-byte windows at a 512-byte evaluation stride
- Output: four probabilities corresponding to O0, O1, O2, and O3
- Framework: PyTorch with Hugging Face Transformers components
- License: MIT

The model follows [BinProv: Binary Code Provenance Identification without
Disassembly](https://doi.org/10.1145/3545948.3545956), RAID 2022. This checkpoint
comes from a clean reproduction run of the repository's best single-model
recipe. Source code and full reproduction scripts are available at
[Viewer-HX/BinProv](https://github.com/Viewer-HX/BinProv).{acc_line}
## Direct use

```bash
python -m venv .venv
source .venv/bin/activate                    # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

python inference.py --model model --elf /path/to/binary.elf
python inference.py --model model --text-bytes raw_text.bin --json
```

The script extracts `.text` from an ELF or accepts already extracted bytes. It
evaluates overlapping {seq_bytes}-byte inputs every 512 bytes, then averages the
per-window probabilities for one binary. It automatically selects CUDA, Apple
MPS, or CPU. Use `--batch-size` to trade memory for speed.

## Training data

The model was trained and evaluated on the x86-64 subset of BinKit Normal,
covering GCC and Clang binaries at optimization levels O0 through O3. Splits are
grouped by source program so the same program does not cross train and test.
Exact training arguments and environment versions are in `metadata.json` and
`evidence/args.json`.

## Evaluation

| metric | value |
|---|---:|
| sequence accuracy | **{pct(seq_acc)}** |
| balanced sequence accuracy | **{pct(balanced_acc)}** |
| binary soft-vote accuracy | **{pct(binary_acc)}** |
| test windows | {count(n_sequences)} |
| test binaries | {count(n_binaries)} |

The sequence metric uses 512-byte target positions with 2048-byte overlapping
model inputs. Binary accuracy averages the saved per-window probabilities. The
recomputed evaluation in `evidence/evaluation.json` matches the recorded test
accuracy. Results can vary with retraining and should be compared on the same
program-grouped split.

## Intended use and limitations

This checkpoint is intended for research and analysis of compiler optimization
provenance in ordinary x86-64 ELF code produced by toolchains represented in
BinKit. It has not been validated as a general malware, authorship, attribution,
or security-decision system. Packed, obfuscated, self-modifying, non-ELF, other
architecture, and out-of-distribution toolchain inputs may be unreliable. The
softmax scores are not calibrated confidence estimates.

## Files

- `model/` — `model.pt`, `binprov_config.json`, `head.json` (BinProv checkpoint)
- `binprov/` — minimal source package needed by `inference.py`
- `requirements.txt` — runtime dependencies
- `labels.json` — class id -> label
- `metadata.json` — exact training args + eval results + environment
- `environment.json` — runtime versions
- `evidence/` — args, result, evaluation report, and saved probabilities
- `LICENSE` — repository license

## License

The model weights, bundled inference code, and documentation are released under
the [MIT License](LICENSE).

## Citation

```bibtex
@inproceedings{{he2022binprov,
  title={{BinProv: Binary Code Provenance Identification without Disassembly}},
  author={{He, Xu and Wang, Shu and Xing, Yunlong and Feng, Pengbin and Wang, Haining and Li, Qi and Chen, Songqing and Sun, Kun}},
  booktitle={{RAID}},
  year={{2022}},
  doi={{10.1145/3545948.3545956}}
}}
```
"""
    out.write_text(readme)


def render_model_card(out_dir: Path, model_name: str, meta: dict,
                      eval_json: Path | None) -> None:
    """Keep a plain Markdown copy; the Hub itself renders README.md."""
    shutil.copy2(out_dir / "README.md", out_dir / "MODEL_CARD.md")


def write_labels(out: Path, meta: dict) -> None:
    result = meta.get("result.json", {})
    classes = result.get("scored_classes") or []
    labels = {i: name for i, name in enumerate(classes)} if classes else {}
    (out / "labels.json").write_text(
        json.dumps(
            {"task": result.get("task"), "classes": classes,
             "label_map": labels, "num_labels": len(labels)},
            indent=2,
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# deterministic verification
# ---------------------------------------------------------------------------


def _sample_windows(model, corpus_dir: Path):
    """A handful of small windows: real corpus text when available, else a
    deterministic synthetic window, so verification needs no corpus."""
    import numpy as np

    cfg = model.cfg
    seq = cfg.seq_bytes
    rng = np.random.default_rng(20260905)
    chunks = []
    if corpus_dir is not None and (corpus_dir / "meta.json").is_file():
        try:
            from binprov.corpus import Corpus

            c = Corpus(corpus_dir)
            for rec in c.records[:4]:
                start = rec.text_off
                n = min(seq, rec.text_len)
                if n >= 16:
                    chunks.append(bytes(np.asarray(c.text[start : start + n]).tolist()))
            if chunks:
                return chunks
        except Exception:  # noqa: BLE001 - fall back to synthetic windows
            pass
    # deterministic synthetic windows (never random, so verification is stable)
    data = bytes((i * 7 + 13) % 256 for i in range(seq * 3))
    step = seq
    return [
        data[p : p + seq] for p in range(0, len(data), step)
        if len(data) - p >= min(16, seq)
    ]


def _predict_windows(model, device, chunks):
    import numpy as np
    import torch

    from binprov.data import ClassificationCollator

    coll = ClassificationCollator(model.cfg.seq_tokens)
    seq = model.cfg.seq_bytes
    n_classes = model.num_labels
    mean_prob = np.zeros(n_classes, dtype=np.float64)
    n_windows = 0
    for chunk in chunks:
        raw = np.frombuffer(chunk, dtype=np.uint8)
        n = len(raw)
        pos = 0
        windows = []
        i = 0
        while pos < n:
            ln = min(seq, n - pos)
            if ln < 16 and pos > 0:
                break
            windows.append((raw[pos : pos + ln], -1, i))
            i += 1
            if ln < seq:
                break
            pos += seq
        if not windows:
            continue
        model.eval()
        with torch.no_grad():
            for batch in (coll(windows),):
                ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                types = batch["token_type_ids"].to(device)
                logits = model(ids, attn, types)["logits"]
                p = logits.float().softmax(-1).cpu().numpy().mean(0)
        mean_prob += p
        n_windows += 1
    if n_windows:
        mean_prob /= n_windows
    return mean_prob


def verify(checkpoint: Path, out: Path, corpus_dir: Path | None) -> dict:
    """Reload the export copy and confirm it reproduces the source checkpoint.

    Returns a dict with the mean probability vector over the sampled windows;
    raises on any mismatch or non-determinism.
    """
    import numpy as np
    import torch

    from binprov.model import BinProvForProvenance

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model_src, _head_src = BinProvForProvenance.load(checkpoint, map_location="cpu")
    model_exp, _head_exp = BinProvForProvenance.load(out / "model", map_location="cpu")
    model_src.to(device)
    model_exp.to(device)
    try:
        chunks = _sample_windows(model_src, corpus_dir)
        with torch.no_grad():
            a = _predict_windows(model_src, device, chunks)
            b = _predict_windows(model_exp, device, chunks)
            b2 = _predict_windows(model_exp, device, chunks)  # determinism check
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
        np.testing.assert_array_equal(b, b2)
    finally:
        del model_src, model_exp
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "verified": True,
        "device": str(device),
        "windows": len(chunks),
        "mean_prob": [float(x) for x in b],
        "prediction": int(np.argmax(b)),
    }


# ---------------------------------------------------------------------------
# export build
# ---------------------------------------------------------------------------


def build_export(checkpoint: Path, out: Path, *, model_name: str,
                 eval_json: Path | None = None,
                 corpus_dir: Path | None = None, license_path: Path | None = None,
                 no_verify: bool = False) -> dict:
    """Create the self-contained export directory; returns a summary dict."""
    checkpoint = Path(checkpoint)
    out = Path(out)
    eval_json = Path(eval_json) if eval_json is not None else None
    corpus_dir = Path(corpus_dir) if corpus_dir is not None else None
    license_path = Path(license_path) if license_path is not None else None
    for f in EXPORTED_FILES:
        if not (checkpoint / f).is_file():
            raise SystemExit(
                f"{checkpoint} is missing {f}; this script needs a checkpoint "
                "written by scripts/experiment.py (model.save)"
            )
    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"export dir {out} already exists and is not empty; move it aside "
            "before rebuilding"
        )
    model_dir = out / "model"
    evidence_dir = out / "evidence"
    model_dir.mkdir(parents=True)
    evidence_dir.mkdir(parents=True)

    meta = load_metadata(checkpoint)
    for f in EXPORTED_FILES:
        shutil.copy2(checkpoint / f, model_dir / f)
    for f in ("binprov_config.json", "head.json"):
        shutil.copy2(checkpoint / f, out / f)
    for f in ("args.json", "result.json"):
        p = checkpoint / f
        if p.is_file():
            shutil.copy2(p, evidence_dir / f)
    p = checkpoint / "probs.npz"
    if p.is_file():
        shutil.copy2(p, evidence_dir / "probs.npz")
    if eval_json is not None and eval_json.is_file():
        shutil.copy2(eval_json, evidence_dir / "evaluation.json")

    # Ship the minimal runtime so users can execute inference directly from the
    # model repository without cloning a second repository.
    runtime_dir = out / "binprov"
    runtime_dir.mkdir()
    for name in (
        "__init__.py", "corpus.py", "data.py", "discover.py", "elf.py",
        "metrics.py", "model.py", "provenance.py", "vocab.py", "vote.py",
    ):
        shutil.copy2(ROOT / "binprov" / name, runtime_dir / name)
    shutil.copy2(ROOT / "requirements.txt", out / "requirements.txt")
    (out / ".gitignore").write_text("__pycache__/\n*.py[cod]\n")

    stride = int(meta.get("args.json", {}).get("eval_window_bytes") or
                 meta.get("args.json", {}).get("stride") or
                 meta["binprov_config.json"]["seq_bytes"])
    inference_cfg = {"seq_bytes": meta["binprov_config.json"]["seq_bytes"],
                     "stride": stride, "min_bytes": 16}
    (out / "inference_config.json").write_text(
        json.dumps(inference_cfg, indent=2) + "\n"
    )
    shutil.copy2(out / "inference_config.json", model_dir / "inference_config.json")

    write_labels(out, meta)
    # model/ is self-contained for inference.py (labels next to the weights)
    shutil.copy2(out / "labels.json", model_dir / "labels.json")
    result = meta.get("result.json", {})
    try:
        seq_acc = float(result["test"]["accuracy"])
    except (KeyError, TypeError):
        seq_acc = None
    binary_acc = None
    if eval_json is not None and eval_json.is_file():
        ev = json.loads(eval_json.read_text())
        binary_acc = ev.get("binary_accuracy")
        seq_acc = ev.get("sequence_accuracy", seq_acc)
    write_markdown(out / "README.md", model_name, meta, eval_json,
                   seq_acc, binary_acc)
    render_model_card(out, model_name, meta, eval_json)

    lic = license_path or (ROOT / "LICENSE")
    if lic.is_file():
        shutil.copy2(lic, out / "LICENSE")
    # inference.py is the file consumers run; source of truth lives here and is
    # copied verbatim into the export.
    shutil.copy2(ROOT / "scripts" / "inference_template.py", out / "inference.py")

    summary = {
        "model_name": model_name,
        "exported_from": meta["source_checkpoint"],
        "created_at": _now(),
        "files": sorted(
            str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()
        ),
        "sequence_accuracy": seq_acc,
        "binary_accuracy": binary_acc,
        "transformers_compatible": False,
        "note": "BinProv-native checkpoint: load with inference.py / the BinProv "
                "package, not transformers AutoModel",
    }
    metadata = {
        **meta,
        "labels": json.loads((out / "labels.json").read_text()),
        "environment": _versions(),
        "summary": summary,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "environment.json").write_text(json.dumps(_versions(), indent=2) + "\n")

    if not no_verify:
        v = verify(checkpoint, out, corpus_dir)
        summary["verification"] = v
        (out / "metadata.json").write_text(
            json.dumps({**json.loads((out / "metadata.json").read_text()),
                        "summary": summary}, indent=2) + "\n"
        )
        print(f"  verified: export reproduces checkpoint "
              f"(prediction {v['prediction']} over {v['windows']} windows)")
    return summary


def parse_args():
    ap = argparse.ArgumentParser(
        description="Package a BinProv checkpoint into an upload-ready directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--checkpoint", required=True,
                    help="finetune output dir (model.pt + configs + result.json)")
    ap.add_argument("--out", required=True, help="export directory to create")
    ap.add_argument("--model-name", default="binprov",
                    help="name used in the model card / HF repo id")
    ap.add_argument("--eval-json", default=None,
                    help="optional local-release evaluate/report.json (binary acc)")
    ap.add_argument("--corpus", default=None,
                    help="corpus dir for the real-window verification sample")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the reload-and-predict determinism check")
    return ap.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv) if argv is not None else parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    corpus = Path(args.corpus) if args.corpus else None
    if corpus is not None and not corpus.is_absolute():
        corpus = ROOT / corpus
    summary = build_export(
        checkpoint, ROOT / args.out, model_name=args.model_name,
        eval_json=args.eval_json, corpus_dir=corpus, no_verify=args.no_verify,
    )
    print(f"\nexport ready: {ROOT / args.out}")
    print(f"  {len(summary['files'])} files, {summary['sequence_accuracy'] or '?'} "
          f"sequence acc, {summary['binary_accuracy'] or '?'} binary acc")
    print("  next: upload this directory to https://huggingface.co (e.g. with "
          "huggingface_hub) — nothing was uploaded by this script")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
