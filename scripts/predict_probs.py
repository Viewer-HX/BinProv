#!/usr/bin/env python3
"""Score an existing checkpoint over another task's window set.

Two uses, both of which avoid a fine-tune:

* **Routing.** `scripts/combine.py --route` can send GCC windows to one
  specialist and Clang windows to another, but only if it has a compiler
  prediction for exactly the windows the O2/O3 runs were scored on. The already
  fine-tuned `compiler` checkpoint provides that; it just has to be run over the
  O2/O3 index rather than its own.

* **Reuse as an ensemble member.** The `opt4` checkpoint already distinguishes
  all four levels. Restricted to O2/O3 windows and marginalized onto {O2, O3} it
  is a second, independently trained opinion on the hard task, free.

``--window-task`` names the task whose windows to score; ``--ckpt`` is a
checkpoint that may have been trained for a different one. Output is a
`probs.npz` in the same format `experiment.py` writes, so `combine.py` treats it
like any other run.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binprov import engine  # noqa: E402
from binprov.corpus import Corpus  # noqa: E402
from binprov.data import (  # noqa: E402
    ByteSequenceDataset,
    ClassificationCollator,
    drop_unlabelled,
    sequence_labels,
)
from binprov.metrics import accuracy, pct  # noqa: E402
from binprov.model import BinProvForProvenance  # noqa: E402
from binprov.provenance import get_task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--window-task", required=True,
                    help="task defining which windows to score and the truth column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="default")
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--keep-classes", nargs="*", default=None,
                    help="for a checkpoint with more classes than the window task, "
                         "the subset to renormalise over, e.g. --keep-classes O2 O3")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    corpus = Corpus(args.corpus)
    wtask = get_task(args.window_task)
    index = corpus.sequences(level="binary", bids=corpus.load_splits(args.splits)[args.split_name])
    wlabels = sequence_labels(corpus, index, wtask)
    index, wlabels = drop_unlabelled(index, wlabels)
    print(f"{len(index):,} windows from task {wtask.name}", flush=True)

    model, head = BinProvForProvenance.load(args.ckpt)
    classes = head.get("classes") or list(range(head["num_labels"]))
    print(f"checkpoint {args.ckpt}: {head['num_labels']} classes {classes}", flush=True)
    device, amp_dtype = engine.pick_device()
    model.to(device)

    collator = ClassificationCollator(model.cfg.seq_tokens)
    loader = engine.make_loader(
        ByteSequenceDataset(corpus, index, wlabels), collator,
        batch_size=args.batch_size, shuffle=False, workers=args.workers,
    )
    res = engine.predict(model, loader, device, amp_dtype,
                         num_labels=head["num_labels"], progress_every=400)
    prob = res["prob"]

    if args.keep_classes:
        cols = [classes.index(c) for c in args.keep_classes]
        # Renormalise over the kept columns: conditioning on "this window is O2
        # or O3", which is exactly what the O2/O3 task assumes anyway.
        prob = prob[:, cols]
        prob = prob / np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
        print(f"restricted to {args.keep_classes} and renormalised", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "probs.npz",
        prob=prob.astype(np.float32), true=wlabels.astype(np.int16),
        bid=index.bid, start=index.start, length=index.length,
        marginal_map=np.zeros(0, dtype=np.int16),
        fingerprint=np.frombuffer(
            hashlib.sha1(index.bid.tobytes() + index.start.tobytes()).digest(), dtype=np.uint8
        ),
    )
    acc = accuracy(prob.argmax(1), wlabels) if prob.shape[1] == wtask.num_labels else float("nan")
    engine.save_json(out / "result.json", {
        "tag": args.tag or Path(args.ckpt).name,
        "task": wtask.name,
        "source_checkpoint": str(args.ckpt),
        "source_classes": classes,
        "keep_classes": args.keep_classes,
        "scored_classes": list(wtask.classes),
        "test": {"accuracy": acc, "n": int(len(wlabels))},
    })
    print(f"accuracy on the window task: {pct(acc) if acc == acc else 'n/a'}")
    print(f"wrote {out}/probs.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
