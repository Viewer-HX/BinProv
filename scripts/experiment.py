#!/usr/bin/env python3
"""Exploratory fine-tuning harness — the axes the paper does not vary.

`scripts/finetune.py` stays faithful to the paper. This script is where the
things the paper does not describe get tried, so the reproduction and the
exploration cannot contaminate each other. It differs from `finetune.py` in five
ways that matter:

**A real validation split.** `finetune.py` selects its best epoch by test
accuracy, which is fine for reproducing a published number but useless for
comparing twenty configurations — whichever one gets luckiest on the test set
wins. Here `--val-frac` holds out whole *programs* from the training side, model
selection uses only those, and the test number is read off at the
val-selected epoch. Both are reported, so a comparison against the paper-faithful
protocol is still available.

**Window jitter.** The paper's non-overlapping cut yields a fixed set of
windows, so an N-epoch run sees each byte string N times. `--jitter` re-draws
the window start on every read. See `JitteredByteSequenceDataset`.

**Pooling that is not a weighted mean.** `--pool attn|max|mil` — see the module
docstrings in `binprov/model.py`. The bet is that the O2/O3 signal is sparse
inside a window and a mean dilutes it.

**Factorized labels.** `--task opt_o2o3_x` trains on compiler x level and sums
the probabilities back down to O2/O3, which supervises the compiler split
instead of asking the model to marginalise it out unaided.

**Split protocol as an explicit knob.** `--split-mode sequence` splits
*sequences* rather than programs, so the same binary appears on both sides. That
is a leakage measurement, not a method — it exists to bound how much of a
published number a loose split can explain.

Every run writes `probs.npz` (per-sequence probabilities plus the bid/start of
each sequence), so ensembling, compiler routing and voting can be done offline
by `scripts/combine.py` without re-running the model.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binprov import engine  # noqa: E402
from binprov.corpus import Corpus  # noqa: E402
from binprov.data import (  # noqa: E402
    ByteSequenceDataset,
    ContextWindowDataset,
    ClassificationCollator,
    JitteredByteSequenceDataset,
    MultiViewByteSequenceDataset,
    drop_unlabelled,
    sequence_labels,
)
from binprov.metrics import accuracy, per_class_prf, pct  # noqa: E402
from binprov.pairs import (  # noqa: E402
    FunctionPairDataset,
    PairBatchSampler,
    build_function_pairs,
    pairwise_margin_loss,
)
from binprov.model import (  # noqa: E402
    BinProvConfig,
    BinProvForProvenance,
    cosine_schedule_with_warmup,
    describe,
)
from binprov.provenance import get_task  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune and evaluate BinProv provenance classifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="default")
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--from-scratch", action="store_true")
    ap.add_argument("--tag", default="", help="free-text label recorded in the result json")

    sel = ap.add_argument_group("data")
    sel.add_argument("--arch", nargs="*", default=None)
    sel.add_argument("--extra", nargs="*", default=None)
    sel.add_argument("--compiler", nargs="*", default=None,
                     help="restrict both training and test to gcc / clang")
    sel.add_argument("--version", nargs="*", default=None,
                     help="restrict to these compiler versions, e.g. --version 8.2.0 7.0")
    sel.add_argument("--test-version", nargs="*", default=None,
                     help="restrict only the *test* split to these compiler versions. "
                          "Training on all nine BinKit versions while testing on the "
                          "original two keeps the test windows identical to the "
                          "baseline's, which is the only way the two numbers compare")
    sel.add_argument("--train-compiler", nargs="*", default=None,
                     help="restrict only *training* to gcc / clang, keeping the full "
                          "test set so the run's rows stay routable against another "
                          "specialist's (see scripts/combine.py --route)")
    sel.add_argument("--max-seqs-per-binary", type=int, default=None)
    sel.add_argument("--stride", type=int, default=None,
                     help="training-window stride; < seq_bytes gives overlapping windows")
    sel.add_argument("--val-frac", type=float, default=0.15,
                     help="fraction of training *programs* held out for model selection")
    sel.add_argument("--val-seed", type=int, default=None,
                     help="seed for the validation carve-out, independent of --seed. "
                          "Defaults to --seed, which redraws the validation programs "
                          "for every replica. Pin this to a constant to hold the "
                          "partition fixed while varying training randomness; that makes "
                          "validation comparable across seeds and the comparison more "
                          "sensitive.")
    sel.add_argument(
        "--split-mode",
        choices=["program", "binary", "sequence"],
        default="program",
        help="'program' is the paper-faithful grouped split; 'binary' and "
        "'sequence' are progressively leakier and exist to be measured",
    )
    sel.add_argument("--seq-bytes", type=int, default=None,
                     help="bytes per sequence, overriding the checkpoint's. A longer "
                          "sequence stretches the pre-trained position table by "
                          "interpolation (see model._resize_position_embeddings)")
    sel.add_argument("--eval-window-bytes", type=int, default=None,
                     help="cut the val/test index at this granularity while still "
                          "feeding --seq-bytes centred on each window. Set it to 512 "
                          "with a longer --seq-bytes to score a long-context model over "
                          "exactly the baseline's 55,657 windows, which is the only way "
                          "the two accuracies are the same metric")
    sel.add_argument("--jitter", type=int, default=0,
                     help="max window-start shift in bytes (0 off, -1 fully random)")

    arch = ap.add_argument_group("model")
    arch.add_argument("--pos-extend", choices=["tile", "interp"], default="tile",
                      help="how to grow the pre-trained position table when "
                           "--seq-bytes exceeds the checkpoint's; see "
                           "model._resize_position_embeddings")
    arch.add_argument("--pool", default="border",
                      choices=["border", "mean", "attn", "max", "mil", "cls"])
    arch.add_argument("--pool-heads", type=int, default=4)
    arch.add_argument("--mil-tau", type=float, default=1.0)
    arch.add_argument("--taper", type=int, default=32)
    arch.add_argument("--dropout", type=float, default=None,
                      help="override the encoder's hidden/attention dropout")
    arch.add_argument("--classifier-dropout", type=float, default=0.1)

    opt = ap.add_argument_group("optimization")
    opt.add_argument("--epochs", type=int, default=3)
    opt.add_argument("--batch-size", type=int, default=128)
    opt.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="micro-batches per optimizer step. Each micro-batch's (combined) "
        "loss is divided by this count before backward, so accumulating "
        "--grad-accum of them reproduces the mean gradient over an effective "
        "batch of --grad-accum x --batch-size. A trailing partial group at the "
        "end of an epoch still steps once, so its gradients cannot leak into the "
        "next epoch. total_steps counts optimizer steps. 1 preserves the "
        "archived loop exactly",
    )
    opt.add_argument("--lr", type=float, default=3e-5)
    opt.add_argument("--head-lr", type=float, default=1e-4)
    opt.add_argument("--llrd", type=float, default=1.0,
                     help="layer-wise lr decay; 0.9 means layer k gets lr*0.9^(L-k)")
    opt.add_argument("--weight-decay", type=float, default=0.01)
    opt.add_argument("--warmup-ratio", type=float, default=0.06)
    opt.add_argument("--clip-grad", type=float, default=1.0)
    opt.add_argument("--label-smoothing", type=float, default=0.0)

    pair = ap.add_argument_group("within-program pairwise loss (binprov/pairs.py)")
    pair.add_argument("--use-pairs", action="store_true",
                      help="build the pair loader even when --pair-loss is 0, so the "
                           "extra function-centred windows can be ablated on their own")
    pair.add_argument("--pair-loss", type=float, default=0.0,
                      help="weight of the pairwise margin term; 0 disables pairs entirely")
    pair.add_argument("--pair-ce", type=float, default=1.0,
                      help="weight of plain cross-entropy on the pair batch. Set 0 to "
                           "attribute a gain to the margin term rather than to the "
                           "extra function-centred windows it brings with it")
    pair.add_argument("--pair-margin", type=float, default=1.0)
    pair.add_argument("--pairs-per-batch", type=int, default=32)
    pair.add_argument("--pair-min-size", type=int, default=32)
    pair.add_argument("--pair-jitter", type=int, default=64)

    run = ap.add_argument_group("run")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue from the per-epoch training_state.pt in --out if one "
        "exists, and snapshot the model/optimizer/scheduler after every epoch so "
        "an interrupted run can be resumed. Best-val weights are persisted to "
        "--out as they are found so a resumed run still evaluates the "
        "val-selected epoch. Combine freely with --grad-accum",
    )
    run.add_argument("--log-every", type=int, default=100)
    run.add_argument("--val-max-seqs", type=int, default=40000)
    run.add_argument("--tta", choices=["none", "crop", "shift"], default="none")
    run.add_argument("--tta-views", type=int, default=4)
    run.add_argument("--tta-span", type=int, default=128)
    run.add_argument("--save-probs", action="store_true", default=True)
    run.add_argument("--no-save-weights", action="store_true",
                     help="keep only metrics and probs; a sweep does not need 20 x 340 MB")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def program_of(corpus, bid: int) -> str:
    return corpus.records[bid].group


def carve_val(corpus, train_bids: list[int], frac: float, seed: int):
    """Hold out whole programs from the training side for model selection.

    Grouped by program for the same reason the train/test split is: a validation
    program that also appears in training would make the selection signal
    optimistic in exactly the way we are trying to avoid.
    """
    if frac <= 0:
        return train_bids, []
    groups = sorted({program_of(corpus, b) for b in train_bids})
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * frac)))
    val_groups = set(groups[:n_val])
    tr = [b for b in train_bids if program_of(corpus, b) not in val_groups]
    va = [b for b in train_bids if program_of(corpus, b) in val_groups]
    return tr, va


def build_index(corpus, task, bids, args, seq_bytes, *, compiler=None, version=None,
                is_train: bool = False):
    keep = set(
        corpus.filter_bids(arch=args.arch, extra=args.extra,
                           compiler=compiler or args.compiler,
                           compiler_version=version or args.version)
    )
    bids = [b for b in bids if b in keep]
    index = corpus.sequences(
        seq_len=seq_bytes,
        stride=args.stride,
        level="binary",
        bids=bids,
        # A per-binary cap is a *training* cost knob. Applying it to the test
        # split silently changes the test-set composition, which makes the run's
        # accuracy incomparable to every other run's while still looking like the
        # same metric -- so it is deliberately train-only.
        max_seqs_per_binary=args.max_seqs_per_binary if is_train else None,
    )
    labels = sequence_labels(corpus, index, task)
    return drop_unlabelled(index, labels)


def subsample(index, labels, cap: int | None, seed: int):
    if not cap or len(labels) <= cap:
        return index, labels
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    mask[rng.choice(len(labels), cap, replace=False)] = True
    return index.subset(mask), labels[mask]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def marginalize(probs: np.ndarray, task) -> np.ndarray:
    """Sum a factorized task's probabilities down onto its scored classes."""
    if task.marginal_of is None:
        return probs
    n_coarse = len(task.scored_classes)
    out = np.zeros((probs.shape[0], n_coarse), dtype=probs.dtype)
    for fine, coarse in enumerate(task.marginal_map):
        out[:, coarse] += probs[:, fine]
    return out


def coarse_labels(labels: np.ndarray, task) -> np.ndarray:
    if task.marginal_of is None:
        return labels
    lut = np.asarray(task.marginal_map, dtype=np.int64)
    return lut[labels]


def balanced_accuracy(pred: np.ndarray, true: np.ndarray, n: int) -> float:
    """Mean per-class recall.

    Reported next to plain accuracy because the corpus is balanced per *binary*
    and not per *sequence*: O0 binaries are longer, and after a re-split the
    O2:O3 sequence ratio can drift far enough that plain accuracy moves without
    the classifier changing at all. A configuration that only shifts the
    decision threshold shows up here as no gain.
    """
    prf = per_class_prf(pred, true, n)
    sup = prf["support"] > 0
    return float(prf["recall"][sup].mean()) if sup.any() else 0.0


def score(probs: np.ndarray, labels: np.ndarray, task) -> dict:
    p = marginalize(probs, task)
    y = coarse_labels(labels, task)
    pred = p.argmax(1)
    n = len(task.scored_classes)
    return {
        "accuracy": accuracy(pred, y),
        "balanced_accuracy": balanced_accuracy(pred, y, n),
        "confusion": per_class_prf(pred, y, n)["confusion"].tolist(),
        "n": int(len(y)),
    }


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------


def build_optimizer(model, args):
    """AdamW groups: head lr, encoder lr, optional layer-wise decay.

    Layer-wise lr decay is the standard remedy for a pre-trained encoder
    overfitting a small fine-tuning set: the lower layers hold the generic byte
    statistics that transfer, and updating them as fast as the top layers is what
    destroys them.
    """
    decay_names, no_decay = [], []

    def split(module_params):
        d, nd = [], []
        for name, p in module_params:
            if not p.requires_grad:
                continue
            (nd if (p.ndim == 1 or name.endswith(".bias") or "LayerNorm" in name
                    or "norm" in name) else d).append(p)
        return d, nd

    groups = []
    head_modules = [m for m in (model.pool, model.classifier) if m is not None]
    head_params = [(n, p) for m in head_modules for n, p in m.named_parameters()]
    d, nd = split(head_params)
    if d:
        groups.append({"params": d, "lr": args.head_lr, "weight_decay": args.weight_decay})
    if nd:
        groups.append({"params": nd, "lr": args.head_lr, "weight_decay": 0.0})

    enc = model.encoder
    n_layers = len(enc.encoder.layer)
    if args.llrd >= 1.0:
        blocks = [("encoder", list(enc.named_parameters()), args.lr)]
    else:
        blocks = [("embeddings", list(enc.embeddings.named_parameters()),
                   args.lr * args.llrd ** n_layers)]
        for i, layer in enumerate(enc.encoder.layer):
            blocks.append((f"layer{i}", list(layer.named_parameters()),
                           args.lr * args.llrd ** (n_layers - 1 - i)))
    for _name, params, lr in blocks:
        d, nd = split(params)
        if d:
            groups.append({"params": d, "lr": lr, "weight_decay": args.weight_decay})
        if nd:
            groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def accum_groups(n_micro: int, grad_accum: int) -> list[int]:
    """Micro-batch group sizes, one group per optimizer step, for one epoch.

    Each group's gradients are summed (every micro-batch's loss was already
    divided by ``grad_accum``), so a group of size ``grad_accum`` is the mean
    gradient over one effective batch of ``grad_accum x batch_size``. If the
    epoch's micro-batch count is not a multiple of ``grad_accum``, the trailing
    partial group still performs one step: its gradients are a correctly scaled
    (smaller) effective batch, and stepping it stops them from leaking into the
    next epoch's first group. ``grad_accum == 1`` yields one single micro-batch
    per step — exactly the archived loop.

    Example: 7 micro-batches at ``grad_accum == 3`` -> ``[3, 3, 1]``.
    """
    if n_micro <= 0:
        return []
    if grad_accum <= 1:
        return [1] * n_micro
    full, rem = divmod(n_micro, grad_accum)
    groups = [grad_accum] * full
    if rem:
        groups.append(rem)
    return groups


def accum_steps_per_epoch(n_micro: int, grad_accum: int) -> int:
    """Optimizer steps in one epoch: full groups plus any partial trailing group."""
    return len(accum_groups(n_micro, grad_accum))


# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if not args.init_from and not args.from_scratch:
        raise SystemExit("pass --init-from <mlm ckpt> or --from-scratch")
    engine.set_seed(args.seed)
    task = get_task(args.task)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = engine.JsonLogger(out_dir / "train_log.jsonl")

    corpus = Corpus(args.corpus)
    print(corpus, flush=True)
    print(f"task {task.name}: {task.classes} -> scored as {task.scored_classes}")

    if args.init_from:
        cfg = BinProvConfig.load(Path(args.init_from) / "binprov_config.json")
    else:
        cfg = BinProvConfig()
    if args.seq_bytes:
        cfg.seq_bytes = args.seq_bytes
    eval_bytes = args.eval_window_bytes or cfg.seq_bytes
    if args.dropout is not None:
        cfg.hidden_dropout_prob = args.dropout
        cfg.attention_probs_dropout_prob = args.dropout

    # ---- splits ---------------------------------------------------------
    saved = corpus.load_splits(args.splits)
    if args.split_mode == "program":
        train_bids, test_bids = saved["train"], saved["test"]
    elif args.split_mode == "binary":
        s = corpus.make_splits(train_ratio=0.8, seed=args.seed, group_by="binary")
        train_bids, test_bids = s["train"], s["test"]
    else:  # sequence: same binaries both sides, split the windows
        train_bids = test_bids = sorted(saved["train"] + saved["test"])

    train_bids, val_bids = carve_val(
        corpus, train_bids, args.val_frac,
        args.seed if args.val_seed is None else args.val_seed,
    )
    train_compiler = args.train_compiler or args.compiler
    train_index, train_labels = build_index(
        corpus, task, train_bids, args, cfg.seq_bytes, compiler=train_compiler,
        is_train=True,
    )
    test_index, test_labels = build_index(
        corpus, task, test_bids, args, eval_bytes,
        version=args.test_version or args.version,
    )
    if args.split_mode == "sequence":
        # One pool of windows, split 80/20 at random. Programs, and indeed
        # individual binaries, therefore appear on both sides. This is the
        # leakage arm; it is not a protocol anyone should report as a result.
        rng = np.random.default_rng(args.seed)
        pick = rng.random(len(train_labels)) < 0.8
        val_index, val_labels = train_index.subset(~pick), train_labels[~pick]
        test_index, test_labels = val_index, val_labels
        train_index, train_labels = train_index.subset(pick), train_labels[pick]
        val_bids = []
    if val_bids:
        val_index, val_labels = build_index(
            corpus, task, val_bids, args, eval_bytes, compiler=train_compiler
        )
    elif args.split_mode != "sequence":
        val_index, val_labels = test_index, test_labels
    val_index, val_labels = subsample(val_index, val_labels, args.val_max_seqs, args.seed)

    def counts(lab):
        return dict(zip(task.classes, np.bincount(lab, minlength=task.num_labels).tolist()))

    print(
        f"split={args.split_mode}  train {len(train_index):,} / val {len(val_index):,} "
        f"/ test {len(test_index):,} sequences", flush=True
    )
    print(f"  train {counts(train_labels)}\n  test  {counts(test_labels)}", flush=True)
    if len(train_index) == 0 or len(test_index) == 0:
        raise SystemExit("empty split after filtering")

    # ---- model ----------------------------------------------------------
    model = BinProvForProvenance(
        cfg, task.num_labels, pool=args.pool, taper=args.taper,
        pool_heads=args.pool_heads, mil_tau=args.mil_tau,
        classifier_dropout=args.classifier_dropout,
        label_smoothing=args.label_smoothing,
    )
    if args.init_from:
        missing = model.load_encoder_from_mlm(args.init_from, pos_extend=args.pos_extend)
        if missing:
            print(f"  WARNING: {len(missing)} encoder tensors missing")
        else:
            print(f"  warm-started encoder from {args.init_from}")
    print(describe(model))
    device, amp_dtype = engine.pick_device()
    model.to(device)

    collator = ClassificationCollator(cfg.seq_tokens)
    train_ds = (
        JitteredByteSequenceDataset(corpus, train_index, train_labels,
                                    jitter=args.jitter, seed=args.seed)
        if args.jitter else ByteSequenceDataset(corpus, train_index, train_labels)
    )
    train_loader = engine.make_loader(train_ds, collator, batch_size=args.batch_size,
                                      shuffle=True, workers=args.workers, drop_last=True)
    def eval_dataset(index, labels):
        """Plain windows, or the baseline's windows widened to the model's input."""
        if eval_bytes == cfg.seq_bytes:
            return ByteSequenceDataset(corpus, index, labels)
        return ContextWindowDataset(corpus, index, labels, length=cfg.seq_bytes)

    val_loader = engine.make_loader(
        eval_dataset(val_index, val_labels), collator,
        batch_size=args.batch_size * 2, shuffle=False, workers=max(2, args.workers // 2),
    )

    # ---- pair loader (auxiliary) ---------------------------------------
    pair_loader = None
    if args.pair_loss > 0 or args.use_pairs:
        pairs = build_function_pairs(
            corpus, train_bids, seq_len=cfg.seq_bytes, min_size=args.pair_min_size
        )
        if train_compiler:
            want = {"gcc": 0, "clang": 1}
            keep = np.isin(pairs.compiler, [want[c] for c in train_compiler])
            pairs = pairs.subset(keep)
        per_bid = np.full(len(corpus.records), -1, dtype=np.int64)
        for rec in corpus.records:
            lab = task.label_of(rec.labels)
            if lab is not None:
                per_bid[rec.bid] = lab
        lo_lab = per_bid[pairs.bid_lo.astype(np.int64)]
        hi_lab = per_bid[pairs.bid_hi.astype(np.int64)]
        ok = (lo_lab >= 0) & (hi_lab >= 0)
        pairs, lo_lab, hi_lab = pairs.subset(ok), lo_lab[ok], hi_lab[ok]
        print(f"  {len(pairs):,} matched O2/O3 function pairs from the training programs",
              flush=True)
        pair_ds = FunctionPairDataset(corpus, pairs, lo_lab, hi_lab,
                                      jitter=args.pair_jitter, seed=args.seed)
        pair_loader = torch.utils.data.DataLoader(
            pair_ds, collate_fn=collator, num_workers=max(2, args.workers // 2),
            batch_sampler=PairBatchSampler(len(pairs), args.pairs_per_batch,
                                           shuffle=True, seed=args.seed),
            pin_memory=True, persistent_workers=True,
        )

    optimizer = build_optimizer(model, args)
    micros_per_epoch = max(1, len(train_loader))
    steps_per_epoch = accum_steps_per_epoch(micros_per_epoch, args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    scheduler = cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    print(f"{args.epochs} epochs x {steps_per_epoch} steps = {total_steps} optimizer steps "
          f"(micro-batch {args.batch_size} x accum {args.grad_accum}, "
          f"effective batch {args.batch_size * args.grad_accum})", flush=True)
    engine.save_json(out_dir / "args.json", vars(args))

    def cycle(loader):
        while True:
            yield from loader

    pair_iter = cycle(pair_loader) if pair_loader is not None else None

    timer = engine.Timer(total_steps)
    best_val, best_epoch, step = -1.0, -1, 0
    best_state = None
    history = []
    start_epoch = 0
    if args.resume:
        state = engine.load_training_state(
            out_dir, model=model, optimizer=optimizer, scheduler=scheduler
        )
        if state is None:
            print("  --resume: no training_state.pt found; starting fresh", flush=True)
        else:
            start_epoch = int(state["epoch"]) + 1
            step = int(state["step"])
            if state.get("best") is not None:
                best_val = float(state["best"])
            extra = state.get("extra") or {}
            if isinstance(extra.get("best_epoch"), int):
                best_epoch = int(extra["best_epoch"])
            if isinstance(extra.get("history"), list):
                history.extend(extra["history"])
            print(f"  resumed from epoch {int(state['epoch'])} (step {step}, "
                  f"best val {pct(best_val)})", flush=True)
            if start_epoch >= args.epochs:
                print("  all epochs already trained; running the final val-selected "
                      "test", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, run_correct, run_seen, micro, run_pair = 0.0, 0, 0, 0, 0.0
        it = iter(train_loader)
        # One group per optimizer step. A trailing partial group still steps, so
        # gradients never survive past an epoch boundary.
        for group in accum_groups(micros_per_epoch, args.grad_accum):
            for _ in range(group):
                batch = next(it)
                ids = batch["input_ids"].to(device, non_blocking=True)
                attn = batch["attention_mask"].to(device, non_blocking=True)
                types = batch["token_type_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    out = model(ids, attn, types, labels=labels)
                    loss = out["loss"]
                    if pair_iter is not None:
                        pb = next(pair_iter)
                        p_ids = pb["input_ids"].to(device, non_blocking=True)
                        p_attn = pb["attention_mask"].to(device, non_blocking=True)
                        p_types = pb["token_type_ids"].to(device, non_blocking=True)
                        p_lab = pb["labels"].to(device, non_blocking=True)
                        p_out = model(p_ids, p_attn, p_types,
                                      labels=p_lab if args.pair_ce > 0 else None)
                        if args.pair_ce > 0:
                            loss = loss + args.pair_ce * p_out["loss"]
                        if args.pair_loss > 0:
                            pm = pairwise_margin_loss(p_out["logits"], p_lab, args.pair_margin)
                            loss = loss + args.pair_loss * pm
                            run_pair += float(pm.detach())
                # The whole *combined* loss is scaled, whatever mix of terms it is.
                (loss / args.grad_accum).backward()
                run_loss += float(out["loss"].detach())
                run_correct += int((out["logits"].argmax(-1) == labels).sum())
                run_seen += labels.numel()
                micro += 1
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                log.log(epoch=epoch, step=step, loss=run_loss / max(1, micro),
                        pair_loss=run_pair / max(1, micro),
                        train_acc=run_correct / max(1, run_seen),
                        lr=scheduler.get_last_lr()[-1], eta=timer.eta(step))
                run_loss, run_correct, run_seen, micro, run_pair = 0.0, 0, 0, 0, 0.0

        res = engine.predict(model, val_loader, device, amp_dtype,
                             num_labels=task.num_labels, progress_every=0)
        s = score(res["prob"], res["true"], task)
        history.append({"epoch": epoch, **{f"val_{k}": v for k, v in s.items() if k != "confusion"}})
        log.log(epoch=epoch, val_acc=s["accuracy"], val_bal_acc=s["balanced_accuracy"],
                elapsed=timer.elapsed())
        if s["accuracy"] > best_val:
            best_val, best_epoch = s["accuracy"], epoch
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            if args.resume:
                # Persist the val-selected weights now: an interrupted run must be
                # able to reload the best epoch even when that epoch was completed
                # by an earlier process.
                model.save(out_dir, extra={"task": task.name, "tag": args.tag})
            print(f"  new best val {pct(best_val)} (epoch {epoch})", flush=True)
        if args.resume:
            engine.save_training_state(
                out_dir, model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, step=step, best=best_val if best_val >= 0 else None,
                extra={"task": task.name, "best_epoch": best_epoch, "history": history},
            )

    # ---- final: the val-selected model on the test split ----------------
    if best_epoch >= 0:
        if best_state is not None:
            model.load_state_dict(best_state)
        elif args.resume and (out_dir / "model.pt").is_file():
            # The best epoch belonged to an earlier process; reload its weights.
            best_w = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(best_w)
            del best_w
    model.to(device)

    def run_test(ds, n_views=1):
        loader = engine.make_loader(ds, collator, batch_size=args.batch_size * 2,
                                    shuffle=False, workers=max(2, args.workers // 2))
        r = engine.predict(model, loader, device, amp_dtype,
                           num_labels=task.num_labels, progress_every=0)
        if n_views > 1:
            prob = r["prob"].reshape(-1, n_views, task.num_labels).mean(1)
            true = r["true"].reshape(-1, n_views)[:, 0]
            return prob, true
        return r["prob"], r["true"]

    prob, true = run_test(eval_dataset(test_index, test_labels))
    test_score = score(prob, true, task)
    print(f"\nTEST (val-selected epoch {best_epoch}): acc {pct(test_score['accuracy'])}  "
          f"bal {pct(test_score['balanced_accuracy'])}", flush=True)

    tta_score = None
    if args.tta != "none":
        mv = MultiViewByteSequenceDataset(
            corpus, test_index, test_labels, mode=args.tta,
            n_views=args.tta_views, span=args.tta_span,
        )
        p2, t2 = run_test(mv, n_views=args.tta_views)
        tta_score = score(p2, t2, task)
        print(f"TEST +TTA[{args.tta} x{args.tta_views}]: acc {pct(tta_score['accuracy'])}  "
              f"bal {pct(tta_score['balanced_accuracy'])}", flush=True)

    if args.save_probs:
        np.savez_compressed(
            out_dir / "probs.npz",
            prob=prob.astype(np.float32),
            true=true.astype(np.int16),
            bid=test_index.bid, start=test_index.start, length=test_index.length,
            marginal_map=np.asarray(task.marginal_map, dtype=np.int16),
            # A guard against silently ensembling two runs whose test sets are
            # not the same set of windows -- the trap that nearly turned
            # hypothesis 8 into a false positive.
            fingerprint=np.frombuffer(
                hashlib.sha1(test_index.bid.tobytes() + test_index.start.tobytes()).digest(),
                dtype=np.uint8,
            ),
        )
    if not args.no_save_weights:
        model.save(out_dir, extra={"task": task.name, "tag": args.tag})

    engine.save_json(out_dir / "result.json", {
        "tag": args.tag,
        "task": task.name,
        "scored_classes": list(task.scored_classes),
        "args": vars(args),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test": test_score,
        "test_tta": tta_score,
        "history": history,
        "n_train_sequences": int(len(train_index)),
        "wall_clock": timer.elapsed(),
    })
    log.close()
    print(f"wrote {out_dir}/result.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
