#!/usr/bin/env python3
"""Stage 2: fine-tune the pre-trained encoder for one provenance task.

This is §3.3 of the paper: stack the two-layer fully connected classifier on the
embedding model and train the whole thing end to end, warm-starting the encoder
from the MLM checkpoint. The paper trains one specialised classifier per task
rather than one 8-way model, because task decomposition measured better.

Tasks (see ``binprov/provenance.py``): compiler, opt_hl, opt4, opt_o0o1,
opt_o2o3, arch.

Example::

    export CUDA_VISIBLE_DEVICES=7
    python scripts/finetune.py --corpus data/corpus/x86_64 --task opt4 \\
        --init-from checkpoints/mlm_x86_64 --out checkpoints/opt4 --epochs 5

The ``--from-scratch`` flag skips the warm start, which is the ablation that
shows what MLM pre-training actually contributed.
"""

from __future__ import annotations

import argparse
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
from binprov.metrics import accuracy, per_class_prf, pct  # noqa: E402
from binprov.model import (  # noqa: E402
    BinProvConfig,
    BinProvForProvenance,
    cosine_schedule_with_warmup,
    describe,
    param_groups,
)
from binprov.provenance import get_task  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune BinProv for one provenance task",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--task", required=True, help="compiler | opt_hl | opt4 | opt_o0o1 | opt_o2o3 | arch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="default")
    ap.add_argument(
        "--init-from",
        default=None,
        help="MLM checkpoint directory to warm-start the encoder from",
    )
    ap.add_argument(
        "--from-scratch",
        action="store_true",
        help="random init instead of warm start (pre-training ablation)",
    )

    sel = ap.add_argument_group("data selection")
    sel.add_argument("--arch", nargs="*", default=None, help="restrict to these architectures")
    sel.add_argument("--extra", nargs="*", default=None, help="restrict to e.g. normal")
    sel.add_argument("--obfuscation", nargs="*", default=None)
    sel.add_argument("--seq-bytes", type=int, default=None, help="default: from the checkpoint")
    sel.add_argument("--max-seqs-per-binary", type=int, default=None)
    sel.add_argument(
        "--balance",
        action="store_true",
        help="subsample sequences so all classes have equal count. The paper "
        "reports plain accuracy on the grounds that its data is balanced; if "
        "your corpus is not, either use this or read the per-class table",
    )

    arch = ap.add_argument_group("architecture (ignored when --init-from is given)")
    arch.add_argument("--layers", type=int, default=12)
    arch.add_argument("--hidden", type=int, default=768)
    arch.add_argument("--heads", type=int, default=12)
    arch.add_argument("--intermediate", type=int, default=3072)
    arch.add_argument("--pool", choices=["border", "mean", "cls"], default="border",
                      help="'border' is the paper's border-weakening first layer")
    arch.add_argument("--taper", type=int, default=32, help="border taper width in tokens")
    arch.add_argument("--freeze-encoder", action="store_true", help="train the head only")

    opt = ap.add_argument_group("optimization")
    opt.add_argument("--epochs", type=int, default=5)
    opt.add_argument("--max-steps", type=int, default=None)
    opt.add_argument("--batch-size", type=int, default=64)
    opt.add_argument("--grad-accum", type=int, default=1)
    opt.add_argument("--lr", type=float, default=3e-5, help="encoder lr")
    opt.add_argument("--head-lr", type=float, default=1e-4, help="classifier lr")
    opt.add_argument("--weight-decay", type=float, default=0.01)
    opt.add_argument("--warmup-ratio", type=float, default=0.06)
    opt.add_argument("--clip-grad", type=float, default=1.0)

    run = ap.add_argument_group("run")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--fp32", action="store_true")
    run.add_argument("--log-every", type=int, default=50)
    run.add_argument("--eval-max-seqs", type=int, default=None,
                     help="cap test sequences during per-epoch eval (full eval: evaluate.py)")
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue from the training_state.pt in --out if one exists",
    )
    return ap.parse_args()


def build_split_data(corpus, task, bids, args, seq_bytes, *, is_train: bool, rng_seed: int):
    """Sequences + labels for one split, filtered to the task's scope."""
    keep = corpus.filter_bids(
        arch=args.arch, extra=args.extra, obfuscation=args.obfuscation
    )
    keep_set = set(keep)
    bids = [b for b in bids if b in keep_set]
    index = corpus.sequences(
        seq_len=seq_bytes,
        level="binary",
        bids=bids,
        max_seqs_per_binary=args.max_seqs_per_binary,
    )
    labels = sequence_labels(corpus, index, task)
    index, labels = drop_unlabelled(index, labels)

    if is_train and args.balance and len(labels):
        counts = np.bincount(labels, minlength=task.num_labels)
        target = counts[counts > 0].min()
        rng = np.random.default_rng(rng_seed)
        pick = np.concatenate(
            [rng.choice(np.flatnonzero(labels == c), size=target, replace=False)
             for c in range(task.num_labels) if counts[c] > 0]
        )
        pick.sort()
        mask = np.zeros(len(labels), dtype=bool)
        mask[pick] = True
        index, labels = index.subset(mask), labels[mask]
    return index, labels


def warn_if_collapsed(res, task, args) -> bool:
    """Detect a degenerate classifier and say what usually causes it.

    A collapsed model — near-constant logits, so every sequence gets the same
    class — looks exactly like "the task is hard" in an accuracy number, and it
    is the failure this pipeline hits most easily. Measured on a small corpus
    (logs of `scripts/run_smoke.sh`), the 12-layer encoder collapses when the
    learning rate is too high *and* when it is warm-started from an
    under-trained MLM checkpoint, while the same config from random init at the
    same learning rate trains fine. So a collapse is worth naming explicitly
    rather than leaving in a table.
    """
    pred_counts = np.bincount(res["pred"], minlength=task.num_labels)
    true_counts = np.bincount(res["true"], minlength=task.num_labels)
    n = max(1, int(pred_counts.sum()))
    dominant = pred_counts.max() / n
    majority_baseline = true_counts.max() / n
    acc = accuracy(res["pred"], res["true"])
    confidence = float(res["prob"].max(axis=1).mean())

    # Both conditions are required. Predicting one class for almost everything is
    # only a problem if it also buys nothing over always guessing that class --
    # and high confidence on its own is a sign of a *good* model, not a collapsed
    # one, so it is reported but never used as a trigger.
    if dominant < 0.95 or acc > majority_baseline + 0.02:
        return False

    print("\n  WARNING: this classifier looks degenerate.")
    print(
        f"    {pct(dominant)} of test sequences predicted as "
        f"{task.classes[int(pred_counts.argmax())]!r}, and accuracy {pct(acc)} is "
        f"no better than always guessing the majority class ({pct(majority_baseline)}).\n"
        f"    Mean top-class probability {confidence:.3f}"
        f"{' (near chance -- logits have collapsed)' if confidence < 0.7 else ''}."
    )
    print("    Most likely causes, in the order worth checking:")
    if args.init_from:
        print(
            "      1. The MLM checkpoint is under-trained. A weak warm start is "
            "worse than\n         none here: re-run pretrain_mlm.py for longer, "
            "or compare against\n         --from-scratch to see whether the "
            "warm start is helping at all."
        )
    print(
        f"      2. Learning rate too high for a {args.layers}-layer encoder "
        f"(--lr {args.lr:g}).\n         Deep post-LayerNorm encoders collapse "
        "above roughly 5e-5; try 1e-5."
    )
    print(
        "      3. Model too large for the corpus. Try --layers 4 --hidden 256 "
        "on a\n         corpus of a few thousand sequences."
    )
    return True


def main() -> int:
    args = parse_args()
    if not args.init_from and not args.from_scratch:
        raise SystemExit(
            "pass --init-from <mlm checkpoint> for the paper's transfer-learning "
            "setup, or --from-scratch to deliberately skip pre-training"
        )
    engine.set_seed(args.seed)
    task = get_task(args.task)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = engine.JsonLogger(out_dir / "train_log.jsonl")

    corpus = Corpus(args.corpus)
    print(corpus)
    print(f"task {task.name}: {task.classes}  ({task.note})")
    splits = corpus.load_splits(args.splits)

    # architecture comes from the pre-trained checkpoint so the encoder matches
    if args.init_from:
        cfg_path = Path(args.init_from) / "binprov_config.json"
        if not cfg_path.exists():
            raise SystemExit(f"{cfg_path} missing; was --init-from produced by pretrain_mlm.py?")
        cfg = BinProvConfig.load(cfg_path)
        if args.seq_bytes and args.seq_bytes != cfg.seq_bytes:
            raise SystemExit(
                f"--seq-bytes {args.seq_bytes} conflicts with the checkpoint's "
                f"{cfg.seq_bytes}; position embeddings would not match"
            )
    else:
        cfg = BinProvConfig(
            seq_bytes=args.seq_bytes or 512,
            hidden_size=args.hidden,
            num_hidden_layers=args.layers,
            num_attention_heads=args.heads,
            intermediate_size=args.intermediate,
        )

    train_index, train_labels = build_split_data(
        corpus, task, splits["train"], args, cfg.seq_bytes, is_train=True, rng_seed=args.seed
    )
    test_index, test_labels = build_split_data(
        corpus, task, splits["test"], args, cfg.seq_bytes, is_train=False, rng_seed=args.seed
    )
    if len(train_index) == 0:
        raise SystemExit("no training sequences after filtering; check --arch/--extra and the task")
    print(
        f"train {len(train_index):,} sequences / test {len(test_index):,} sequences\n"
        f"  train class counts: "
        f"{dict(zip(task.classes, np.bincount(train_labels, minlength=task.num_labels)))}\n"
        f"  test  class counts: "
        f"{dict(zip(task.classes, np.bincount(test_labels, minlength=task.num_labels)))}"
    )

    if args.eval_max_seqs and len(test_index) > args.eval_max_seqs:
        rng = np.random.default_rng(args.seed)
        sub = np.zeros(len(test_index), dtype=bool)
        sub[rng.choice(len(test_index), args.eval_max_seqs, replace=False)] = True
        test_index, test_labels = test_index.subset(sub), test_labels[sub]
        print(f"  (per-epoch eval capped at {len(test_index):,} sequences)")

    model = BinProvForProvenance(cfg, task.num_labels, pool=args.pool, taper=args.taper)
    if args.init_from:
        missing = model.load_encoder_from_mlm(args.init_from)
        if missing:
            print(f"  WARNING: {len(missing)} encoder tensors missing from the checkpoint")
        else:
            print(f"  warm-started encoder from {args.init_from}")
    else:
        print("  training from scratch (no MLM pre-training)")
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
        print("  encoder frozen; training the classifier head only")
    print(describe(model))

    device, amp_dtype = engine.pick_device(prefer_bf16=not args.fp32)
    if args.fp32:
        amp_dtype = None
    model.to(device)

    collator = ClassificationCollator(cfg.seq_tokens)
    train_loader = engine.make_loader(
        ByteSequenceDataset(corpus, train_index, train_labels),
        collator, batch_size=args.batch_size, shuffle=True, workers=args.workers, drop_last=True,
    )
    test_loader = engine.make_loader(
        ByteSequenceDataset(corpus, test_index, test_labels),
        collator, batch_size=args.batch_size * 2, shuffle=False, workers=max(2, args.workers // 2),
    )

    # Two learning rates: the pre-trained encoder needs gentler updates than a
    # randomly initialised head (paper §3.3: "the parameters of classification
    # model ... need to be adjusted significantly during the fine-tuning phase").
    # Within each, biases and normalisation weights are exempt from weight decay.
    optimizer = torch.optim.AdamW(
        [
            {**g, "lr": args.lr} for g in param_groups(model.encoder, args.weight_decay)
        ]
        + [
            {**g, "lr": args.head_lr}
            for g in param_groups(
                torch.nn.ModuleList(
                    [m for m in (model.pool, model.dropout, model.classifier) if m is not None]
                ),
                args.weight_decay,
            )
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    print(f"{args.epochs} epochs x {steps_per_epoch} steps = {total_steps} optimizer steps")
    engine.save_json(out_dir / "finetune_args.json", vars(args))

    timer = engine.Timer(total_steps)
    best_acc, step, stop = -1.0, 0, False
    start_epoch = 0
    if args.resume:
        state = engine.load_training_state(
            out_dir, model=model, optimizer=optimizer, scheduler=scheduler
        )
        if state is None:
            print("  --resume: no training_state.pt found, starting fresh")
        else:
            start_epoch = state["epoch"] + 1
            step = state["step"]
            if state["best"] is not None:
                best_acc = state["best"]
            print(
                f"  resumed from epoch {state['epoch']} (step {step}, "
                f"best {pct(best_acc)})"
            )
            if start_epoch >= args.epochs:
                print(f"  already finished {args.epochs} epochs; raise --epochs to continue")
                log.close()
                return 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, run_correct, run_seen, micro = 0.0, 0, 0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            types = batch["token_type_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(ids, attn, types, labels=labels)
            (out["loss"] / args.grad_accum).backward()

            run_loss += float(out["loss"].detach())
            run_correct += int((out["logits"].argmax(-1) == labels).sum())
            run_seen += labels.numel()
            micro += 1

            if micro % args.grad_accum == 0:
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % args.log_every == 0:
                    log.log(
                        epoch=epoch, step=step,
                        loss=run_loss / max(1, micro),
                        train_acc=run_correct / max(1, run_seen),
                        lr=scheduler.get_last_lr()[0],
                        eta=timer.eta(step),
                    )
                    run_loss, run_correct, run_seen, micro = 0.0, 0, 0, 0
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break

        res = engine.predict(
            model, test_loader, device, amp_dtype, num_labels=task.num_labels, progress_every=0
        )
        acc = accuracy(res["pred"], res["true"])
        log.log(epoch=epoch, test_seq_acc=acc, elapsed=timer.elapsed())
        if acc > best_acc:
            best_acc = acc
            model.save(out_dir, extra={"task": task.name, "classes": list(task.classes),
                                       "epoch": epoch, "test_seq_acc": acc})
            print(f"  new best -> saved to {out_dir}")
        # Two different snapshots, deliberately: model.pt above is the *best*
        # epoch and is what evaluate.py loads; training_state.pt is the *latest*
        # and exists only so an interrupted run can pick up where it stopped.
        engine.save_training_state(
            out_dir, model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, step=step, best=best_acc,
            extra={"task": task.name},
        )
        if stop:
            break

    print(f"\nbest sequence-level test accuracy: {pct(best_acc)}")

    # a final per-class breakdown, which is where O2/O3 confusion shows up
    model_best, _ = BinProvForProvenance.load(out_dir)
    model_best.to(device)
    res = engine.predict(
        model_best, test_loader, device, amp_dtype, num_labels=task.num_labels, progress_every=0
    )
    prf = per_class_prf(res["pred"], res["true"], task.num_labels)
    for i, cls in enumerate(task.classes):
        print(
            f"  {cls:>6}: P={pct(prf['precision'][i])} R={pct(prf['recall'][i])} "
            f"F1={pct(prf['f1'][i])} n={prf['support'][i]}"
        )
    warn_if_collapsed(res, task, args)
    engine.save_json(
        out_dir / "finetune_result.json",
        {
            "task": task.name,
            "classes": list(task.classes),
            "best_seq_accuracy": best_acc,
            "precision": prf["precision"],
            "recall": prf["recall"],
            "f1": prf["f1"],
            "support": prf["support"],
            "confusion": prf["confusion"],
        },
    )
    print(f"\nnext: python scripts/evaluate.py --corpus {args.corpus} "
          f"--ckpt {task.name}={out_dir}   # adds function/binary-level voting")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
