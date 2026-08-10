#!/usr/bin/env python3
"""Stage 1: pre-train the byte-level embedding model on masked language modeling.

This is §3.2 of the paper. The model learns the contextual semantics of machine
code by predicting randomly masked bytes — the point being that the same byte
means different things in different contexts (``c3`` is ``ret`` in one place and
part of a register encoding in another), so a bidirectional encoder has
something real to learn. No labels are involved.

Masking follows the paper's §4.1 settings: 20% of bytes selected, of which 50%
become ``<mask>`` and 50% become a random byte, re-drawn every epoch.

The pre-training set is the ``pretrain`` split written by ``build_corpus.py``:
one program per package, all its compiled variants — the paper's "at least one
binary (2x4 variants) from each software project". It is restricted to the
training binaries, so nothing from the test set is ever seen, even unlabelled.

Example (pin the GPU first — this machine is shared)::

    export CUDA_VISIBLE_DEVICES=7
    python scripts/pretrain_mlm.py --corpus data/corpus/x86_64 \\
        --out checkpoints/mlm_x86_64 --epochs 10 --batch-size 64
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
from binprov.data import IGNORE_INDEX, MLMCollator, PairedByteSequenceDataset  # noqa: E402
from binprov.model import (  # noqa: E402
    BinProvConfig,
    build_mlm_model,
    cosine_schedule_with_warmup,
    describe,
    param_groups,
)


def parse_args():
    ap = argparse.ArgumentParser(
        description="MLM pre-training of the BinProv embedding model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="checkpoint directory")
    ap.add_argument("--splits", default="default", help="split file name in the corpus")
    ap.add_argument(
        "--split-name",
        default="pretrain",
        help="which split to train on; 'train' uses every training binary "
        "(more data, slower) instead of the paper's one-program-per-package set",
    )

    arch = ap.add_argument_group("architecture")
    arch.add_argument("--seq-bytes", type=int, default=512)
    arch.add_argument("--layers", type=int, default=12)
    arch.add_argument("--hidden", type=int, default=768)
    arch.add_argument("--heads", type=int, default=12)
    arch.add_argument("--intermediate", type=int, default=3072)

    opt = ap.add_argument_group("optimization")
    opt.add_argument("--epochs", type=int, default=10)
    opt.add_argument("--max-steps", type=int, default=None, help="stop early (smoke tests)")
    opt.add_argument("--batch-size", type=int, default=64)
    opt.add_argument("--grad-accum", type=int, default=1)
    # RoBERTa's published 6e-4 goes with a batch of 8192. At the batch sizes that
    # fit one GPU here, that rate makes the loss *rise* back toward the unigram
    # solution once warmup ends -- measured, see docs/REPRODUCTION.md. Scale it
    # with the batch size (sqrt scaling): ~1e-4 at batch 256, lower for smaller.
    opt.add_argument("--lr", type=float, default=1e-4)
    opt.add_argument("--weight-decay", type=float, default=0.01)
    opt.add_argument("--warmup-ratio", type=float, default=0.06)
    opt.add_argument("--clip-grad", type=float, default=1.0)
    opt.add_argument("--mask-prob", type=float, default=0.20, help="paper §4.1")
    opt.add_argument("--mask-replace", type=float, default=0.5)
    opt.add_argument("--random-replace", type=float, default=0.5)

    data = ap.add_argument_group("data")
    data.add_argument("--max-seqs-per-binary", type=int, default=None)
    data.add_argument(
        "--pair-prob",
        type=float,
        default=0.0,
        help="probability of splicing two binaries into one sequence, which is "
        "what makes the segment embedding E_s carry information; the paper does "
        "not say it did this, so the default is off",
    )
    data.add_argument("--val-fraction", type=float, default=0.02)

    run = ap.add_argument_group("run")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--fp32", action="store_true", help="disable bf16 autocast")
    run.add_argument("--log-every", type=int, default=50)
    run.add_argument("--save-every-epoch", action="store_true")
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue from the training_state.pt in --out if one exists. The "
        "GPUs here are shared and unscheduled, so long runs should assume they "
        "will be interrupted",
    )
    return ap.parse_args()


def unigram_baseline(corpus, bids) -> tuple[float, float]:
    """Cross-entropy and top-1 accuracy of predicting the marginal byte.

    Without this reference an MLM loss curve is uninterpretable. Machine code is
    dominated by a few byte values -- 0x00 alone is ~15% of BinKit's x86_64
    ``.text`` -- so a model that has learned nothing but the marginal
    distribution still reports a respectable-looking ~15% masked-byte accuracy.
    Measured against this baseline, that number is revealed as zero progress.

    A weak encoder is not a cosmetic problem here: fine-tuning from an
    under-trained MLM checkpoint collapses outright (see docs/REPRODUCTION.md),
    so it is worth knowing early whether pre-training is actually working.
    """
    hist = np.zeros(256, dtype=np.int64)
    for b in bids:
        rec = corpus.records[b]
        chunk = np.asarray(corpus.text[rec.text_off : rec.text_off + rec.text_len])
        hist += np.bincount(chunk, minlength=256)
    total = hist.sum()
    if total == 0:
        return float("nan"), float("nan")
    p = hist / total
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum()), float(p.max())


def main() -> int:
    args = parse_args()
    engine.set_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = engine.JsonLogger(out_dir / "train_log.jsonl")

    corpus = Corpus(args.corpus)
    print(corpus)
    splits = corpus.load_splits(args.splits)
    if args.split_name not in splits:
        raise SystemExit(f"split {args.split_name!r} not in {sorted(splits)}")
    bids = splits[args.split_name]
    print(f"pre-training on split {args.split_name!r}: {len(bids)} binaries")

    index = corpus.sequences(
        seq_len=args.seq_bytes,
        level="binary",
        bids=bids,
        max_seqs_per_binary=args.max_seqs_per_binary,
    )
    print(f"{len(index):,} sequences of {args.seq_bytes} bytes")
    if len(index) == 0:
        raise SystemExit("no sequences; is the corpus empty or seq-bytes too large?")

    uni_loss, uni_acc = unigram_baseline(corpus, bids)
    print(
        f"unigram baseline: loss {uni_loss:.3f} nats, top-1 {100 * uni_acc:.2f}%\n"
        f"  -> MLM is only learning context once loss drops clearly below "
        f"{uni_loss:.3f}. A curve that flattens near it has learned the byte\n"
        f"     frequencies and nothing more, and will make a poor warm start."
    )

    # a held-out slice purely to watch MLM loss; it comes from the same binaries,
    # so it measures fit, not generalization
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(index))
    n_val = max(1, int(len(index) * args.val_fraction)) if args.val_fraction > 0 else 0
    val_mask = np.zeros(len(index), dtype=bool)
    val_mask[perm[:n_val]] = True
    train_index = index.subset(~val_mask)
    val_index = index.subset(val_mask) if n_val else None

    cfg = BinProvConfig(
        seq_bytes=args.seq_bytes,
        hidden_size=args.hidden,
        num_hidden_layers=args.layers,
        num_attention_heads=args.heads,
        intermediate_size=args.intermediate,
    )
    model = build_mlm_model(cfg)
    print(describe(model))

    device, amp_dtype = engine.pick_device(prefer_bf16=not args.fp32)
    if args.fp32:
        amp_dtype = None
    model.to(device)

    train_ds = PairedByteSequenceDataset(
        corpus, train_index, pair_prob=args.pair_prob, seed=args.seed
    )
    collator = MLMCollator(
        cfg.seq_tokens,
        mask_prob=args.mask_prob,
        mask_replace=args.mask_replace,
        random_replace=args.random_replace,
        seed=args.seed,
    )
    train_loader = engine.make_loader(
        train_ds, collator, batch_size=args.batch_size, shuffle=True,
        workers=args.workers, drop_last=True,
    )
    val_loader = None
    if val_index is not None:
        val_ds = PairedByteSequenceDataset(corpus, val_index, pair_prob=0.0, seed=args.seed + 1)
        # a fixed seed here makes the validation mask identical across epochs, so
        # the curve reflects the model rather than the mask
        val_collator = MLMCollator(
            cfg.seq_tokens, mask_prob=args.mask_prob,
            mask_replace=args.mask_replace, random_replace=args.random_replace, seed=7,
        )
        val_loader = engine.make_loader(
            val_ds, val_collator, batch_size=args.batch_size, shuffle=False, workers=2
        )

    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer = torch.optim.AdamW(
        param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.98), eps=1e-6
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    print(
        f"{args.epochs} epochs x {steps_per_epoch} steps = {total_steps} optimizer steps "
        f"(batch {args.batch_size} x accum {args.grad_accum})"
    )
    cfg.save(out_dir / "binprov_config.json")
    engine.save_json(out_dir / "pretrain_args.json", vars(args))
    # Persist the baseline next to the log so plot_training.py can draw it as a
    # reference line without being told: the MLM curve is only interpretable
    # against it.
    engine.save_json(
        out_dir / "baseline.json",
        {"unigram_loss_nats": uni_loss, "unigram_top1": uni_acc,
         "split": args.split_name, "num_sequences": len(index)},
    )

    timer = engine.Timer(total_steps)
    step = 0
    start_epoch = 0
    if args.resume:
        state = engine.load_training_state(
            out_dir, model=model, optimizer=optimizer, scheduler=scheduler
        )
        if state is None:
            print("  --resume: no training_state.pt found, starting fresh")
        else:
            # the snapshot is written after an epoch completes, so restart at the next
            start_epoch = state["epoch"] + 1
            step = state["step"]
            print(f"  resumed from epoch {state['epoch']} (step {step})")
            if start_epoch >= args.epochs:
                print(
                    f"  nothing to do: already finished epoch {state['epoch']} of "
                    f"{args.epochs}. Raise --epochs to train further."
                )
                save(model, out_dir, cfg)
                log.close()
                return 0

    stop = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        run_loss, run_correct, run_total, micro = 0.0, 0, 0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device, non_blocking=True)
            attn = batch["attention_mask"].to(device, non_blocking=True)
            types = batch["token_type_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(
                    input_ids=ids, attention_mask=attn, token_type_ids=types, labels=labels
                )
                loss = out.loss
            (loss / args.grad_accum).backward()

            with torch.no_grad():
                sel = labels != IGNORE_INDEX
                if sel.any():
                    pred = out.logits[sel].argmax(-1)
                    run_correct += int((pred == labels[sel]).sum())
                    run_total += int(sel.sum())
            run_loss += float(loss.detach())
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
                        epoch=epoch,
                        step=step,
                        loss=run_loss / max(1, micro),
                        mlm_acc=run_correct / max(1, run_total),
                        lr=scheduler.get_last_lr()[0],
                        eta=timer.eta(step),
                    )
                    run_loss, run_correct, run_total, micro = 0.0, 0, 0, 0
                if args.max_steps and step >= args.max_steps:
                    stop = True
                    break

        if val_loader is not None:
            vl, va = evaluate_mlm(model, val_loader, device, amp_dtype)
            log.log(epoch=epoch, val_loss=vl, val_mlm_acc=va, elapsed=timer.elapsed())
        if args.save_every_epoch:
            save(model, out_dir / f"epoch{epoch}", cfg)
        # Snapshot every epoch regardless of --resume: the cost is one ~1 GB
        # write, and the alternative is discovering after six hours that a
        # co-tenant took the GPU and nothing was kept.
        engine.save_training_state(
            out_dir, model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, step=step,
        )
        # also keep the HF-format weights current, so fine-tuning can start from
        # a partially pre-trained encoder without waiting for the whole run
        save(model, out_dir, cfg)
        if stop:
            break

    save(model, out_dir, cfg)
    print(f"\nsaved MLM checkpoint to {out_dir}")
    print("next: python scripts/finetune.py --corpus ... --task opt4 --init-from", out_dir)
    log.close()
    return 0


@torch.no_grad()
def evaluate_mlm(model, loader, device, amp_dtype) -> tuple[float, float]:
    model.eval()
    total_loss, n_batches, correct, total = 0.0, 0, 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        types = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            out = model(input_ids=ids, attention_mask=attn, token_type_ids=types, labels=labels)
        total_loss += float(out.loss)
        n_batches += 1
        sel = labels != IGNORE_INDEX
        if sel.any():
            correct += int((out.logits[sel].argmax(-1) == labels[sel]).sum())
            total += int(sel.sum())
    model.train()
    return total_loss / max(1, n_batches), correct / max(1, total)


def save(model, path: Path, cfg: BinProvConfig) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    cfg.save(path / "binprov_config.json")


if __name__ == "__main__":
    raise SystemExit(main())
