#!/usr/bin/env python3
"""Stage 3+4: evaluate fine-tuned models and reproduce the paper's tables.

Runs each checkpoint over the test split at every granularity and applies the
majority voting of §3.4, then writes the paper's tables as markdown plus a JSON
dump of everything.

Which table needs which checkpoints:

    Table 3  compiler + opt_hl        sequence level, plus the joint "Overall"
    Table 4  opt4 + opt_o0o1 + opt_o2o3   sequence level
    Table 5  opt4                     per-class P/R/F1, split by compiler
    Table 6  compiler + opt_hl + opt_o2o3   function and binary level voting

Example::

    export CUDA_VISIBLE_DEVICES=7
    python scripts/evaluate.py --corpus data/corpus/x86_64 \\
        --ckpt compiler=checkpoints/compiler \\
        --ckpt opt_hl=checkpoints/opt_hl \\
        --ckpt opt4=checkpoints/opt4 \\
        --ckpt opt_o0o1=checkpoints/opt_o0o1 \\
        --ckpt opt_o2o3=checkpoints/opt_o2o3 \\
        --out results/x86_64

Missing checkpoints are simply skipped, and the tables they feed are omitted —
so a partial run is useful.

A note on the function level. The paper cuts a binary into fixed 512-byte
sequences and then votes "over the sequences belonging to the same function",
without saying how sequences are assigned to functions. Most functions are
shorter than 512 bytes, so a non-overlapping cut gives one sequence per function
and voting would be a no-op — yet the paper reports a large gain there. This
script therefore cuts *within* each function with an overlapping stride
(``--function-stride``, default seq_len/4), which yields several sequences for a
medium function and no sequence that straddles a boundary. See
docs/REPRODUCTION.md.
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
from binprov.metrics import (  # noqa: E402
    accuracy,
    confusion_as_markdown,
    markdown_table,
    pct,
    per_class_prf,
)
from binprov.model import BinProvForProvenance  # noqa: E402
from binprov.provenance import get_task  # noqa: E402
from binprov.vote import vote_report  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(
        description="Evaluate BinProv checkpoints and emit the paper's tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--corpus", required=True)
    ap.add_argument(
        "--ckpt",
        action="append",
        default=[],
        metavar="TASK=DIR",
        help="repeatable, e.g. --ckpt opt4=checkpoints/opt4",
    )
    ap.add_argument("--out", default="results/eval", help="output directory")
    ap.add_argument("--splits", default="default")
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--arch", nargs="*", default=None)
    ap.add_argument("--extra", nargs="*", default=None)
    ap.add_argument("--obfuscation", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument(
        "--levels",
        nargs="+",
        default=["binary", "function"],
        choices=["binary", "function"],
        help="'binary' gives the sequence-level numbers and binary-level voting; "
        "'function' gives function-level voting",
    )
    ap.add_argument(
        "--function-stride",
        type=int,
        default=None,
        help="stride within a function; default seq_len//4 (see the note above)",
    )
    ap.add_argument(
        "--function-context",
        action="store_true",
        help="at the function level, give a function shorter than one sequence a "
        "full-length window centred on it instead of its own bytes padded. In "
        "BinKit 81.5%% of functions are shorter than 512 bytes (median 127), so "
        "without this the function level measures a harder input distribution "
        "rather than a voted one",
    )
    ap.add_argument("--vote", choices=["hard", "soft"], default="hard",
                    help="'hard' counts votes, as the paper does")
    ap.add_argument("--max-seqs-per-binary", type=int, default=None)
    return ap.parse_args()


def load_checkpoints(specs) -> dict[str, Path]:
    out = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--ckpt expects TASK=DIR, got {spec!r}")
        task, path = spec.split("=", 1)
        p = Path(path)
        if not (p / "model.pt").exists():
            print(f"  skipping {task}: no model.pt in {p}")
            continue
        out[task] = p
    return out


def evaluate_one(corpus, task, ckpt_dir, args, device, amp_dtype, bids, level):
    """Predict over one granularity and return the sequence + voted results."""
    stride = None
    if level == "function":
        stride = args.function_stride

    model, head = BinProvForProvenance.load(ckpt_dir)
    if head.get("task") and head["task"] != task.name:
        raise SystemExit(
            f"{ckpt_dir} was trained for task {head['task']!r}, not {task.name!r}"
        )
    seq_bytes = model.cfg.seq_bytes
    if level == "function" and stride is None:
        stride = max(1, seq_bytes // 4)

    index = corpus.sequences(
        seq_len=seq_bytes,
        stride=stride,
        level=level,
        bids=bids,
        max_seqs_per_binary=args.max_seqs_per_binary,
        function_context=args.function_context,
    )
    labels = sequence_labels(corpus, index, task)
    index, labels = drop_unlabelled(index, labels)
    if len(index) == 0:
        print(f"  no {level}-level sequences for {task.name}")
        return None

    model.to(device).eval()
    loader = engine.make_loader(
        ByteSequenceDataset(corpus, index, labels),
        ClassificationCollator(model.cfg.seq_tokens),
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.workers,
    )
    print(f"  {task.name} @ {level}: {len(index):,} sequences", flush=True)
    res = engine.predict(model, loader, device, amp_dtype, num_labels=task.num_labels)

    groups = index.group_ids()
    report = vote_report(
        groups, res["pred"], res["true"], task.num_labels,
        probs=res["prob"], mode=args.vote,
    )
    prf = per_class_prf(res["pred"], res["true"], task.num_labels)
    # Majority-class rate of the test set. Not the same as 1/num_classes: the
    # corpus is balanced per binary, but O0 binaries are longer than O1 ones
    # (paper Figure 2b), so O0 contributes ~1.6x as many sequences as O1 and the
    # trivial baseline for the 4-way task is ~32%, not 25%.
    counts = np.bincount(res["true"], minlength=task.num_labels)
    majority = float(counts.max() / max(1, counts.sum()))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "level": level,
        "seq_accuracy": report["seq_accuracy"],
        "vote_accuracy": report["vote_accuracy"],
        "majority_baseline": majority,
        "num_sequences": report["num_sequences"],
        "num_groups": report["num_groups"],
        "mean_seqs_per_group": report["mean_seqs_per_group"],
        "median_seqs_per_group": report["median_seqs_per_group"],
        "per_class": {
            "precision": prf["precision"], "recall": prf["recall"],
            "f1": prf["f1"], "support": prf["support"], "confusion": prf["confusion"],
        },
        # kept in memory (not serialised) for the cross-task tables below
        "_pred": res["pred"],
        "_true": res["true"],
        "_bid": index.bid,
    }


def main() -> int:
    args = parse_args()
    ckpts = load_checkpoints(args.ckpt)
    if not ckpts:
        raise SystemExit("no usable checkpoints given; pass --ckpt TASK=DIR")

    corpus = Corpus(args.corpus)
    print(corpus)
    splits = corpus.load_splits(args.splits)
    bids = splits[args.split_name]
    keep = set(corpus.filter_bids(arch=args.arch, extra=args.extra, obfuscation=args.obfuscation))
    bids = [b for b in bids if b in keep]
    print(f"evaluating on {len(bids)} binaries from split {args.split_name!r}")
    if not bids:
        raise SystemExit("no binaries after filtering")

    n_func = sum(1 for b in bids if corpus.records[b].functions)
    if "function" in args.levels and n_func == 0:
        print("  no function boundaries in the test binaries; dropping the function level")
        args.levels = [lv for lv in args.levels if lv != "function"]
    elif "function" in args.levels and n_func < len(bids):
        print(f"  note: only {n_func}/{len(bids)} test binaries carry function symbols")

    device, amp_dtype = engine.pick_device(prefer_bf16=not args.fp32)
    if args.fp32:
        amp_dtype = None

    results: dict[str, dict[str, dict]] = {}
    for task_name, ckpt_dir in ckpts.items():
        task = get_task(task_name)
        results[task_name] = {}
        for level in args.levels:
            r = evaluate_one(
                corpus, task, ckpt_dir, args, device, amp_dtype, bids, level
            )
            if r is not None:
                results[task_name][level] = r

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = render_tables(corpus, results, args)
    (out_dir / "tables.md").write_text(report)
    engine.save_json(
        out_dir / "results.json",
        {
            task: {
                lv: {k: v for k, v in r.items() if not k.startswith("_")}
                for lv, r in per_level.items()
            }
            for task, per_level in results.items()
        },
    )
    print("\n" + report)
    print(f"written to {out_dir}/tables.md and {out_dir}/results.json")
    return 0


def _seq(results, task, level="binary", key="seq_accuracy"):
    r = results.get(task, {}).get(level)
    return None if r is None else r[key]


def render_tables(corpus, results, args) -> str:
    """Assemble the paper's tables from whatever checkpoints were evaluated."""
    out = [
        "# BinProv evaluation",
        "",
        f"corpus: `{args.corpus}`  ({len(corpus)} binaries)",
        f"split: `{args.split_name}`   voting: {args.vote}",
        "",
        "Sequence-level rows correspond to the paper's \"BinProv w/o\" "
        "(no majority voting); function/binary rows are \"BinProv w/\".",
        "",
    ]

    def fmt(x):
        return "-" if x is None else pct(x)

    # ---- Table 3: coarse tasks, sequence level ---------------------------
    if any(t in results for t in ("compiler", "opt_hl")):
        rows = [
            ["Compiler (GCC/Clang)", fmt(_seq(results, "compiler")),
             fmt(_seq(results, "compiler", key="majority_baseline"))],
            ["Opt level (High/Low)", fmt(_seq(results, "opt_hl")),
             fmt(_seq(results, "opt_hl", key="majority_baseline"))],
        ]
        joint = joint_accuracy(results, "compiler", "opt_hl")
        rows.append(["Overall (both correct)", fmt(joint), "-"])
        out += [
            "## Table 3 — basic tasks, sequence level",
            "",
            markdown_table(["Basic task", "BinProv w/o", "majority baseline"], rows),
            "",
        ]

    # ---- Table 4: fine-grained optimization ------------------------------
    if any(t in results for t in ("opt4", "opt_o0o1", "opt_o2o3")):
        rows = [
            [name, fmt(_seq(results, t)), fmt(_seq(results, t, key="majority_baseline"))]
            for name, t in (("O0/O1/O2/O3", "opt4"), ("O0/O1", "opt_o0o1"),
                            ("O2/O3", "opt_o2o3"))
        ]
        out += [
            "## Table 4 — fine-grained optimization level, sequence level",
            "",
            markdown_table(["Opt level", "BinProv w/o", "majority baseline"], rows),
            "",
            "The majority baseline is above 1/num_classes because the corpus is "
            "balanced per *binary*, not per *sequence*: O0 binaries are longer "
            "than O1 ones (paper Figure 2b), so they contribute more sequences.",
            "",
        ]

    # ---- Table 5: per-class P/R/F1 for opt4, split by compiler -----------
    if "opt4" in results and "binary" in results["opt4"]:
        out += ["## Table 5 — optimization level per class, split by compiler", ""]
        out.append(per_compiler_prf(corpus, results["opt4"]["binary"]))
        out += [
            "",
            confusion_as_markdown(
                results["opt4"]["binary"]["per_class"]["confusion"], get_task("opt4").classes
            ),
            "",
        ]

    # ---- Table 6: voting at function and binary level --------------------
    voting_tasks = [
        ("Compiler (GCC/Clang)", "compiler"),
        ("Opt level (High/Low)", "opt_hl"),
        ("Opt level (O2/O3)", "opt_o2o3"),
        ("Opt level (O0/O1/O2/O3)", "opt4"),
    ]
    have_any = any(
        t in results and lv in results[t] for _, t in voting_tasks for lv in ("function", "binary")
    )
    if have_any:
        rows = []
        for label, task in voting_tasks:
            if task not in results:
                continue
            rows.append(
                [
                    label,
                    fmt(_seq(results, task, "binary", "seq_accuracy")),
                    fmt(_seq(results, task, "function", "vote_accuracy")),
                    fmt(_seq(results, task, "binary", "vote_accuracy")),
                ]
            )
        out += [
            "## Table 6 — joint inference by majority voting",
            "",
            markdown_table(
                ["Provenance", "sequence", "function (voted)", "binary (voted)"], rows
            ),
            "",
        ]
        # how much there was to vote over, which explains the gain
        rows = []
        for label, task in voting_tasks:
            for level in ("function", "binary"):
                r = results.get(task, {}).get(level)
                if r:
                    rows.append(
                        [
                            label, level, f"{r['num_groups']:,}",
                            f"{r['mean_seqs_per_group']:.1f}",
                            f"{r['median_seqs_per_group']:.0f}",
                        ]
                    )
        if rows:
            out += [
                "### voting group sizes",
                "",
                markdown_table(
                    ["Provenance", "level", "#groups", "mean seqs/group", "median"], rows
                ),
                "",
            ]
            # A "voted" number over groups of one sequence is just the
            # sequence-level number wearing a different hat. Say so, loudly,
            # rather than letting it be read as a voting result.
            degenerate = sorted(
                {
                    task
                    for _, task in voting_tasks
                    if (r := results.get(task, {}).get("function"))
                    and r["median_seqs_per_group"] <= 1
                }
            )
            if degenerate:
                out += [
                    f"> **Caveat on the function column.** For "
                    f"{', '.join(degenerate)} the median function holds only one "
                    "sequence, so that column is not a voting result. Most "
                    "functions are shorter than one 512-byte sequence, so what it "
                    "actually measures is single-sequence accuracy on "
                    "function-sized inputs — which carry less context than a full "
                    "window and are therefore *harder*. That is why the number can "
                    "come out below the sequence-level column instead of above it. "
                    "Lower `--function-stride` to get real votes on the larger "
                    "functions. See docs/REPRODUCTION.md, interpretation 1.",
                    "",
                ]

    return "\n".join(out)


def joint_accuracy(results, task_a: str, task_b: str) -> float | None:
    """Accuracy when both sub-tasks must be right — the paper's "Overall" row.

    Only defined when both tasks were scored over the same sequence set, which
    holds for compiler and opt_hl since neither filters binaries out.
    """
    a = results.get(task_a, {}).get("binary")
    b = results.get(task_b, {}).get("binary")
    if a is None or b is None:
        return None
    if len(a["_pred"]) != len(b["_pred"]) or not np.array_equal(a["_bid"], b["_bid"]):
        print(
            "  note: skipping the joint 'Overall' row — the two tasks did not "
            "score the same sequence set"
        )
        return None
    ok = (a["_pred"] == a["_true"]) & (b["_pred"] == b["_true"])
    return float(ok.mean())


def per_compiler_prf(corpus, result) -> str:
    """Table 5: precision/recall/F1 for each optimization level, per compiler."""
    task = get_task("opt4")
    compiler_of_bid = {r.bid: r.labels.get("compiler") for r in corpus.records}
    compilers = [c for c in ("gcc", "clang") if any(
        compiler_of_bid.get(int(b)) == c for b in result["_bid"]
    )]
    if not compilers:
        return "_(no compiler labels available)_"

    header = ["Metric"] + [f"{c.upper()} {cls}" for c in compilers for cls in task.classes]
    rows = {"Precision": ["Precision"], "Recall": ["Recall"], "F1 score": ["F1 score"]}
    for c in compilers:
        mask = np.array([compiler_of_bid.get(int(b)) == c for b in result["_bid"]])
        prf = per_class_prf(result["_pred"][mask], result["_true"][mask], task.num_labels)
        for i in range(task.num_labels):
            rows["Precision"].append(pct(prf["precision"][i]))
            rows["Recall"].append(pct(prf["recall"][i]))
            rows["F1 score"].append(pct(prf["f1"][i]))
    return markdown_table(header, list(rows.values()))


if __name__ == "__main__":
    raise SystemExit(main())
