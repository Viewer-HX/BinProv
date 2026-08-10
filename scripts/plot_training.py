#!/usr/bin/env python3
"""Plot the training curves that pretrain_mlm.py and finetune.py write.

Both scripts log one JSON object per line to ``<out>/train_log.jsonl``, so the
curves are always recoverable after the fact — no plotting library needed during
training, and no lost history if a run is interrupted.

    # one MLM run: train + validation, against the unigram baseline
    python scripts/plot_training.py --log checkpoints/binkit_x86_64/mlm \\
        --out results/mlm_curve.png

    # compare runs (validation only, one colour per run)
    python scripts/plot_training.py --out results/warmstart_vs_scratch.png \\
        --log "warm start=checkpoints/compiler" \\
        --log "from scratch=checkpoints/compiler_scratch"

Loss, accuracy and learning rate go in **separate stacked panels sharing the
x-axis**, never on twin y-axes: two measures on one plot with two scales invents a
correlation that is not in the data.

For an MLM run the unigram baseline is drawn as a reference line, read from
``baseline.json`` if present. That line is the whole point of the plot — machine
code is dominated by a few byte values, so a model that has learned only the byte
histogram still reaches ~15% masked-byte accuracy, and a loss curve flattening
near the baseline means no context was learned at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display on this host
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order, light mode (see the dataviz reference
# palette). Never cycled, never reassigned by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e6e5e0"  # one step off the surface, recessive

LINE_W = 2.0  # thin marks
MARKER_SZ = 5.0  # ~10px diameter
REF_W = 1.5


def read_log(path: Path) -> list[dict]:
    """Read train_log.jsonl, tolerating a partially written final line."""
    log_file = path / "train_log.jsonl" if path.is_dir() else path
    if not log_file.exists():
        raise SystemExit(f"no log at {log_file}")
    rows = []
    for line in log_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # a run killed mid-write leaves one truncated line; keep the rest
            continue
    if not rows:
        raise SystemExit(f"{log_file} has no parseable rows yet")
    return rows


def read_baseline(path: Path) -> dict | None:
    f = (path if path.is_dir() else path.parent) / "baseline.json"
    return json.loads(f.read_text()) if f.exists() else None


def series(rows, key, xkey="step"):
    """Extract (x, y) for one logged field, skipping rows that lack it."""
    xs, ys = [], []
    for r in rows:
        if key in r and r[key] is not None:
            ys.append(r[key])
            # per-epoch rows carry no 'step'; fall back to the last step seen
            xs.append(r.get(xkey, xs[-1] if xs else 0))
        elif xkey in r:
            pass
    return xs, ys


def epoch_series(rows, key):
    """(x, y) for per-epoch rows, x placed at the step the epoch ended on."""
    xs, ys, last_step = [], [], 0
    for r in rows:
        if "step" in r:
            last_step = r["step"]
        if key in r and r[key] is not None:
            xs.append(last_step)
            ys.append(r[key])
    return xs, ys


def style_axis(ax, ylabel: str, *, last: bool = False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=1.0, linestyle="-")  # hairline, solid
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if last:
        ax.set_xlabel("optimizer step", color=INK_SECONDARY, fontsize=10)


def reference_line(ax, y: float, label: str):
    """A solid, recessive reference line with a direct label.

    Solid rather than dashed: dashing reads as noise, and this line matters more
    than the grid, not less.
    """
    ax.axhline(y, color=INK_SECONDARY, linewidth=REF_W, zorder=1)
    ax.annotate(
        label,
        xy=(0.985, y),
        xycoords=("axes fraction", "data"),
        ha="right",
        va="bottom",
        fontsize=9,
        color=INK_SECONDARY,
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--log",
        action="append",
        required=True,
        metavar="[LABEL=]DIR",
        help="checkpoint directory (or a train_log.jsonl). Repeatable to compare runs",
    )
    ap.add_argument("--out", default="results/training_curve.png")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--no-lr", action="store_true", help="omit the learning-rate panel")
    args = ap.parse_args()

    runs = []
    for spec in args.log:
        label, _, raw = spec.partition("=") if "=" in spec else ("", "", spec)
        path = Path(raw)
        rows = read_log(path)
        runs.append({
            "label": label or path.name,
            "path": path,
            "rows": rows,
            "baseline": read_baseline(path),
        })

    multi = len(runs) > 1
    # Which accuracy field this is: MLM logs mlm_acc, fine-tuning logs train_acc
    is_mlm = any("mlm_acc" in r for run in runs for r in run["rows"])
    train_acc_key = "mlm_acc" if is_mlm else "train_acc"
    val_acc_key = "val_mlm_acc" if is_mlm else "test_seq_acc"
    val_loss_key = "val_loss"

    have_acc = any(
        k in r for run in runs for r in run["rows"] for k in (train_acc_key, val_acc_key)
    )
    have_lr = (not args.no_lr) and any("lr" in r for run in runs for r in run["rows"])
    panels = ["loss"] + (["acc"] if have_acc else []) + (["lr"] if have_lr else [])

    heights = {"loss": 2.6, "acc": 2.2, "lr": 1.3}
    fig, axes = plt.subplots(
        len(panels), 1, sharex=True,
        figsize=(8.4, sum(heights[p] for p in panels) + 0.9),
        gridspec_kw={"height_ratios": [heights[p] for p in panels]},
    )
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    ax = dict(zip(panels, axes))

    # ---- loss ------------------------------------------------------------
    for i, run in enumerate(runs):
        if multi:
            # Comparing runs: one colour per run, so the panel stays at N series
            # instead of 2N. Prefer the validation curve; fall back to the
            # training curve when a run logs no validation loss (finetune.py logs
            # test accuracy, not test loss) -- an empty panel is worse than the
            # train curve clearly labelled as such.
            x, y = epoch_series(run["rows"], val_loss_key)
            if not y:
                x, y = series(run["rows"], "loss")
            ax["loss"].plot(x, y, color=SERIES[i % 8], linewidth=LINE_W,
                            label=run["label"])
        else:
            x, y = series(run["rows"], "loss")
            ax["loss"].plot(x, y, color=SERIES[0], linewidth=LINE_W, label="train")
            x, y = epoch_series(run["rows"], val_loss_key)
            if y:
                ax["loss"].plot(x, y, color=SERIES[1], linewidth=LINE_W,
                                marker="o", markersize=MARKER_SZ, label="validation")
    base = runs[0]["baseline"]
    if base and base.get("unigram_loss_nats"):
        reference_line(ax["loss"], base["unigram_loss_nats"],
                       f"unigram baseline {base['unigram_loss_nats']:.2f}")
    style_axis(ax["loss"], "cross-entropy (nats)", last=panels[-1] == "loss")

    # ---- accuracy --------------------------------------------------------
    if have_acc:
        for i, run in enumerate(runs):
            if multi:
                # line = the dense train curve; markers = the sparse eval points
                x, y = series(run["rows"], train_acc_key)
                if y:
                    ax["acc"].plot(x, [100 * v for v in y], color=SERIES[i % 8],
                                   linewidth=LINE_W, label=run["label"])
                xe, ye = epoch_series(run["rows"], val_acc_key)
                if ye:
                    ax["acc"].plot(xe, [100 * v for v in ye], color=SERIES[i % 8],
                                   linestyle="none", marker="o",
                                   markersize=MARKER_SZ + 1.5,
                                   markeredgecolor=SURFACE, markeredgewidth=1.5,
                                   label=None if y else run["label"])
            else:
                x, y = series(run["rows"], train_acc_key)
                if y:
                    ax["acc"].plot(x, [100 * v for v in y], color=SERIES[0],
                                   linewidth=LINE_W, label="train")
                x, y = epoch_series(run["rows"], val_acc_key)
                if y:
                    ax["acc"].plot(x, [100 * v for v in y], color=SERIES[1],
                                   linewidth=LINE_W, marker="o",
                                   markersize=MARKER_SZ, label="validation")
        if base and base.get("unigram_top1"):
            reference_line(ax["acc"], 100 * base["unigram_top1"],
                           f"unigram top-1 {100 * base['unigram_top1']:.1f}%")
        label = "masked-byte accuracy (%)" if is_mlm else "accuracy (%)"
        style_axis(ax["acc"], label, last=panels[-1] == "acc")

    # ---- learning rate ---------------------------------------------------
    if have_lr:
        lrs = [series(run["rows"], "lr") for run in runs]
        # A controlled comparison gives every run the same schedule, and drawing
        # them in series colours makes the last one drawn look like the only run
        # with a learning rate. Draw one neutral curve instead when they match.
        shared = multi and all(l[1] == lrs[0][1] for l in lrs) and lrs[0][1]
        if shared or not multi:
            x, y = lrs[0]
            ax["lr"].plot(x, y, color=INK_SECONDARY, linewidth=LINE_W)
            if shared:
                ax["lr"].annotate("shared schedule", xy=(0.985, 0.82),
                                  xycoords="axes fraction", ha="right",
                                  fontsize=9, color=INK_SECONDARY)
        else:
            for i, (x, y) in enumerate(lrs):
                ax["lr"].plot(x, y, color=SERIES[i % 8], linewidth=LINE_W)
        style_axis(ax["lr"], "learning rate", last=panels[-1] == "lr")
        ax["lr"].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax["lr"].yaxis.get_offset_text().set_color(INK_SECONDARY)

    # ---- legend: always present for >= 2 series --------------------------
    handles, labels = ax["loss"].get_legend_handles_labels()
    if len(handles) >= 2:
        # center-right, not upper-right: a descending loss curve leaves that
        # region empty, and upper-right is where the reference-line label sits.
        leg = ax["loss"].legend(
            handles, labels, frameon=False, loc="center right", fontsize=9,
            labelcolor=INK_SECONDARY, handlelength=1.8,
        )
        leg.set_zorder(5)

    title = args.title or (
        ("MLM pre-training" if is_mlm else "Fine-tuning")
        + (f" — {runs[0]['label']}" if not multi else "")
    )
    fig.suptitle(title, color=INK, fontsize=12, x=0.06, ha="left", y=0.985)
    if base and base.get("num_sequences"):
        fig.text(0.06, 0.945,
                 f"{base['num_sequences']:,} sequences, split '{base.get('split','?')}'",
                 color=INK_SECONDARY, fontsize=9, ha="left")

    fig.tight_layout(rect=(0, 0, 0.995, 0.935))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print(f"wrote {out}")

    # A chart is not the only output: print the numbers too, so a run can be
    # checked without opening an image.
    for run in runs:
        xs, ys = epoch_series(run["rows"], val_loss_key)
        if ys:
            print(f"\n{run['label']}: validation {val_loss_key} by epoch")
            for x, y in zip(xs, ys):
                print(f"  step {x:>7,}  {y:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
