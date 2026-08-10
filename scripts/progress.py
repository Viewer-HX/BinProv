#!/usr/bin/env python3
"""Show progress of a running pipeline, from its logs alone.

Nothing is instrumented for this: `pretrain_mlm.py` and `finetune.py` already log
each step's count and ETA to ``train_log.jsonl``, and the driver script prints a
``--- <task> ---`` banner per task. That is enough to reconstruct where a run is,
which means progress can be checked without attaching to the process — useful
when the job is detached under nohup on a shared machine.

    # a whole run_binkit.sh pipeline
    python scripts/progress.py --run-log logs/binkit_x86_64_full.log \\
        --ckpt-root checkpoints/binkit_x86_64

    # a single training run
    python scripts/progress.py --ckpt checkpoints/binkit_x86_64/mlm

    # refresh in place every 30s
    watch -n 30 python scripts/progress.py --run-log ... --ckpt-root ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BAR_W = 24
FULL, EMPTY = "█", "░"  # █ ░

_TASK_RE = re.compile(r"^--- (\S+) ---\s*$")
_TOTAL_RE = re.compile(r"(\d+) epochs? x (\d+) steps = (\d+) optimizer steps")
_BEST_RE = re.compile(r"best sequence-level test accuracy: ([0-9.]+)%")
_ETA_RE = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


def bar(frac: float, width: int = BAR_W) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return FULL * n + EMPTY * (width - n)


def parse_eta(s: str) -> float | None:
    """'1h07m' / '13m20s' / '45s' -> seconds."""
    m = _ETA_RE.match(s.strip())
    if not m or not any(m.groups()):
        return None
    h, mi, se = (int(g) if g else 0 for g in m.groups())
    return h * 3600 + mi * 60 + se


def hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def last_log_row(ckpt: Path) -> dict | None:
    f = ckpt / "train_log.jsonl"
    if not f.exists():
        return None
    last = None
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "step" in row:
            last = row
    return last


def read_run_log(path: Path):
    """Reconstruct per-task state from the driver's stdout.

    Returns ``(order, state)`` where state maps task -> dict with 'total' steps
    and 'best' accuracy when they have been printed.
    """
    order: list[str] = []
    state: dict[str, dict] = {}
    current = None
    for line in path.read_text().splitlines():
        if m := _TASK_RE.match(line):
            current = m.group(1)
            if current not in state:
                order.append(current)
                state[current] = {}
            continue
        if current is None:
            continue
        if m := _TOTAL_RE.search(line):
            state[current]["total"] = int(m.group(3))
        if m := _BEST_RE.search(line):
            state[current]["best"] = float(m.group(1))
    return order, state


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-log", help="stdout log of run_binkit.sh / run_smoke.sh")
    ap.add_argument("--ckpt-root", help="parent of the per-task checkpoint dirs")
    ap.add_argument("--ckpt", help="a single training run's checkpoint dir")
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=["compiler", "opt_hl", "opt4", "opt_o0o1", "opt_o2o3"],
        help="expected task order, so queued tasks are listed before they start",
    )
    args = ap.parse_args()

    if args.ckpt and not args.run_log:
        return single(Path(args.ckpt))
    if not (args.run_log and args.ckpt_root):
        ap.error("give --ckpt for one run, or --run-log with --ckpt-root for a pipeline")

    run_log = Path(args.run_log)
    if not run_log.exists():
        raise SystemExit(f"no run log at {run_log}")
    root = Path(args.ckpt_root)
    seen, state = read_run_log(run_log)
    # tasks the driver has not reached yet still deserve a row
    order = seen + [t for t in args.tasks if t not in seen]

    print(f"pipeline: {run_log.name}   checkpoints: {root}")
    print()

    remaining_s = 0.0
    done_steps = total_steps = 0
    rate = None  # steps/second, measured from the in-flight task

    rows = []
    for task in order:
        st = state.get(task, {})
        row = last_log_row(root / task)
        total = st.get("total")
        step = row.get("step", 0) if row else 0
        if "best" in st:
            status, frac = f"{st['best']:.2f}%", 1.0
            done_steps += total or step
            total_steps += total or step
        elif row and total:
            frac = step / total
            eta = parse_eta(str(row.get("eta", "")))
            if eta and step < total:
                rate = (total - step) / max(1.0, eta)
                remaining_s += eta
            status = f"epoch {row.get('epoch', '?')}  " + (
                f"train {100 * row['train_acc']:.1f}%" if "train_acc" in row
                else f"loss {row.get('loss', float('nan')):.3f}"
            )
            done_steps += step
            total_steps += total
        else:
            frac, status, total = 0.0, "queued", None
        rows.append((task, frac, step, total, status))

    # estimate the queued tasks from the measured rate and a typical task size
    typical = [t for _, _, _, t, _ in rows if t]
    est_total = max(typical) if typical else 0
    n_queued = sum(1 for _, _, _, t, s in rows if s == "queued")
    if rate and est_total and n_queued:
        # queued tasks vary in size (O0/O1 and O2/O3 use half the data), so this
        # is an upper bound, flagged as approximate below
        remaining_s += n_queued * est_total / rate
        total_steps += n_queued * est_total

    width = max(len(t) for t, *_ in rows)
    for task, frac, step, total, status in rows:
        steps = f"{step:>6,}/{total:,}" if total else " " * 13
        print(f"  {task:<{width}}  {bar(frac)} {100 * frac:5.1f}%  {steps}  {status}")

    print()
    overall = done_steps / total_steps if total_steps else 0.0
    line = f"  {'overall':<{width}}  {bar(overall)} {100 * overall:5.1f}%"
    if remaining_s:
        approx = "~" if n_queued else ""
        line += f"  ETA {approx}{hms(remaining_s)}"
    print(line)
    if n_queued:
        print(f"  ({n_queued} task(s) not started; their size is estimated from the largest so far)")
    return 0


def single(ckpt: Path) -> int:
    """Progress of one training run, from its log and args."""
    row = last_log_row(ckpt)
    if not row:
        raise SystemExit(f"no train_log.jsonl rows in {ckpt}")
    args_f = next((ckpt / n for n in ("finetune_args.json", "pretrain_args.json")
                   if (ckpt / n).exists()), None)
    meta = json.loads(args_f.read_text()) if args_f else {}
    print(f"run: {ckpt}")
    if meta:
        print(f"  {meta.get('epochs')} epochs, batch {meta.get('batch_size')}, "
              f"lr {meta.get('lr')}")
    step = row.get("step", 0)
    eta = parse_eta(str(row.get("eta", "")))
    # total is not in the log, so derive it from step and the reported ETA
    total = None
    if eta and eta > 0:
        # step/elapsed is unknown, but ETA + step gives the fraction implicitly
        pass
    print(f"  step {step:,}" + (f"   ETA {hms(eta)}" if eta else ""))
    for k in ("epoch", "loss", "mlm_acc", "train_acc", "lr"):
        if k in row:
            v = row[k]
            print(f"  {k:<10} {v:.4g}" if isinstance(v, float) else f"  {k:<10} {v}")
    mtime = (ckpt / "train_log.jsonl").stat().st_mtime
    print(f"  last log write {hms(time.time() - mtime)} ago")
    if total is None and eta:
        print(f"  (total steps not logged per row; overall progress needs --run-log)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
