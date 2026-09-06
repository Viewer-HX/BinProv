#!/usr/bin/env python3
"""Offline analysis over saved per-sequence probabilities.

`scripts/experiment.py` writes `probs.npz` for every run: the probability rows
for the test split plus the (bid, start, length) of each window. Everything in
this file is then a numpy operation on those rows — no GPU, no re-running a
model — which is what makes it cheap to ask questions that would otherwise each
cost a fine-tune:

* **ensemble** — do two configurations make different mistakes?
* **context** — average the rows of windows that sit next to each other in the
  same binary. If accuracy climbs steeply with the neighbourhood radius, then the
  512-byte window is the binding constraint and a wider input is worth building;
  if it climbs slowly, context is not the problem. This is the cheap proxy for
  "should we train on longer sequences", answered before paying for it.
* **route** — send GCC windows to one specialist and Clang windows to another,
  using a compiler prediction rather than the true compiler where one is given.
* **per-compiler / per-binary breakdowns** of any of the above.

Every load checks the ``fingerprint`` stored with the rows, so combining two
runs whose test sets are not the same windows fails loudly instead of producing
a plausible number. That check exists because a mismatched test-set composition
already produced one near-miss false positive in this project.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from binprov.corpus import Corpus  # noqa: E402
from binprov.metrics import accuracy, markdown_table, pct, per_class_prf  # noqa: E402


def load_run(path: str | Path) -> dict:
    d = Path(path)
    npz = np.load(d / "probs.npz")
    res = json.loads((d / "result.json").read_text())
    prob = npz["prob"]
    mmap = npz["marginal_map"]
    if mmap.size:  # factorized run: sum the fine classes onto the scored ones
        out = np.zeros((prob.shape[0], int(mmap.max()) + 1), dtype=prob.dtype)
        for fine, coarse in enumerate(mmap.tolist()):
            out[:, coarse] += prob[:, fine]
        prob = out
        lut = np.asarray(mmap, dtype=np.int64)
        true = lut[npz["true"].astype(np.int64)]
    else:
        true = npz["true"].astype(np.int64)
    return {
        "tag": res.get("tag") or d.name,
        "dir": d,
        "prob": prob.astype(np.float64),
        "true": true,
        "bid": npz["bid"].astype(np.int64),
        "start": npz["start"].astype(np.int64),
        "length": npz["length"].astype(np.int64),
        "fingerprint": bytes(npz["fingerprint"].tolist()),
        "result": res,
    }


def require_same_windows(runs: list[dict]) -> None:
    ref = runs[0]
    for r in runs[1:]:
        if r["fingerprint"] != ref["fingerprint"]:
            raise SystemExit(
                f"{r['tag']} was scored over a different set of windows than "
                f"{ref['tag']}. Numbers over different test-set compositions are "
                "not comparable and must not be ensembled."
            )


def bal_acc(pred: np.ndarray, true: np.ndarray, n: int) -> float:
    prf = per_class_prf(pred, true, n)
    sup = prf["support"] > 0
    return float(prf["recall"][sup].mean()) if sup.any() else 0.0


def row(name: str, prob: np.ndarray, true: np.ndarray) -> list[str]:
    pred = prob.argmax(1)
    n = prob.shape[1]
    return [name, pct(accuracy(pred, true)), pct(bal_acc(pred, true, n)), str(len(true))]


# ---------------------------------------------------------------------------


def neighbourhood_average(prob, bid, start, radius: int) -> np.ndarray:
    """Average each window's row with the ``radius`` windows either side of it.

    Windows are contiguous non-overlapping cuts of one ``.text``, so
    "neighbouring rows in the same binary" is the same thing as "the surrounding
    bytes". Averaging r rows either side therefore approximates what a model with
    a ``(2r+1) x 512``-byte input could see, at zero training cost. The average
    never crosses a binary boundary.
    """
    order = np.lexsort((start, bid))
    p = prob[order]
    b = bid[order]
    csum = np.concatenate([np.zeros((1, p.shape[1])), np.cumsum(p, axis=0)])
    n = len(p)
    idx = np.arange(n)
    # first/last row index of each binary's contiguous block
    change = np.flatnonzero(np.diff(b)) + 1
    block_start = np.zeros(n, dtype=np.int64)
    block_start[change] = change
    block_start = np.maximum.accumulate(block_start)
    block_end = np.full(n, n - 1, dtype=np.int64)
    block_end[change - 1] = change - 1
    block_end = np.minimum.accumulate(block_end[::-1])[::-1]
    lo = np.maximum(idx - radius, block_start)
    hi = np.minimum(idx + radius, block_end)
    summed = csum[hi + 1] - csum[lo]
    avg = summed / (hi - lo + 1)[:, None]
    out = np.empty_like(prob)
    out[order] = avg
    return out


def program_of(corpus: Corpus, bid: np.ndarray) -> np.ndarray:
    """Dense program index per window — the unit a confidence interval must use."""
    groups = sorted({r.group for r in corpus.records})
    lut_by_group = {g: i for i, g in enumerate(groups)}
    lut = np.full(len(corpus.records), -1, dtype=np.int64)
    for rec in corpus.records:
        lut[rec.bid] = lut_by_group[rec.group]
    return lut[bid]


def bootstrap_ci(prob, true, prog, n_boot: int = 2000, seed: int = 0, level: float = 0.95):
    """95% interval for accuracy, resampling *programs* rather than windows.

    Resampling windows would give an interval a tenth this wide and would be
    wrong: a program contributes hundreds of windows whose errors are strongly
    correlated, because provenance is a property of the whole compilation and the
    thing that varies between programs is what their code looks like. The
    effective sample size is the 47 test programs, not the 55,657 windows, and
    resampling at the window level is exactly the mistake that makes a
    two-point difference look significant.
    """
    correct = (prob.argmax(1) == true).astype(np.int64)
    n_prog = int(prog.max()) + 1
    per_correct = np.bincount(prog, weights=correct, minlength=n_prog)
    per_total = np.bincount(prog, minlength=n_prog).astype(np.float64)
    present = np.flatnonzero(per_total > 0)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(present), size=(n_boot, len(present)))
    idx = present[draws]
    accs = per_correct[idx].sum(1) / np.maximum(per_total[idx].sum(1), 1)
    lo, hi = np.quantile(accs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi), int(len(present))


def bootstrap_paired_diff(prob_a, prob_b, true, prog, n_boot=2000, seed=0, level=0.95):
    """95% interval for accuracy(a) - accuracy(b), resampling programs.

    This, not the interval on either accuracy alone, is the statistic for "is A
    better than B on this test set". The two models are scored on identical
    programs, so their errors are highly correlated and the paired difference has
    a far narrower interval than either absolute number: the question "how
    accurate is this model" and the question "is this model better than that one"
    have very different amounts of evidence behind them here.
    """
    ca = (prob_a.argmax(1) == true).astype(np.int64)
    cb = (prob_b.argmax(1) == true).astype(np.int64)
    n_prog = int(prog.max()) + 1
    sa = np.bincount(prog, weights=ca, minlength=n_prog)
    sb = np.bincount(prog, weights=cb, minlength=n_prog)
    tot = np.bincount(prog, minlength=n_prog).astype(np.float64)
    present = np.flatnonzero(tot > 0)
    rng = np.random.default_rng(seed)
    idx = present[rng.integers(0, len(present), size=(n_boot, len(present)))]
    denom = np.maximum(tot[idx].sum(1), 1)
    diffs = (sa[idx].sum(1) - sb[idx].sum(1)) / denom
    lo, hi = np.quantile(diffs, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi), float((diffs > 0).mean())


def compiler_of(corpus: Corpus, bid: np.ndarray) -> np.ndarray:
    """0 = gcc, 1 = clang, per window."""
    lut = np.full(len(corpus.records), -1, dtype=np.int64)
    for rec in corpus.records:
        c = rec.labels.get("compiler")
        lut[rec.bid] = 0 if c == "gcc" else (1 if c == "clang" else -1)
    return lut[bid]


def binary_vote(prob, bid, true) -> tuple[np.ndarray, np.ndarray]:
    """Soft majority vote per binary — the paper's §3.4 binary level."""
    uniq, inv = np.unique(bid, return_inverse=True)
    tally = np.zeros((len(uniq), prob.shape[1]))
    np.add.at(tally, inv, prob)
    truth = np.zeros(len(uniq), dtype=np.int64)
    truth[inv] = true
    return tally.argmax(1), truth


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run directories written by experiment.py")
    ap.add_argument("--corpus", default="data/corpus/binkit_x86_64")
    ap.add_argument("--ensemble", action="store_true", help="mean of all given runs")
    ap.add_argument("--context", nargs="*", type=int, default=None,
                    help="neighbourhood radii to sweep, e.g. --context 0 1 2 4 8")
    ap.add_argument("--route", nargs=2, metavar=("GCC_RUN", "CLANG_RUN"), default=None,
                    help="per-compiler specialists to route between")
    ap.add_argument("--router", default=None,
                    help="a compiler-task run whose predictions do the routing "
                         "(default: the true compiler label, an upper bound)")
    ap.add_argument("--per-compiler", action="store_true")
    ap.add_argument("--binary-vote", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=0, metavar="N",
                    help="add a 95%% interval from N bootstrap resamples of the test "
                         "*programs*. Use it before reporting any difference of a few "
                         "points: over 10 runs here validation and test rankings "
                         "disagreed 38%% of the time")
    ap.add_argument("--out", default=None, help="write the tables here as markdown")
    args = ap.parse_args()

    runs = [load_run(r) for r in args.runs]
    require_same_windows(runs)
    true = runs[0]["true"]
    bid, start = runs[0]["bid"], runs[0]["start"]
    corpus = Corpus(args.corpus)
    comp = compiler_of(corpus, bid)
    lines: list[str] = []

    def emit(title, header, rows):
        lines.append(f"\n### {title}\n")
        lines.append(markdown_table(header, rows))
        print(f"\n### {title}\n")
        print(markdown_table(header, rows), flush=True)

    hdr = ["run", "accuracy", "balanced", "n"]
    emit("Individual runs", hdr, [row(r["tag"], r["prob"], true) for r in runs])

    if args.bootstrap:
        prog = program_of(corpus, bid)
        rows = []
        members = list(runs)
        if args.ensemble and len(runs) > 1:
            members = members + [{"tag": f"ensemble of {len(runs)}",
                                  "prob": np.mean([r["prob"] for r in runs], axis=0)}]
        for r in members:
            lo, hi, npg = bootstrap_ci(r["prob"], true, prog, args.bootstrap)
            acc = accuracy(r["prob"].argmax(1), true)
            rows.append([r["tag"], pct(acc), f"[{pct(lo)}, {pct(hi)}]",
                         f"+/-{100 * (hi - lo) / 2:.2f}", str(npg)])
        emit(f"95% interval over test programs ({args.bootstrap} resamples)",
             ["run", "accuracy", "95% interval", "half-width", "programs"], rows)

        base = members[0]
        rows = []
        for r in members[1:]:
            lo, hi, pwin = bootstrap_paired_diff(
                r["prob"], base["prob"], true, prog, args.bootstrap
            )
            d = accuracy(r["prob"].argmax(1), true) - accuracy(base["prob"].argmax(1), true)
            verdict = "resolved" if lo > 0 or hi < 0 else "not resolved"
            rows.append([r["tag"], f"{100 * d:+.2f}", f"[{100 * lo:+.2f}, {100 * hi:+.2f}]",
                         f"{pwin:.3f}", verdict])
        emit(f"Paired difference against `{base['tag']}` — the statistic for "
             f"\"is A better than B\"",
             ["run", "Δ accuracy", "95% interval on Δ", "P(Δ>0)", "verdict"], rows)

    if args.per_compiler:
        rows = []
        for r in runs:
            for ci, cname in enumerate(("gcc", "clang")):
                m = comp == ci
                rows.append(row(f"{r['tag']} / {cname}", r["prob"][m], true[m]))
        emit("Per compiler", hdr, rows)

    ens = None
    if args.ensemble and len(runs) > 1:
        ens = np.mean([r["prob"] for r in runs], axis=0)
        rows = [row(r["tag"], r["prob"], true) for r in runs]
        rows.append(row(f"ensemble of {len(runs)}", ens, true))
        # a logit-space mean weights confident members more heavily
        logit = np.mean([np.log(np.clip(r["prob"], 1e-9, 1)) for r in runs], axis=0)
        rows.append(row("ensemble (log-mean)", logit, true))
        emit("Ensemble", hdr, rows)

    if args.context is not None:
        radii = args.context or [0, 1, 2, 4, 8, 16]
        for r in runs + ([{"tag": "ensemble", "prob": ens}] if ens is not None else []):
            rows = []
            for rad in radii:
                p = neighbourhood_average(r["prob"], bid, start, rad)
                label = f"radius {rad} ({(2 * rad + 1) * 512} bytes seen)"
                rows.append(row(label, p, true))
            emit(f"Context scaling — {r['tag']}", hdr, rows)

    if args.route:
        gcc_run, clang_run = (load_run(p) for p in args.route)
        require_same_windows([runs[0], gcc_run, clang_run])
        if args.router:
            router = load_run(args.router)
            require_same_windows([runs[0], router])
            which = router["prob"].argmax(1)
            agree = float((which == comp).mean())
            router_name = f"{router['tag']} ({pct(agree)} agreement with the label)"
        else:
            which = comp
            router_name = "the true compiler label (upper bound)"
        routed = np.where((which == 0)[:, None], gcc_run["prob"], clang_run["prob"])
        rows = [
            row("gcc specialist, all windows", gcc_run["prob"], true),
            row("clang specialist, all windows", clang_run["prob"], true),
            row(f"routed by {router_name}", routed, true),
        ]
        for ci, cname in enumerate(("gcc", "clang")):
            m = comp == ci
            rows.append(row(f"  routed / true {cname} windows", routed[m], true[m]))
        emit("Compiler routing", hdr, rows)

    if args.binary_vote:
        rows = []
        for r in runs + ([{"tag": "ensemble", "prob": ens}] if ens is not None else []):
            pred, truth = binary_vote(r["prob"], bid, true)
            rows.append([r["tag"], pct(accuracy(pred, truth)),
                         pct(bal_acc(pred, truth, r["prob"].shape[1])), str(len(truth))])
        emit("Binary-level soft vote", hdr, rows)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
