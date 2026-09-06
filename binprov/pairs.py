"""Matched O2/O3 windows of the same function — the data for the pairwise loss.

Why pairs. On a program-grouped split the fine-tuned model reaches 83% training
accuracy and 67% test accuracy, so its problem is not capacity but that a large
part of what it learned is about the *programs* it trained on rather than about
optimization. A 512-byte window carries both signals at once and the plain
cross-entropy cannot tell the model which of the two to use.

BinKit compiles every program at both levels, so the confounder can be held
fixed instead of hoped away: take function ``F`` of program ``P`` from the O2
build and the same ``F`` from the O3 build. The two windows share the source
code, the compiler, the architecture and the function's role in the program, and
differ only in the optimization level. Any feature of "this is code from
program P" takes the same value on both members, so a loss that depends only on
the *difference* between the two members' scores cannot be reduced by such a
feature. That is what :func:`pairwise_margin_loss` is for.

Windows are full length and centred on the function, so a pair member looks like
an ordinary sequence rather than a short padded one — the same reasoning as
``--function-context`` in evaluation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .corpus import Corpus


@dataclass
class FunctionPairs:
    """Parallel arrays, one entry per matched (program, compiler, function)."""

    start_lo: np.ndarray   # absolute offset of the O2 member's window
    start_hi: np.ndarray   # absolute offset of the O3 member's window
    length: np.ndarray     # window length, the same for both members
    bid_lo: np.ndarray
    bid_hi: np.ndarray
    compiler: np.ndarray   # 0 gcc, 1 clang

    def __len__(self) -> int:
        return int(self.start_lo.shape[0])

    def subset(self, mask) -> FunctionPairs:
        return FunctionPairs(
            self.start_lo[mask], self.start_hi[mask], self.length[mask],
            self.bid_lo[mask], self.bid_hi[mask], self.compiler[mask],
        )


def _centred_window(rec, off: int, size: int, seq_len: int) -> int:
    """Start offset of a ``seq_len`` window centred on ``[off, off+size)``."""
    centre = off + size // 2
    start = min(max(centre - seq_len // 2, 0), max(0, rec.text_len - seq_len))
    return rec.text_off + start


def build_function_pairs(
    corpus: Corpus,
    bids,
    *,
    seq_len: int = 512,
    min_size: int = 32,
    levels: tuple[str, str] = ("O2", "O3"),
) -> FunctionPairs:
    """Match functions by name between the two builds of each program.

    ``bids`` restricts to one split, so pairs never cross the train/test line.
    Functions smaller than ``min_size`` are dropped: a 16-byte thunk is the same
    at every level and would contribute a pair with no signal in it.
    """
    allowed = set(bids)
    by_key: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for rec in corpus.records:
        if rec.bid not in allowed:
            continue
        opt, comp = rec.labels.get("opt"), rec.labels.get("compiler")
        if opt in levels and comp:
            by_key[(rec.group, comp)][opt] = rec.bid

    s_lo, s_hi, ln, b_lo, b_hi, cmp_ = [], [], [], [], [], []
    for (_group, comp), variants in sorted(by_key.items()):
        if len(variants) != 2:
            continue
        r_lo = corpus.records[variants[levels[0]]]
        r_hi = corpus.records[variants[levels[1]]]
        idx_hi = {n: i for i, n in enumerate(r_hi.func_names) if n}
        for i, name in enumerate(r_lo.func_names):
            if not name or name not in idx_hi:
                continue
            off_lo, size_lo = r_lo.functions[i]
            off_hi, size_hi = r_hi.functions[idx_hi[name]]
            if size_lo < min_size or size_hi < min_size:
                continue
            n = min(seq_len, r_lo.text_len, r_hi.text_len)
            s_lo.append(_centred_window(r_lo, off_lo, size_lo, n))
            s_hi.append(_centred_window(r_hi, off_hi, size_hi, n))
            ln.append(n)
            b_lo.append(r_lo.bid)
            b_hi.append(r_hi.bid)
            cmp_.append(0 if comp == "gcc" else 1)
    return FunctionPairs(
        np.asarray(s_lo, dtype=np.int64), np.asarray(s_hi, dtype=np.int64),
        np.asarray(ln, dtype=np.int32), np.asarray(b_lo, dtype=np.int32),
        np.asarray(b_hi, dtype=np.int32), np.asarray(cmp_, dtype=np.int8),
    )


class FunctionPairDataset(torch.utils.data.Dataset):
    """A flat dataset of ``2 * len(pairs)`` windows, pair members adjacent.

    Item ``2p`` is the low-level member of pair ``p`` and ``2p+1`` the high-level
    one. Keeping them adjacent means the ordinary
    :class:`~binprov.data.ClassificationCollator` can be reused unchanged and the
    training loop recovers the members as ``logits[0::2]`` and ``logits[1::2]``,
    with no pair bookkeeping in the collator.

    ``jitter`` shifts each member's window independently, so the model cannot
    solve a pair by comparing alignment artefacts.
    """

    def __init__(
        self,
        corpus: Corpus,
        pairs: FunctionPairs,
        label_lo,
        label_hi,
        *,
        jitter: int = 0,
        seed: int = 0,
    ):
        self.text = corpus.text
        self.pairs = pairs
        # Per-pair class indices rather than a fixed (0, 1): the factorized task
        # puts a gcc pair on classes (0, 1) and a clang pair on (2, 3).
        self.label_lo = np.asarray(label_lo, dtype=np.int64)
        self.label_hi = np.asarray(label_hi, dtype=np.int64)
        self.jitter = jitter
        self._rng = np.random.default_rng(seed)
        off = np.asarray([r.text_off for r in corpus.records], dtype=np.int64)
        ln = np.asarray([r.text_len for r in corpus.records], dtype=np.int64)
        self._bounds = (off, ln)

    def __len__(self) -> int:
        return 2 * len(self.pairs)

    def __getitem__(self, i: int):
        p, side = divmod(i, 2)
        if side == 0:
            start, bid = int(self.pairs.start_lo[p]), int(self.pairs.bid_lo[p])
        else:
            start, bid = int(self.pairs.start_hi[p]), int(self.pairs.bid_hi[p])
        n = int(self.pairs.length[p])
        if self.jitter:
            off, ln = self._bounds
            lo = int(off[bid])
            hi = max(lo, int(off[bid] + ln[bid]) - n)
            shift = int(self._rng.integers(-self.jitter, self.jitter + 1))
            start = min(max(start + shift, lo), hi)
        chunk = np.asarray(self.text[start : start + n], dtype=np.uint8)
        label = int(self.label_lo[p] if side == 0 else self.label_hi[p])
        return chunk, label, i


class PairBatchSampler(torch.utils.data.Sampler):
    """Batches of whole pairs, members adjacent and in order.

    Every batch therefore holds each chosen program's function at both levels,
    which also makes the batch label-balanced by construction.
    """

    def __init__(self, n_pairs: int, pairs_per_batch: int, *, shuffle: bool = True, seed: int = 0):
        self.n_pairs = n_pairs
        self.pairs_per_batch = pairs_per_batch
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.n_pairs // self.pairs_per_batch

    def __iter__(self):
        order = (self._rng.permutation(self.n_pairs) if self.shuffle
                 else np.arange(self.n_pairs))
        for b in range(len(self)):
            chunk = order[b * self.pairs_per_batch : (b + 1) * self.pairs_per_batch]
            out = []
            for p in chunk:
                out.extend((2 * int(p), 2 * int(p) + 1))
            yield out


def pairwise_margin_loss(
    logits: torch.Tensor, labels: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """Rank the two members of each pair by their level, and nothing else.

    ``logits`` and ``labels`` hold a pair-ordered batch: rows ``0::2`` are the
    low-level members, rows ``1::2`` the high-level ones. For the pair's own two
    classes ``lo`` and ``hi``, write ``d(x) = logit_hi(x) - logit_lo(x)``. The
    loss is ``softplus(margin - (d(x_hi) - d(x_lo)))``.

    The point is what drops out. Write ``d(x) = g(optimization) + h(program)``
    for any feature the encoder computes: ``h`` takes the same value on both
    members, so it cancels in the difference and cannot lower this loss. Only a
    feature that responds to the level can. Plain cross-entropy has no such
    property — it is happy to reach a low value using ``h`` alone on the
    programs it was shown.

    Taking the classes from ``labels`` rather than assuming ``(0, 1)`` is what
    lets the same loss serve the factorized task, where a gcc pair spans classes
    ``(gcc_O2, gcc_O3)`` and a clang pair spans ``(clang_O2, clang_O3)``.
    """
    if logits.size(0) % 2:
        raise ValueError("pairwise loss needs a pair-ordered batch of even size")
    lg = logits.float()
    lo_cls = labels[0::2].unsqueeze(1)
    hi_cls = labels[1::2].unsqueeze(1)
    lo_rows, hi_rows = lg[0::2], lg[1::2]
    d_lo = (lo_rows.gather(1, hi_cls) - lo_rows.gather(1, lo_cls)).squeeze(1)
    d_hi = (hi_rows.gather(1, hi_cls) - hi_rows.gather(1, lo_cls)).squeeze(1)
    return F.softplus(margin - (d_hi - d_lo)).mean()
