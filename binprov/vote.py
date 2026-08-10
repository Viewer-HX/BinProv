"""Joint inference by majority voting (paper §3.4).

A single 512-byte sequence is not a meaningful entity: it may hold a fragment of
one function, or span several. The paper therefore aggregates sequence-level
predictions over all sequences belonging to the same function, or the same
binary, and takes the majority. That relies on the §2.3 measurement that >96% of
real projects compile an individual binary with a single optimization level, so
the sequences of one binary do share a label.

Voting is where most of the paper's headline numbers come from: on the hard
O2/O3 task it lifts accuracy from 83.6% (sequence) to 94.7% (function) to 99.8%
(binary), Tables 4 and 6.
"""

from __future__ import annotations

import numpy as np


def majority_vote(
    group_ids: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
    *,
    probs: np.ndarray | None = None,
    mode: str = "hard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-sequence predictions into per-group predictions.

    Args:
        group_ids: dense group index per sequence (see
            :meth:`~binprov.corpus.SequenceIndex.group_ids`).
        predictions: predicted class per sequence.
        num_classes: size of the label set.
        probs: per-sequence class probabilities, required for ``mode="soft"``.
        mode: ``"hard"`` counts votes, as the paper does. ``"soft"`` sums
            probabilities, which breaks ties more gracefully when a group has
            few sequences.

    Returns:
        ``(groups, voted, n_seqs)`` — the sorted unique group ids, the voted
        class for each, and how many sequences each group had.
    """
    if len(group_ids) != len(predictions):
        raise ValueError("group_ids and predictions must be the same length")
    if len(group_ids) == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return empty_i, empty_i, empty_i

    groups = np.unique(group_ids)
    n_groups = int(groups.max()) + 1  # group_ids are dense, so this is exact

    if mode == "hard":
        flat = group_ids.astype(np.int64) * num_classes + predictions.astype(np.int64)
        tally = np.bincount(flat, minlength=n_groups * num_classes).reshape(n_groups, num_classes)
    elif mode == "soft":
        if probs is None:
            raise ValueError("mode='soft' needs probs")
        tally = np.zeros((n_groups, num_classes), dtype=np.float64)
        np.add.at(tally, group_ids.astype(np.int64), probs.astype(np.float64))
    else:
        raise ValueError(f"unknown mode {mode!r} (hard|soft)")

    voted = tally.argmax(axis=1)
    counts = np.bincount(group_ids.astype(np.int64), minlength=n_groups)
    return groups, voted[groups], counts[groups]


def group_truth(group_ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The ground-truth label of each group.

    Every sequence in a group must carry the same label — a group is one
    function or one binary, and a function has exactly one provenance. A
    mismatch means the corpus or the grouping is wrong, so this raises rather
    than silently picking one.
    """
    groups = np.unique(group_ids)
    n_groups = int(groups.max()) + 1 if len(groups) else 0
    first = np.full(n_groups, -1, dtype=np.int64)
    order = np.argsort(group_ids, kind="stable")
    g_sorted = group_ids[order]
    l_sorted = labels[order]
    starts = np.searchsorted(g_sorted, groups, side="left")
    first[groups] = l_sorted[starts]

    expanded = first[group_ids]
    bad = expanded != labels
    if bad.any():
        n = int(bad.sum())
        raise ValueError(
            f"{n} sequences disagree with their group's label; "
            "the voting groups do not match the label granularity"
        )
    return first[groups]


def vote_report(
    group_ids: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    probs: np.ndarray | None = None,
    mode: str = "hard",
) -> dict:
    """Voting accuracy plus the sequence-level accuracy it was built from."""
    groups, voted, counts = majority_vote(
        group_ids, predictions, num_classes, probs=probs, mode=mode
    )
    truth = group_truth(group_ids, labels)
    correct = voted == truth
    return {
        "num_groups": int(len(groups)),
        "num_sequences": int(len(predictions)),
        "seq_accuracy": float((predictions == labels).mean()) if len(predictions) else 0.0,
        "vote_accuracy": float(correct.mean()) if len(groups) else 0.0,
        "mean_seqs_per_group": float(counts.mean()) if len(counts) else 0.0,
        "median_seqs_per_group": float(np.median(counts)) if len(counts) else 0.0,
        "mode": mode,
        # kept for the per-binary drill-down in the analysis experiments
        "_groups": groups,
        "_voted": voted,
        "_truth": truth,
        "_counts": counts,
    }
