"""Torch datasets and collators over a packed corpus.

The dataset returns raw byte slices; tokenization and MLM masking happen in the
collator, vectorised over the batch. That keeps DataLoader workers doing almost
nothing but memory-mapped reads, and it makes the masking *dynamic*: a fresh
random mask every time a sequence is drawn, which is what the paper means by
"these masked bytes are different at each epoch" (§4.1).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .corpus import Corpus, SequenceIndex
from .provenance import Task
from .vocab import BYTE_OFFSET, EOS_ID, BOS_ID, MASK_ID, PAD_ID, VOCAB_SIZE

IGNORE_INDEX = -100


def sequence_labels(corpus: Corpus, index: SequenceIndex, task: Task) -> np.ndarray:
    """Per-sequence class index, ``-1`` where the binary is out of task scope."""
    per_bid = np.full(len(corpus.records), -1, dtype=np.int64)
    for rec in corpus.records:
        lab = task.label_of(rec.labels)
        if lab is not None:
            per_bid[rec.bid] = lab
    return per_bid[index.bid]


def drop_unlabelled(index: SequenceIndex, labels: np.ndarray) -> tuple[SequenceIndex, np.ndarray]:
    """Filter out sequences whose binary the task does not cover.

    Needed for the O0/O1 and O2/O3 tasks of Table 4, which each use half the
    optimization levels.
    """
    keep = labels >= 0
    return index.subset(keep), labels[keep]


class ByteSequenceDataset(Dataset):
    """Fixed-length byte sequences cut from a corpus.

    Yields ``(bytes, label, position)`` where ``position`` is the sequence's
    index in the :class:`SequenceIndex` — evaluation needs it to map predictions
    back to the voting group.
    """

    def __init__(self, corpus: Corpus, index: SequenceIndex, labels: np.ndarray | None = None):
        if labels is not None and len(labels) != len(index):
            raise ValueError(f"labels ({len(labels)}) and index ({len(index)}) disagree")
        self.text = corpus.text
        self.index = index
        self.labels = labels

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        start = int(self.index.start[i])
        n = int(self.index.length[i])
        chunk = np.asarray(self.text[start : start + n], dtype=np.uint8)
        label = -1 if self.labels is None else int(self.labels[i])
        return chunk, label, i



class JitteredByteSequenceDataset(ByteSequenceDataset):
    """Training sequences whose window start is re-drawn every time they are read.

    Motivation: a non-overlapping cut (§3.1) gives a *fixed* set of windows, so
    an N-epoch fine-tune shows the model the identical byte strings N times. The
    corpus holds 211 MB of ``.text`` but only 315,710 distinct 512-byte windows,
    and where a window happens to begin is an artefact of the cut, not a
    property of the code. Re-drawing the start turns those 315,710 samples into
    a continuum, and it also stops the model keying on "byte 0 of a window is
    usually an instruction boundary" -- which is true of the training cut and
    only accidentally true at deployment.

    ``jitter`` is the maximum shift in bytes. ``jitter < 0`` means "start
    anywhere in this binary's ``.text``", the strongest form. The window is
    always clamped to stay inside the binary it came from, so a jittered sample
    never mixes two binaries and its label stays correct.
    """

    def __init__(
        self,
        corpus: Corpus,
        index: SequenceIndex,
        labels: np.ndarray | None = None,
        *,
        jitter: int = 256,
        seed: int = 0,
    ):
        super().__init__(corpus, index, labels)
        self.jitter = jitter
        self._rng = np.random.default_rng(seed)
        # Per-sequence bounds on the window start, so the clamp needs no lookup
        # into corpus.records at __getitem__ time (which runs in a worker).
        off = np.asarray([r.text_off for r in corpus.records], dtype=np.int64)
        ln = np.asarray([r.text_len for r in corpus.records], dtype=np.int64)
        bid = index.bid.astype(np.int64)
        self._lo = off[bid]
        self._hi = np.maximum(off[bid], off[bid] + ln[bid] - index.length.astype(np.int64))

    def __getitem__(self, i: int):
        start = int(self.index.start[i])
        n = int(self.index.length[i])
        lo, hi = int(self._lo[i]), int(self._hi[i])
        if self.jitter < 0:
            start = int(self._rng.integers(lo, hi + 1)) if hi > lo else lo
        elif self.jitter > 0:
            shift = int(self._rng.integers(-self.jitter, self.jitter + 1))
            start = min(max(start + shift, lo), hi)
        chunk = np.asarray(self.text[start : start + n], dtype=np.uint8)
        label = -1 if self.labels is None else int(self.labels[i])
        return chunk, label, i



class ContextWindowDataset(ByteSequenceDataset):
    """The paper's window set, each window widened to ``length`` centred bytes.

    This exists to make a longer-input model *comparable*. Training on 2048-byte
    sequences changes the unit being classified, so per-sequence accuracy at 2048
    bytes and at 512 bytes are different metrics and must not be put in the same
    column. Keeping the 512-byte index and only widening what the model reads
    around each window fixes that: the row count, the labels and the voting
    groups are identical to the baseline's, and the only thing that changed is
    how much context the prediction had.

    It is the same manoeuvre as ``--function-context``, and it carries the same
    caveat: the widened window contains bytes the 512-byte window does not, so
    this is not the paper's sequence level. Provenance is a property of the whole
    binary, so those bytes share the label and this is not label leakage — but it
    is a different, clearly-labelled operating point.
    """

    def __init__(self, corpus: Corpus, index: SequenceIndex, labels=None, *, length: int = 2048):
        super().__init__(corpus, index, labels)
        self.length = length
        off = np.asarray([r.text_off for r in corpus.records], dtype=np.int64)
        ln = np.asarray([r.text_len for r in corpus.records], dtype=np.int64)
        bid = index.bid.astype(np.int64)
        self._lo = off[bid]
        self._hi = off[bid] + ln[bid]

    def __getitem__(self, i: int):
        start = int(self.index.start[i])
        n = int(self.index.length[i])
        lo, hi = int(self._lo[i]), int(self._hi[i])
        want = min(self.length, hi - lo)
        centre = start + n // 2
        s = min(max(centre - want // 2, lo), max(lo, hi - want))
        chunk = np.asarray(self.text[s : s + want], dtype=np.uint8)
        label = -1 if self.labels is None else int(self.labels[i])
        return chunk, label, i

class MultiViewByteSequenceDataset(ByteSequenceDataset):
    """One sequence, several views of it — the evaluation side of jitter.

    Item ``i`` of this dataset is view ``i % n_views`` of sequence
    ``i // n_views``, so a plain non-shuffled pass yields every view of every
    sequence and the caller averages the ``n_views`` probability rows back down.
    The ``positions`` field still carries ``i``, so :func:`engine.predict`
    scatters correctly without knowing anything about views.

    Two modes, kept separate because they differ in what information the
    prediction is allowed to use:

    ``crop``
        Views are sub-windows *inside* the sequence's own bytes (length
        ``seq_len - k*shift`` for view ``k``). Nothing outside the 512 bytes is
        read, so an averaged prediction is still a sequence-level prediction and
        is comparable to the paper's number.

    ``shift``
        Views slide the full-length window off the sequence's start by up to
        ``span`` bytes, clamped to the binary. This reads bytes the sequence does
        not contain, so it is *not* the paper's sequence level -- report it as
        its own row, the way ``--function-context`` is.
    """

    def __init__(
        self,
        corpus: Corpus,
        index: SequenceIndex,
        labels: np.ndarray | None = None,
        *,
        mode: str = "crop",
        n_views: int = 4,
        span: int = 128,
    ):
        super().__init__(corpus, index, labels)
        if mode not in ("crop", "shift"):
            raise ValueError(f"mode must be 'crop' or 'shift', got {mode!r}")
        if n_views < 1:
            raise ValueError("n_views must be >= 1")
        self.mode = mode
        self.n_views = n_views
        self.span = span
        off = np.asarray([r.text_off for r in corpus.records], dtype=np.int64)
        ln = np.asarray([r.text_len for r in corpus.records], dtype=np.int64)
        bid = index.bid.astype(np.int64)
        self._lo = off[bid]
        self._hi = np.maximum(off[bid], off[bid] + ln[bid] - index.length.astype(np.int64))

    def __len__(self) -> int:
        return len(self.index) * self.n_views

    def _offsets(self) -> list[int]:
        """Signed shifts for the views, symmetric around 0 and including 0."""
        if self.n_views == 1:
            return [0]
        half = self.n_views // 2
        step = max(1, self.span // max(1, half))
        out = [0]
        for k in range(1, half + 1):
            out.append(-k * step)
            if len(out) < self.n_views:
                out.append(k * step)
        return out[: self.n_views]

    def __getitem__(self, j: int):
        i, view = divmod(j, self.n_views)
        start = int(self.index.start[i])
        n = int(self.index.length[i])
        delta = self._offsets()[view]
        if self.mode == "crop":
            # keep the window inside its own bytes: trim from one end
            trim = abs(delta)
            if delta >= 0:
                start, n = start + trim, max(16, n - trim)
            else:
                n = max(16, n - trim)
        else:
            start = min(max(start + delta, int(self._lo[i])), int(self._hi[i]))
        chunk = np.asarray(self.text[start : start + n], dtype=np.uint8)
        label = -1 if self.labels is None else int(self.labels[i])
        return chunk, label, j

class PairedByteSequenceDataset(ByteSequenceDataset):
    """Optionally splices two half-sequences from different binaries.

    The paper's segment embedding ``E_s`` exists to mark "which binary program
    each byte belongs to, when a byte sequence contains multiple fragments from
    different programs" (§3.2). With plain single-binary sequences that embedding
    never varies and stays dead weight. Setting ``pair_prob > 0`` during
    pre-training makes it meaningful. Off by default, since the paper does not
    say it used spliced inputs.
    """

    def __init__(self, corpus: Corpus, index: SequenceIndex, *, pair_prob: float = 0.0, seed: int = 0):
        super().__init__(corpus, index, None)
        self.pair_prob = pair_prob
        self._rng = np.random.default_rng(seed)

    def __getitem__(self, i: int):
        chunk, _, _ = super().__getitem__(i)
        if self.pair_prob <= 0 or self._rng.random() >= self.pair_prob or len(chunk) < 16:
            return chunk, -1, i
        j = int(self._rng.integers(0, len(self.index)))
        other, _, _ = super().__getitem__(j)
        if self.index.bid[j] == self.index.bid[i] or len(other) < 16:
            return chunk, -1, i
        cut = len(chunk) // 2
        spliced = np.concatenate([chunk[:cut], other[: len(chunk) - cut]])
        # negative position marks "segment boundary at `cut`" for the collator
        return spliced, -(cut + 1), i


def _tokenize_batch(batch, seq_tokens: int):
    """Build ``(input_ids, attention_mask, token_type_ids)`` for a batch.

    Layout per row: ``<s> b0 b1 ... bn-1 </s> <pad> ...``
    """
    bsz = len(batch)
    ids = np.full((bsz, seq_tokens), PAD_ID, dtype=np.int64)
    attn = np.zeros((bsz, seq_tokens), dtype=np.int64)
    types = np.zeros((bsz, seq_tokens), dtype=np.int64)

    max_bytes = seq_tokens - 2
    for row, (chunk, label, _pos) in enumerate(batch):
        n = min(len(chunk), max_bytes)
        ids[row, 0] = BOS_ID
        ids[row, 1 : 1 + n] = chunk[:n].astype(np.int64) + BYTE_OFFSET
        ids[row, 1 + n] = EOS_ID
        attn[row, : n + 2] = 1
        if label is not None and label < -1:  # spliced pair, see dataset above
            cut = -label - 1
            types[row, 1 + cut : 1 + n] = 1
    return ids, attn, types


class MLMCollator:
    """Masked-language-model batches (paper §3.2 / §4.1).

    Masking follows Pei et al. as the paper states: 20% of bytes are chosen; of
    those, 50% become ``<mask>`` and 50% become a random byte value. Note there
    is no BERT-style "keep original 10%" bucket here — that is the paper's
    setting, not an omission.
    """

    def __init__(
        self,
        seq_tokens: int,
        *,
        mask_prob: float = 0.20,
        mask_replace: float = 0.5,
        random_replace: float = 0.5,
        seed: int | None = None,
    ):
        if mask_replace + random_replace > 1.0 + 1e-9:
            raise ValueError("mask_replace + random_replace must not exceed 1")
        self.seq_tokens = seq_tokens
        self.mask_prob = mask_prob
        self.mask_replace = mask_replace
        self.random_replace = random_replace
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch):
        ids, attn, types = _tokenize_batch(batch, self.seq_tokens)

        # only real byte tokens are maskable: never <s>, </s> or padding
        maskable = (ids >= BYTE_OFFSET) & (attn == 1)
        selected = maskable & (self.rng.random(ids.shape) < self.mask_prob)

        labels = np.full(ids.shape, IGNORE_INDEX, dtype=np.int64)
        labels[selected] = ids[selected]

        draw = self.rng.random(ids.shape)
        to_mask = selected & (draw < self.mask_replace)
        to_random = selected & (draw >= self.mask_replace) & (
            draw < self.mask_replace + self.random_replace
        )
        ids[to_mask] = MASK_ID
        n_rand = int(to_random.sum())
        if n_rand:
            ids[to_random] = self.rng.integers(BYTE_OFFSET, VOCAB_SIZE, size=n_rand)

        return {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(attn),
            "token_type_ids": torch.from_numpy(types),
            "labels": torch.from_numpy(labels),
        }


class ClassificationCollator:
    """Batches for fine-tuning and evaluation."""

    def __init__(self, seq_tokens: int):
        self.seq_tokens = seq_tokens

    def __call__(self, batch):
        ids, attn, types = _tokenize_batch(batch, self.seq_tokens)
        labels = np.asarray([max(b[1], 0) for b in batch], dtype=np.int64)
        positions = np.asarray([b[2] for b in batch], dtype=np.int64)
        return {
            "input_ids": torch.from_numpy(ids),
            "attention_mask": torch.from_numpy(attn),
            "token_type_ids": torch.from_numpy(types),
            "labels": torch.from_numpy(labels),
            "positions": torch.from_numpy(positions),
        }
