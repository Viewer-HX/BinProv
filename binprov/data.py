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
