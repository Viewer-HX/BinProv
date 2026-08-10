"""Byte vocabulary for BinProv (paper §3.2, "Input representation E").

The paper defines a vocabulary of 256 possible byte values plus 5 special
tags: <PAD>, <S>, </S>, <UNK> and <MASK>, for a total of 261 tokens.

Layout used here (ids are stable and must not be reshuffled once a model has
been pre-trained, since they are baked into the embedding matrix):

    0   <pad>
    1   <s>
    2   </s>
    3   <unk>
    4   <mask>
    5   0x00
    6   0x01
    ...
    260 0xff
"""

from __future__ import annotations

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3
MASK_ID = 4

NUM_SPECIAL = 5
BYTE_OFFSET = NUM_SPECIAL
VOCAB_SIZE = NUM_SPECIAL + 256  # 261

SPECIAL_TOKENS = ("<pad>", "<s>", "</s>", "<unk>", "<mask>")

#: ids that are never a prediction target of the MLM task
NON_BYTE_IDS = frozenset({PAD_ID, BOS_ID, EOS_ID, UNK_ID, MASK_ID})


def byte_to_id(b: int) -> int:
    """Map a raw byte value (0..255) to its token id."""
    if not 0 <= b <= 255:
        raise ValueError(f"not a byte value: {b}")
    return b + BYTE_OFFSET


def id_to_byte(i: int) -> int:
    """Inverse of :func:`byte_to_id`. Raises for special-token ids."""
    if i < BYTE_OFFSET or i >= VOCAB_SIZE:
        raise ValueError(f"not a byte token id: {i}")
    return i - BYTE_OFFSET


def encode(data: bytes) -> list[int]:
    """Encode raw bytes as token ids, without <s>/</s>."""
    return [b + BYTE_OFFSET for b in data]


def decode(ids) -> bytes:
    """Decode token ids back to bytes, dropping special tokens."""
    return bytes(i - BYTE_OFFSET for i in ids if i >= BYTE_OFFSET)


def token_repr(i: int) -> str:
    """Human-readable form of a token id, e.g. ``0x8b`` or ``<mask>``."""
    if i < NUM_SPECIAL:
        return SPECIAL_TOKENS[i]
    return f"0x{i - BYTE_OFFSET:02x}"
