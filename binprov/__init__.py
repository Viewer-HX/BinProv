"""BinProv — compilation provenance identification straight from binary code.

A re-implementation of the experiment pipeline from:

    Xu He, Shu Wang, Yunlong Xing, Pengbin Feng, Haining Wang, Qi Li,
    Songqing Chen, Kun Sun. "BinProv: Binary Code Provenance Identification
    without Disassembly." RAID 2022. https://doi.org/10.1145/3545948.3545956

The pipeline, mirroring the paper's four stages (§3):

    1. pre-process  .text section -> fixed-length byte sequences
    2. embed        byte-level RoBERTa/BERT encoder, MLM pre-trained
    3. classify     2-layer FC head, fine-tuned per provenance task
    4. joint infer  majority vote over sequences of a function / a binary

The original code was lost; this rebuild follows the paper and marks every place
where the paper leaves a detail unspecified — see docs/REPRODUCTION.md.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Only the torch-free modules are imported eagerly, so that corpus building and
# inspection work in an environment without torch/transformers installed.
# `binprov.model` and `binprov.data` are imported directly where needed.
from . import corpus, discover, elf, metrics, provenance, vocab, vote

__all__ = [
    "corpus",
    "discover",
    "elf",
    "metrics",
    "provenance",
    "vocab",
    "vote",
]
