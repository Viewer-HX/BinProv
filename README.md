# BinProv

BinProv identifies compilation provenance—such as the compiler family and
optimization level of a binary—without disassembly. It reads raw bytes from the
ELF `.text` section, learns contextual byte representations with a
bidirectional Transformer encoder, and performs provenance classification at
sequence, function, and binary granularity.

The method was published at RAID 2022:
[BinProv: Binary Code Provenance Identification without Disassembly](https://doi.org/10.1145/3545948.3545956).

This repository provides a reconstruction of the experiment pipeline described
in the paper. It uses Hugging Face `transformers` for the byte-level encoder and
keeps the data preparation, training, evaluation, and voting stages in separate,
reusable modules.

## Pipeline

| Stage | Description | Main implementation |
|---|---|---|
| Data preparation | Locate ELF binaries, read `.text`, and record function boundaries | `binprov/elf.py`, `binprov/discover.py` |
| Corpus construction | Pack bytes and metadata; create program-grouped train/test splits | `binprov/corpus.py`, `scripts/build_corpus.py` |
| Pre-training | Masked-language modeling over byte sequences | `binprov/model.py`, `scripts/pretrain_mlm.py` |
| Fine-tuning | Train one classifier for each provenance task | `scripts/finetune.py` |
| Evaluation | Report sequence accuracy and function/binary majority voting | `binprov/vote.py`, `scripts/evaluate.py` |

The vocabulary contains 261 tokens: 256 byte values and five special tokens
(`<pad>`, `<s>`, `</s>`, `<unk>`, and `<mask>`). The same representation is used
for x86, ARM, and MIPS binaries.

## Verified results

The reconstructed pipeline has been validated on the x86_64 subset of BinKit
Normal using GCC 8.2.0, Clang 7.0, and optimization levels O0–O3. Selected
results that reproduce the paper's reported behavior are:

| Evaluation | Reconstruction | Paper |
|---|---:|---:|
| Overall compiler + High/Low optimization, sequence level | **94.98%** | 94.77% |
| Compiler identification, binary level | **100.00%** | 100.00% |
| O0/O1 identification, sequence level | **99.80%** | 98.49% |

The experiments also confirm the benefits of binary-level majority voting and
the positive contribution of masked-language-model pre-training. See
[`docs/RESULTS.md`](docs/RESULTS.md) for the verified setup and measurements.

## Installation

Python 3.11 is recommended.

```bash
conda create -n binprov python=3.11
conda activate binprov
pip install -r requirements.txt
```

The runtime dependencies are PyTorch, Transformers, and NumPy. ELF parsing and
evaluation metrics are implemented in this repository.

## Quick start

The smoke test builds a deterministic local dataset and exercises the complete
pipeline with a compact model:

```bash
export CUDA_VISIBLE_DEVICES=0
scripts/run_smoke.sh
```

For the BinKit experiment:

```bash
pip install gdown
scripts/fetch_binkit.sh normal

export CUDA_VISIBLE_DEVICES=0
scripts/run_binkit.sh
```

The full run creates:

```text
data/corpus/x86_64/   packed corpus and split metadata
checkpoints/x86_64/   MLM and task-specific checkpoints
results/x86_64/       tables.md and results.json
```

The training scripts support `--resume`, and the driver reuses completed corpus
and checkpoint stages. Detailed commands and configuration are documented in
[`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

## Manual workflow

```bash
# Build a packed corpus from an existing BinKit extraction.
python scripts/build_corpus.py \
    --root data/binkit/normal \
    --out data/corpus/x86_64 \
    --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --extra normal

# Pre-train the byte encoder.
python scripts/pretrain_mlm.py \
    --corpus data/corpus/x86_64 \
    --out checkpoints/x86_64/mlm \
    --epochs 10 --batch-size 64 --resume

# Fine-tune a compiler classifier.
python scripts/finetune.py \
    --corpus data/corpus/x86_64 \
    --task compiler \
    --init-from checkpoints/x86_64/mlm \
    --out checkpoints/x86_64/compiler \
    --epochs 5 --batch-size 64 --resume

# Evaluate sequence- and binary-level predictions.
python scripts/evaluate.py \
    --corpus data/corpus/x86_64 \
    --out results/x86_64 \
    --levels binary \
    --ckpt compiler=checkpoints/x86_64/compiler
```

## Classification tasks

| Task | Classes |
|---|---|
| `compiler` | GCC, Clang |
| `opt_hl` | low (O0/O1), high (O2/O3) |
| `opt4` | O0, O1, O2, O3 |
| `opt_o0o1` | O0, O1 |
| `opt_o2o3` | O2, O3 |
| `arch` | x86_64, x86_32, ARM, MIPS, and other configured ISAs |

Task definitions and label mappings live in `binprov/provenance.py`.

## Repository layout

```text
binprov/       reusable corpus, model, provenance, voting, and metric modules
scripts/       dataset, training, evaluation, plotting, and driver scripts
tests/         pipeline correctness tests
docs/          data and reproduction guides
reports/       selected tables, figures, and structured run logs
```

- [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md): end-to-end reproduction steps.
- [`docs/DATA.md`](docs/DATA.md): BinKit selection, storage, and corpus layout.
- [`docs/RESULTS.md`](docs/RESULTS.md): verified experimental setup and results.
- [`reports/README.md`](reports/README.md): organization of stored artifacts.

## Scope

The current reconstruction covers data preparation, byte-level MLM
pre-training, task-specific fine-tuning, and joint inference. The external
baselines and the later analysis/case-study experiments from the paper are not
included in the current pipeline.

## Baselines referenced by the paper

1. [BinRNN](https://github.com/shuwang127/BinRNN)
2. [o-glassesX](https://github.com/yotsubo/o-glassesX)
3. [Origin](https://github.com/dyninst/toolchain-origin)

## Citation

```bibtex
@inproceedings{xu2022binprov,
  author = {He, Xu and Wang, Shu and Xing, Yunlong and Feng, Pengbin and
            Wang, Haining and Li, Qi and Chen, Songqing and Sun, Kun},
  title = {BinProv: Binary Code Provenance Identification without Disassembly},
  year = {2022},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 25th International Symposium on Research in
               Attacks, Intrusions and Defenses},
  pages = {350--363},
  location = {Limassol, Cyprus},
  series = {RAID '22}
}
```
