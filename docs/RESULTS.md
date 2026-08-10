# Verified results

This page records the experimental configuration and the measurements used to
validate the reconstructed BinProv pipeline. The corresponding tables, figures,
and structured logs are stored under [`../reports/`](../reports/).

## Experimental setup

| Item | Configuration |
|---|---|
| Dataset | BinKit Normal |
| Compilers | GCC 8.2.0 and Clang 7.0 |
| Optimization levels | O0, O1, O2, O3 |
| Main architecture | x86_64 |
| Main corpus | 1,880 binaries; 211 MB packed `.text` data |
| Input | 512-byte sequences |
| Split | 8:2, grouped by source program |
| Encoder | 12 layers, 768 hidden units, 12 attention heads |
| Model size | 86.2 million trainable parameters |
| MLM masking | 20% selected bytes; half mask replacement, half random byte |
| MLM training | 10 epochs, batch 256, learning rate 1e-4 |
| Fine-tuning | 3 epochs, batch 128; one specialized model per task |
| Voting | Hard majority vote at binary level |

Grouping the split by program keeps every compiled variant of a program on the
same side of the train/test boundary. The corpus metadata records the exact
filters and split identifiers used by each run.

## Reproduced measurements

The following measurements are the primary validation points for this
reconstruction.

| Evaluation | Reconstruction | Paper |
|---|---:|---:|
| Overall compiler + High/Low optimization, sequence level | **94.98%** | 94.77% |
| Compiler identification, binary level | **100.00%** | 100.00% |
| O0/O1 identification, sequence level | **99.80%** | 98.49% |

The joint result counts a sequence as correct only when both the compiler and
the High/Low optimization classifiers are correct.

## Additional verified observations

- Compiler identification reaches 99.81% at sequence level and 100% after
  binary-level majority voting.
- Binary-level voting consistently improves predictions by aggregating evidence
  from all sequences belonging to the same binary.
- O0 and O1 are cleanly separated, matching the paper's observation that the
  lower optimization levels have distinct byte-level characteristics.
- GCC provenance is identified more consistently than Clang provenance in the
  fine-grained optimization experiment.
- Fixed-width instruction-set architectures are generally easier to identify
  than x86 in the multi-architecture experiment.

## Pre-training validation

Masked-language-model pre-training was evaluated against the corpus unigram
baseline. On the x86_64 BinKit corpus:

| Measurement | Value |
|---|---:|
| Pre-training sequences | 315,710 |
| Unigram cross-entropy | 4.351 nats |
| Unigram top-1 accuracy | 13.98% |
| Final validation loss | 0.9413 |
| Final masked-byte accuracy | 75.15% |

An equal-budget compiler-task control also confirms positive transfer from MLM
pre-training: the warm-started model reaches 99.77%, compared with 92.78% from
random initialization.

## Reproduce the run

```bash
pip install gdown
scripts/fetch_binkit.sh normal

export CUDA_VISIBLE_DEVICES=0
export ARCH=x86_64
export CORPUS=data/corpus/binkit_x86_64
export CKPT=checkpoints/binkit_x86_64
export RESULTS=results/binkit_x86_64

scripts/run_binkit.sh
```

The driver builds or reuses the corpus, pre-trains the byte encoder, fine-tunes
the task-specific classifiers, and writes `tables.md` plus `results.json` to the
selected results directory. See [`REPRODUCTION.md`](REPRODUCTION.md) for the
expanded commands and validation checklist.

## Artifact traceability

- `reports/tables/`: Markdown tables and machine-readable JSON output.
- `reports/figures/`: training and validation curves.
- `reports/runs/`: exact arguments, JSONL training logs, baselines, and final
  metrics for individual runs.

Model weights and raw datasets are reproducible but large, so they remain
gitignored.
