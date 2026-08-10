# Reproducing BinProv

This guide reproduces the reconstructed BinProv experiment pipeline, from
BinKit binaries to sequence- and binary-level provenance measurements.

Paper: He et al., *BinProv: Binary Code Provenance Identification without
Disassembly*, RAID 2022. <https://doi.org/10.1145/3545948.3545956>

## Pipeline overview

```text
scripts/fetch_binkit.sh
          |
          v
   BinKit ELF binaries
          |
          v
scripts/build_corpus.py  ----> packed .text bytes, metadata, and splits
          |
          v
scripts/pretrain_mlm.py  ----> byte-level encoder checkpoint
          |
          v
scripts/finetune.py      ----> one checkpoint per provenance task
          |
          v
scripts/evaluate.py      ----> tables.md and results.json
```

## 1. Environment

Python 3.11 and a CUDA-capable GPU are recommended for the full experiment.

```bash
conda create -n binprov python=3.11
conda activate binprov
pip install -r requirements.txt
pip install gdown
```

Confirm the installation and run the correctness tests:

```bash
python -c "import torch, transformers, numpy; print(torch.__version__)"
python tests/test_pipeline.py
```

Pin a GPU explicitly before training:

```bash
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
export PY=python3
```

The paper-sized model contains approximately 86 million parameters. A GPU with
at least 24 GB of memory is recommended; reduce `--batch-size` and use
`--grad-accum` when necessary.

## 2. Download BinKit

The main experiment uses BinKit Normal with GCC 8.2.0, Clang 7.0, and O0–O3.
The fetch script downloads the archive and selectively extracts the four paper
architectures by default.

```bash
scripts/fetch_binkit.sh normal --list
scripts/fetch_binkit.sh normal
```

To extract only x86_64:

```bash
KEEP_ARCHES="x86_64" scripts/fetch_binkit.sh normal
```

The extracted binaries are placed in `data/binkit/normal`. See
[`DATA.md`](DATA.md) for expected storage and file naming.

## 3. Build the corpus

Inspect label discovery before starting the full build:

```bash
$PY scripts/build_corpus.py \
    --root data/binkit/normal \
    --dry-run \
    --arch x86_64 \
    --compiler gcc clang \
    --opt O0 O1 O2 O3 \
    --extra normal
```

Build the packed corpus:

```bash
$PY scripts/build_corpus.py \
    --root data/binkit/normal \
    --out data/corpus/binkit_x86_64 \
    --arch x86_64 \
    --compiler gcc clang \
    --opt O0 O1 O2 O3 \
    --extra normal
```

The command extracts `.text` bytes, records provenance labels and function
boundaries, and creates an 8:2 train/test split grouped by source program. All
compiled variants of a program therefore remain in the same split.

Expected corpus files:

```text
data/corpus/binkit_x86_64/
  text.u8
  binaries.jsonl.gz
  meta.json
  splits/default.json
```

Check `meta.json` before training. For the selected BinKit x86_64 configuration,
the corpus should contain 1,880 binaries.

## 4. Pre-train the byte encoder

```bash
$PY scripts/pretrain_mlm.py \
    --corpus data/corpus/binkit_x86_64 \
    --out checkpoints/binkit_x86_64/mlm \
    --split-name train \
    --epochs 10 \
    --batch-size 256 \
    --workers 12 \
    --resume
```

The default architecture follows the paper configuration:

| Parameter | Value |
|---|---:|
| Sequence length | 512 bytes plus boundary tokens |
| Encoder layers | 12 |
| Hidden size | 768 |
| Attention heads | 12 |
| Intermediate size | 3,072 |
| Vocabulary size | 261 |
| Mask probability | 20% |

The pre-training command prints a unigram loss/accuracy baseline. A trained
contextual model should improve substantially over this reference. Checkpoints,
training state, arguments, and JSONL logs are written to the output directory.
The stored validation run used a physical batch of 256. On a smaller GPU, use
`--batch-size 64 --grad-accum 4` to retain the same effective batch size.

## 5. Fine-tune provenance classifiers

Train one model for each task:

```bash
for task in compiler opt_hl opt4 opt_o0o1 opt_o2o3; do
  $PY scripts/finetune.py \
      --corpus data/corpus/binkit_x86_64 \
      --task "$task" \
      --init-from checkpoints/binkit_x86_64/mlm \
      --out "checkpoints/binkit_x86_64/$task" \
      --epochs 3 \
      --batch-size 128 \
      --workers 8 \
      --resume
done
```

The classifier uses a two-layer head on top of the pre-trained encoder. Encoder
and classifier parameters are fine-tuned jointly. Each output directory stores
the model configuration, weights, launch arguments, and final metrics. If a
physical batch of 128 does not fit, use `--batch-size 64 --grad-accum 2`.

## 6. Evaluate and vote

For the selected verified measurements, evaluate the basic tasks, O0/O1, and
binary-level voting:

```bash
$PY scripts/evaluate.py \
    --corpus data/corpus/binkit_x86_64 \
    --out results/binkit_x86_64 \
    --levels binary \
    --ckpt compiler=checkpoints/binkit_x86_64/compiler \
    --ckpt opt_hl=checkpoints/binkit_x86_64/opt_hl \
    --ckpt opt_o0o1=checkpoints/binkit_x86_64/opt_o0o1
```

The evaluator reports sequence-level metrics first and then groups predictions
by binary for hard majority voting. It writes:

```text
results/binkit_x86_64/tables.md
results/binkit_x86_64/results.json
```

## One-command driver

After downloading BinKit, the complete experiment can be run with:

```bash
CUDA_VISIBLE_DEVICES=0 \
ARCH=x86_64 \
CORPUS=data/corpus/binkit_x86_64 \
CKPT=checkpoints/binkit_x86_64 \
RESULTS=results/binkit_x86_64 \
scripts/run_binkit.sh
```

Useful driver overrides for a shorter or lower-memory run:

```bash
MLM_EPOCHS=10 FT_EPOCHS=3 BATCH=64 scripts/run_binkit.sh
```

The driver is restart-friendly: it reuses an existing corpus and MLM checkpoint,
while each training stage supports resumable state. Use the expanded commands
above when reproducing the stored experiment configuration exactly.

## Paper table mapping

| Paper output | Task/checkpoint |
|---|---|
| Compiler identification | `compiler` |
| High/Low optimization | `opt_hl` |
| Four optimization levels | `opt4` |
| O0 versus O1 | `opt_o0o1` |
| O2 versus O3 | `opt_o2o3` |
| Architecture identification | `arch` |
| Joint compiler + High/Low result | `compiler` and `opt_hl` evaluated together |
| Binary-level joint inference | `evaluate.py --levels binary` |

## Implementation configuration

- **Framework:** Hugging Face Transformers provides the RoBERTa-style encoder.
- **Input:** raw `.text` bytes; no disassembly or handcrafted instruction
  features are used.
- **Split policy:** programs, rather than individual compiled binaries, are the
  split unit.
- **Pre-training selection:** the stored validation run uses all training
  binaries (`--split-name train`). The corpus also records a smaller
  one-program-per-package pre-training subset for controlled runs.
- **Optimization:** AdamW with warmup and cosine decay; mixed precision is used
  automatically on supported CUDA devices.
- **Voting:** hard majority voting is the default and matches the joint-inference
  design in the paper.
- **Reproducibility:** seeds, model configuration, corpus filters, split IDs,
  arguments, and training logs are persisted with each stage.

## Validation checklist

Before recording results, verify:

1. `python tests/test_pipeline.py` passes.
2. Corpus discovery reports the expected compiler, architecture, and
   optimization-level counts.
3. Train and test program groups do not overlap.
4. The MLM validation curve improves over the printed unigram baseline.
5. Fine-tuning loads the intended MLM checkpoint.
6. `results.json` and the relevant checkpoint argument files are archived
   together.

## Compact local validation

Use the smoke driver when checking installation or code changes:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_smoke.sh
```

It uses a small deterministic local corpus and a compact encoder so the entire
pipeline can be exercised quickly. Its measurements validate the software flow;
paper comparisons should use the BinKit configuration above.

## Current scope

The reconstruction implements the core BinProv pipeline and the main
provenance-classification tasks. External baselines, NCD/sequence-position
analyses, and the two case studies described later in the paper are outside the
current implementation.
