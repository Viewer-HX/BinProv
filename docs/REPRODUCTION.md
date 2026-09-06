# Reproducing the measured results

This guide covers the supported end-to-end paths for rebuilding the reported
measurements. Commands are run from the repository root.

## 1. Install dependencies

Python 3.12 is recommended. The dependency versions are pinned to the verified
training environment.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Prepare the data

Download BinKit Normal and build the packed x86_64 corpus by following
[DATA.md](DATA.md). The finished input must be:

```text
data/corpus/binkit_x86_64/
├── text.u8
├── binaries.jsonl.gz
├── meta.json
└── splits/default.json
```

The training drivers verify the corpus and program split before execution.

## 3. Reproduce the released single model

The verified CUDA profile is `configs/gpu_release.json`. It requires one A100
80 GB GPU or an equivalent CUDA device with enough memory.

Inspect the plan:

```bash
python scripts/train_release.py \
  --config configs/gpu_release.json \
  --profile opt4_wide_seed29
```

Run training, evaluation, and export:

```bash
python scripts/train_release.py \
  --config configs/gpu_release.json \
  --profile opt4_wide_seed29 \
  --execute
```

The complete pipeline runs these stages in order:

```text
MLM512 → MLM2048 → opt4 fine-tuning → evaluation → export
```

Outputs are written under `results/gpu_release/opt4_wide_seed29/`. A successful
run produces:

```text
opt4/result.json
opt4/probs.npz
evaluate/report.json
export/model/
export/inference.py
```

Resume an interrupted run by adding `--resume` with the same configuration.

## 4. Reproduce the ensemble measurements

The ensemble driver reads `configs/best_results.json`. It is a dry run unless
`--execute` is supplied.

List the measured groups:

```bash
python scripts/run_best.py --list
```

Inspect or execute a group:

```bash
python scripts/run_best.py --group o2o3_7wide_16.9KB
python scripts/run_best.py --group o2o3_7wide_16.9KB --execute

python scripts/run_best.py --group opt4_6run_16.9KB
python scripts/run_best.py --group opt4_6run_16.9KB --execute
```

Generated checkpoints, probabilities, and tables are written under
`results/best/`. Shared MLM and fine-tuning runs are reused when several groups
are executed together:

```bash
python scripts/run_best.py --group all --execute
```

To recompute an ensemble from the separately stored probability archive without
retraining, extract the archive at the repository root and run the offline
phase:

```bash
python scripts/run_best.py \
  --group opt4_6run_16.9KB \
  --phase offline \
  --archive-probs \
  --execute
```

## 5. Check the results

Compare generated metrics with [RESULTS.md](RESULTS.md). Version-controlled
reference evidence is available under:

```text
reports/runs/release/opt4_wide_seed29/
reports/runs/best/
reports/tables/best/
reports/tables/split_programs.txt
```

Repeated training can vary by roughly 1–2 percentage points. Keep the recorded
seed, split, dependency versions, effective batch size, and evaluation stride
when comparing a new run with the reported measurements.
