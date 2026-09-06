# Local training on Apple Silicon

This workflow trains and evaluates BinProv on a Mac through PyTorch MPS. It is
intended for local reproduction when a CUDA GPU is unavailable.

## Requirements

- Apple Silicon Mac with at least 48 GB unified memory
- macOS with MPS support
- Python 3.12
- The prepared corpus at `data/corpus/binkit_x86_64/`
- AC power for long unattended runs

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Prepare the corpus first if it is missing; see [DATA.md](DATA.md).

## Training profiles

The profiles are defined in `configs/local_release.json`.

| profile | model input | stages | measured/estimated requirement |
|---|---:|---|---|
| `opt4_narrow_seed13` | 512 bytes | MLM512 → fine-tune → evaluate → export | about 18.7 GB peak MPS memory; roughly 2–4 days |
| `opt4_wide_seed29` | 2048 bytes | MLM512 → MLM2048 → fine-tune → evaluate → export | 48 GB Mac supported by gradient accumulation; roughly two weeks or more |

Use `opt4_narrow_seed13` for a practical local run. The released model uses the
wide profile; training it on one A100 80 GB took 12:42:09.

## Run

List the available profiles:

```bash
python scripts/train_local_release.py --list
```

Inspect the plan without starting training:

```bash
python scripts/train_local_release.py --profile opt4_narrow_seed13
```

Run the complete local pipeline:

```bash
python scripts/train_local_release.py \
  --profile opt4_narrow_seed13 \
  --execute
```

Run only one phase:

```bash
python scripts/train_local_release.py \
  --profile opt4_narrow_seed13 \
  --phase finetune \
  --execute
```

Valid phases are `pretrain512`, `pretrain2048`, `finetune`, `evaluate`,
`export`, and `all`. The narrow profile does not use `pretrain2048`.

## Resume an interrupted run

The runner saves training state after each epoch. Resume with the same profile
and parameters:

```bash
python scripts/train_local_release.py \
  --profile opt4_narrow_seed13 \
  --resume \
  --execute
```

The runner refuses to resume when the saved arguments differ from the selected
profile. Move an incompatible output directory aside before starting a new run.

## Outputs

All files are written under:

```text
results/local_release/<profile>/
├── mlm512/                 512-byte MLM checkpoint
├── mlm2048/                wide profile only
├── opt4/                   fine-tuned checkpoint and probabilities
├── evaluate/report.json    sequence and binary metrics
├── export/                 self-contained model package
└── logs/                   stage logs
```

Stage status is recorded in `results/local_release/status.json`. The export
contains the model, inference script, label map, runtime package, license, and
recorded evaluation metadata.

Use the exported model locally with:

```bash
python results/local_release/<profile>/export/inference.py \
  --model results/local_release/<profile>/export/model \
  --elf /path/to/binary.elf
```

Reported measurements and their evidence are listed in [RESULTS.md](RESULTS.md).
