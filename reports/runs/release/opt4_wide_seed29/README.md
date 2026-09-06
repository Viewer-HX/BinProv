# A100 release replica: `opt4_wide_seed29`

This is the compact, version-controlled evidence for the release model trained
from the repository recipe on one A100 80 GB GPU. The upload-ready weights stay in the
gitignored `results/gpu_release/opt4_wide_seed29/export/` directory.

| item | value |
|---|---|
| hardware | one A100 80 GB, 8 CPU cores, 64 GB system memory, bf16 |
| elapsed | 12:42:09 (2026-09-05 15:06:56 to 2026-09-06 03:49:05 EDT) |
| best epoch | 4 of 6 |
| validation accuracy | 77.6125% |
| sequence accuracy | **84.2255%** (balanced 83.0951%, 116,321 windows) |
| binary soft-vote accuracy | **93.8830%** (376 binaries) |
| archived seed-29 reference | 83.85% sequence; 93.88% binary |

The fresh sequence score is 0.3755 percentage points above the archived
seed-29 result; binary accuracy agrees to rounding. `evaluation.json` also
records that recomputation from the saved probabilities matches `result.json`.

Files:

- `args.json`: exact fine-tuning arguments.
- `result.json`: epoch history, selected epoch, confusion matrix, and test metrics.
- `evaluation.json`: independently recomputed sequence and binary metrics.
- `pipeline_status.json`: completion timestamps for MLM512, MLM2048, fine-tune,
  and export.
- `hardware.json`: portable hardware allocation and elapsed-time summary.
- `mlm512/` and `mlm2048/`: compact pre-training traces and unigram baselines.
- `train_log.jsonl`: compact fine-tuning trace.

The three loss curves are shown in
[`reports/figures/release_training_loss.png`](../../../figures/release_training_loss.png).
Regenerate the figure from these traces with:

```bash
python scripts/plot_training.py --sequential \
  --log "MLM512=reports/runs/release/opt4_wide_seed29/mlm512" \
  --log "MLM2048=reports/runs/release/opt4_wide_seed29/mlm2048" \
  --log "O0/O1/O2/O3 fine-tuning=reports/runs/release/opt4_wide_seed29/train_log.jsonl" \
  --title "Released model: training loss by stage" \
  --out reports/figures/release_training_loss.png
```

The portable training parameters are in `configs/gpu_release.json`; hardware
requirements are documented in the repository README.
