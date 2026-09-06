# A100 release replica: `opt4_wide_seed29`

This is the compact, version-controlled evidence for the release model trained
from the repository recipe on GMU Hopper. The upload-ready weights stay in the
gitignored `results/hopper_release/opt4_wide_seed29/export/` directory.

| item | value |
|---|---|
| Slurm job | `9558937`, one A100 80 GB, completed with exit code `0:0` |
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
- `job.json`: scheduler outcome, allocation, and elapsed time.
- `train_log.jsonl`: compact fine-tuning trace.

Reproduce the complete training and export with
`configs/hopper_release.json`, `scripts/train_local_release.py`, and
`scripts/hopper_release.sbatch`, as documented in `docs/HOPPER_TRAINING.md`.
