# Curated result evidence

This directory contains only the compact evidence needed to inspect and
reproduce the selected configurations. Raw datasets, checkpoints, probabilities,
temporary logs, and the broader experiment history are excluded from Git.

## Layout

| path | contents |
|---|---|
| `figures/release_training_loss.png` | MLM512, MLM2048, and fine-tuning loss curves for the released model |
| `tables/best/` | independently recomputed result tables |
| `tables/split_programs.txt` | exact program-grouped test and validation split |
| `runs/best/` | exact `args.json` and `result.json` for the 13 ensemble members, plus the MLM512 recipe |
| `runs/release/opt4_wide_seed29/` | A100 run parameters, metrics, hardware summary, pipeline status, and training traces |

The machine-readable recipe map is
[`configs/best_results.json`](../configs/best_results.json), and the main reader
guide is [`docs/RESULTS.md`](../docs/RESULTS.md).

The released 335 MiB model is hosted at
[XuViewer/binprov](https://huggingface.co/XuViewer/binprov). A local export is
written under gitignored `results/gpu_release/`; model weights are never
committed to this repository.

To replay a recipe, first inspect the mutation-free plan:

```bash
python scripts/run_best.py --list
python scripts/run_best.py --group opt4_6run_16.9KB
```

Pass `--execute` only after preparing the BinKit corpus as described in the
repository README.
