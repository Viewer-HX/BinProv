# Experiment artifacts

This directory stores compact artifacts used to validate the reconstructed
BinProv pipeline. Raw datasets, model weights, and generated `results/`
directories are excluded from version control because they are large and can be
recreated from the documented commands.

```text
tables/    evaluation output in Markdown and JSON
figures/   training and validation curves
runs/      launch arguments, JSONL logs, baselines, and final metrics
```

## Tables

| Artifact | Description |
|---|---|
| `binkit_x86_64.md` / `.json` | Main x86_64 evaluation output |
| `binkit_x86_64_fnctx.md` / `.json` | Function-context evaluation output |
| `o2o3_4arch_<arch>.md` / `.json` | Per-architecture evaluation output |

The verified measurements highlighted in
[`../docs/RESULTS.md`](../docs/RESULTS.md) are derived from these structured
outputs.

## Figures

| Artifact | Description |
|---|---|
| `mlm_curve.png` | x86_64 MLM training curve and unigram reference |
| `mlm_curve_4arch.png` | Pooled four-architecture MLM training curve |
| `control_warm_vs_scratch.png` | Equal-budget MLM warm start and random initialization control |

Regenerate the figures from the stored logs:

```bash
python scripts/plot_training.py \
    --log "x86_64=reports/runs/binkit_x86_64__mlm" \
    --out reports/figures/mlm_curve.png

python scripts/plot_training.py \
    --log "four architectures=reports/runs/binkit_4arch__mlm" \
    --out reports/figures/mlm_curve_4arch.png

python scripts/plot_training.py \
    --log "warm start=reports/runs/binkit_x86_64__ctrl_warm" \
    --log "random init=reports/runs/binkit_x86_64__ctrl_scratch" \
    --out reports/figures/control_warm_vs_scratch.png
```

## Run records

Each run directory is named `<corpus>__<task>` and may contain:

- `train_log.jsonl`: one structured record per logged training step;
- `pretrain_args.json` or `finetune_args.json`: the exact launch arguments;
- `baseline.json`: the unigram reference for MLM runs;
- `finetune_result.json`: final task metrics.

These files make the published configuration and selected results auditable
without distributing checkpoint weights.
