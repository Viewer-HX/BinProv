# Hopper (CUDA) release training

This is the *bounded, Slurm-managed* path to reproduce the best-wide
`opt4_wide_seed29` model on GMU Hopper using A100 80 GB (or H100 contributor)
GPUs. It reuses the same training driver and config structure as the local MPS
path but with CUDA micro-batches sized for 80 GB GPUs and data-loader workers.

All generated output lands under `results/hopper_release/<profile>/` — never on
the archived paths the paper numbers were produced from.

## Files

| file | role |
|---|---|
| `configs/hopper_release.json` | CUDA profile with conservative 80 GB micro-batches |
| `scripts/train_local_release.py --config` | the driver (dry-run default) |
| `scripts/hopper_release.sbatch` | Slurm batch script |
| `scripts/export_hf.py` | builds the upload-ready directory |

## Staging

Compute-node `/home` is read-only. Work lives under `$SCRATCH` (typically
`/scratch/$USER`).

```bash
# 1. Clone or copy the repository under scratch
REPO=/scratch/$USER/binprov
git clone <repo-url> "$REPO"   # or cp -r the existing checkout
cd "$REPO"

# 2. Create a CUDA venv on a login node (or an interactive GPU node)
module load hosts/hopper gnu/12.3.0 python/3.12.1-33 cuda/12.6.3
python -m venv .venv-hopper
.venv-hopper/bin/pip install --upgrade pip
.venv-hopper/bin/pip install -r requirements-hopper.txt

# 3. Verify imports. Login nodes have no GPU; the batch job checks CUDA after
# Slurm assigns a compute node.
.venv-hopper/bin/python -c "import torch, transformers; print(torch.__version__, transformers.__version__)"
```

The venv must be created **once** and must remain at `$REPO/.venv-hopper`. The
batch script and the training driver do not install anything at compute time —
no network access is required inside the job.

## Submitting

### Default (A100 80 GB)

```bash
sbatch scripts/hopper_release.sbatch
```

### Contributor H100 override

Pass the partition/QOS/GRES at submission time; the script's defaults are
overridden automatically:

```bash
sbatch \
  -p contrib-H100 \
  --qos=gpu \
  --gres=gpu:H100.80gb:1 \
  scripts/hopper_release.sbatch
```

## Monitoring

```bash
# queue status
squeue -u $USER

# finished jobs
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed

# live tail of output (job ID from squeue)
tail -f hopper_release_<JOBID>.out
```

## Output locations

| what | path |
|---|---|
| training output | `results/hopper_release/opt4_wide_seed29/<phase>/` |
| status file | `results/hopper_release/status.json` |
| logs (Slurm) | `hopper_release_<JOBID>.out` / `.err` in the submission directory |
| logs (training) | `results/hopper_release/opt4_wide_seed29/logs/` |
| HF export | `results/hopper_release/opt4_wide_seed29/export/` |

## Resuming and retry

The training driver enforces a safe resume policy:

- **`--resume`** is passed by the batch script. A stage is only relaunched when
  its saved `args.json`/`pretrain_args.json` exactly matches the profile's
  recipe.
- **Partial output** (interrupted mid-epoch) is resumed from the last
  `training_state.pt` snapshot.
- **Complete output** is skipped.
- **Incompatible output** (different recipe) is refused — move it aside first.
- **Requeue** (`#SBATCH --requeue`): if the job is preempted or times out,
  Slurm automatically resubmits it. The driver picks up from the last snapshot.

To restart a stage from scratch, move its output directory aside:

```bash
mv results/hopper_release/opt4_wide_seed29/mlm512 results/hopper_release/opt4_wide_seed29/mlm512.bak
```

## Hugging Face export

After training completes, export the model:

```bash
# from a login node or another job
.venv-hopper/bin/python scripts/train_local_release.py \
  --config configs/hopper_release.json \
  --profile opt4_wide_seed29 \
  --phase export \
  --execute
```

The export lands at `results/hopper_release/opt4_wide_seed29/export/`. It is
self-contained and ready for upload, but **nothing is uploaded automatically**.

## Verified A100 run

Job `9558937` completed this exact recipe on one A100 80 GB on 2026-09-06 with
exit code `0:0` and no restart. Total elapsed time was **12:42:09**:

| milestone | completion time (EDT) |
|---|---|
| MLM512 | 2026-09-05 17:29:14 |
| MLM2048 | 2026-09-05 20:53:31 |
| fine-tune and evaluation | 2026-09-06 03:48:35 |
| export | 2026-09-06 03:49:04 |

The best fine-tuning epoch (4) produced **84.2255% sequence accuracy** and
**93.8830% binary soft-vote accuracy**. The exported model passed a CPU load and
four-window prediction check. Compact evidence is committed at
[`reports/runs/release/opt4_wide_seed29`](../reports/runs/release/opt4_wide_seed29/README.md).

## Duration estimate

The archived run on one NVIDIA H200 took roughly:

| stage | wall-clock |
|---|---|
| MLM512 (10 epochs, eff. batch 256) | ~1 h 11 min |
| MLM2048 (10 epochs, eff. batch 64) | ~2 h 15 min |
| opt4 finetune (6 epochs, eff. batch 32) | ~3 h 15 min |
| evaluation and export | additional time |
| **full release pipeline** | **~7.5–9 h** |

Those archived stages used a 141 GB H200. The conservative 80 GB micro-batches
in this profile add some overhead. The verified A100 run took **12 h 42 min**;
retain the 24-hour Slurm request to cover queue-dependent hardware variation and
resume overhead.

## See also

- Local (MPS) training path: `docs/LOCAL_TRAINING.md`
- Archived numbers: `docs/BEST_RESULTS.md`
- Gradient-accumulation correctness: `tests/test_grad_accum.py`
