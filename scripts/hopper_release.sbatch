#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Slurm batch script for the BinProv Hopper (CUDA) release training.
#
# Default: A100 80 GB on the public gpuq partition.
# To use an H100 contributor node instead, supply at sbatch time:
#   sbatch -p contrib-H100 --qos=gpu --gres=gpu:H100.80gb:1 scripts/hopper_release.sbatch
# (No Slurm account name is embedded in this script.)
# ---------------------------------------------------------------------------
#SBATCH --job-name=binprov-hop
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# A one-day request fits Hopper's current H100 backfill window and still leaves
# substantial margin over the measured 7.5--9 h H200 reference run.
#SBATCH --time=24:00:00
#SBATCH --requeue
# Keep Slurm's own log in the submission directory. Slurm opens it before this
# script runs, so a not-yet-created logs/ directory is unsafe.
#SBATCH --output=hopper_release_%j.out
#SBATCH --error=hopper_release_%j.err

set -euo pipefail

# ---- prerequisites --------------------------------------------------------

REPO_DIR="${BINPROV_REPO:-${SCRATCH:-/scratch/${USER}}/binprov}"
VENV_DIR="${REPO_DIR}/.venv-hopper"

if [ ! -d "$REPO_DIR" ]; then
  echo "ERROR: staged repo not found at $REPO_DIR" >&2
  echo "Clone or copy the repository there before submitting." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "ERROR: virtualenv not found at $VENV_DIR" >&2
  echo "Create it first, e.g.:" >&2
  echo "  module load python/3.12.1-33 && python -m venv $VENV_DIR && $VENV_DIR/bin/pip install -r $REPO_DIR/requirements-hopper.txt" >&2
  exit 1
fi

module purge
module load hosts/hopper gnu/12.3.0 python/3.12.1-33 cuda/12.6.3

# ---- diagnostics ----------------------------------------------------------

echo "=== BinProv Hopper release ==="
echo "hostname : $(hostname)"
echo "date     : $(date -Iseconds)"
echo "repo     : $REPO_DIR"
echo "venv     : $VENV_DIR"
echo "python   : $VENV_DIR/bin/python"
echo "GPU env  : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
$VENV_DIR/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>/dev/null || echo "torch/cuda check failed"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"

# ---- training -------------------------------------------------------------

cd "$REPO_DIR"

exec $VENV_DIR/bin/python scripts/train_local_release.py \
  --config configs/hopper_release.json \
  --profile opt4_wide_seed29 \
  --resume \
  --execute
