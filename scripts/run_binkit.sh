#!/usr/bin/env bash
# The paper's main experiment, end to end, on a BinKit corpus.
#
# Reproduces Tables 3-6 for one architecture (x86_64 by default, which the paper
# calls its worst case because x86's variable-length instructions get cut
# mid-instruction at sequence boundaries).
#
# Prerequisites:
#   1. scripts/fetch_binkit.sh normal
#   2. scripts/build_corpus.py  (see below, or let this script do it)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=7 scripts/run_binkit.sh
#   ARCH=arm_64 CUDA_VISIBLE_DEVICES=7 scripts/run_binkit.sh     # Table 7 ISA rows
#
# Runtime is dominated by MLM pre-training. Everything writes under /data.

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
ARCH="${ARCH:-x86_64}"
BINKIT="${BINKIT:-data/binkit/normal}"
CORPUS="${CORPUS:-data/corpus/$ARCH}"
CKPT="${CKPT:-checkpoints/$ARCH}"
RESULTS="${RESULTS:-results/$ARCH}"

MLM_EPOCHS="${MLM_EPOCHS:-10}"
FT_EPOCHS="${FT_EPOCHS:-5}"
BATCH="${BATCH:-64}"
TASKS=(compiler opt_hl opt4 opt_o0o1 opt_o2o3)

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Pin a GPU first -- this host is shared and mostly memory-full:" >&2
  echo "  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | sort -t, -k2 -rn" >&2
  exit 2
fi

echo "############ corpus ($ARCH) ############"
if [[ ! -f "$CORPUS/text.u8" ]]; then
  if [[ ! -d "$BINKIT" ]]; then
    echo "no BinKit tree at $BINKIT. Fetch it first:" >&2
    echo "  pip install gdown && scripts/fetch_binkit.sh normal" >&2
    exit 1
  fi
  # Always eyeball the parsed labels before a long build: BinKit's directory
  # naming has changed between releases.
  $PY scripts/build_corpus.py --root "$BINKIT" --dry-run \
      --arch "$ARCH" --compiler gcc clang --opt O0 O1 O2 O3 --extra normal
  $PY scripts/build_corpus.py --root "$BINKIT" --out "$CORPUS" \
      --arch "$ARCH" --compiler gcc clang --opt O0 O1 O2 O3 --extra normal
else
  echo "reusing $CORPUS"
fi

echo
echo "############ MLM pre-training (§3.2) ############"
if [[ ! -f "$CKPT/mlm/model.safetensors" && ! -f "$CKPT/mlm/pytorch_model.bin" ]]; then
  $PY scripts/pretrain_mlm.py --corpus "$CORPUS" --out "$CKPT/mlm" \
      --epochs "$MLM_EPOCHS" --batch-size "$BATCH" --workers 8 --resume
else
  echo "reusing $CKPT/mlm"
fi

echo
echo "############ fine-tuning (§3.3) ############"
for task in "${TASKS[@]}"; do
  echo "--- $task ---"
  $PY scripts/finetune.py --corpus "$CORPUS" --task "$task" \
      --init-from "$CKPT/mlm" --out "$CKPT/$task" \
      --epochs "$FT_EPOCHS" --batch-size "$BATCH" --workers 8 --resume
done

echo
echo "############ evaluation with voting (§3.4) ############"
CKPT_ARGS=()
for task in "${TASKS[@]}"; do CKPT_ARGS+=(--ckpt "$task=$CKPT/$task"); done
$PY scripts/evaluate.py --corpus "$CORPUS" --out "$RESULTS" \
    --batch-size $((BATCH * 2)) --workers 8 "${CKPT_ARGS[@]}"

cat <<EOF

Tables -> $RESULTS/tables.md

Compare against the paper (x86_64):
  Table 3  compiler 95.47%   opt H/L 98.90%   overall 94.77%   (sequence level)
  Table 4  O0/O1/O2/O3 91.07%   O0/O1 98.49%   O2/O3 83.64%
  Table 6  compiler 99.98% / 100%, opt H/L 99.40% / 100%, O2/O3 94.70% / 99.8%
           (function / binary level)

Read docs/REPRODUCTION.md before treating a gap as a bug -- several details the
paper leaves unspecified are interpreted there, and the function-level number in
particular depends on how sequences are assigned to functions.
EOF
