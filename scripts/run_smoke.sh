#!/usr/bin/env bash
# End-to-end pipeline check on the locally built dataset.
#
# Runs every stage — corpus, MLM pre-training, fine-tuning, voting evaluation —
# at a size that finishes in a few minutes. The point is to prove the plumbing
# works, not to produce meaningful accuracy: a couple of hundred steps on two
# compiler versions of one architecture is nowhere near the paper's setting.
# See docs/REPRODUCTION.md.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=7 scripts/run_smoke.sh
#
# Set FULL=1 for a longer run (more epochs, all five tasks).

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python3}"
CORPUS="${CORPUS:-data/corpus/local}"
LOCAL_DATA="${LOCAL_DATA:-data/local}"
CKPT="${CKPT:-checkpoints/smoke}"
RESULTS="${RESULTS:-results/smoke}"
FULL="${FULL:-0}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is unset. The GPUs on this host are shared and" >&2
  echo "usually memory-full; pick a free one before launching:" >&2
  echo "  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader | sort -t, -k2 -rn" >&2
  exit 2
fi

# Two settings differ from the paper's, both because this corpus is ~1000x
# smaller than BinKit and both established by measurement (the table in
# docs/REPRODUCTION.md, "Training failure mode you will probably hit"):
#
#  * Model size. A 12-layer/768-hidden encoder does not train on a few thousand
#    sequences: at lr 1e-4 it collapses outright (constant logits, spread 0.14,
#    accuracy 53.7%), and even at a working lr it only reached 69.8% where a
#    4-layer/256-hidden model reached 90.6% in the same 300 steps. The paper's
#    size is right for BinKit and wrong here, so the smoke test uses a small one.
#  * Learning rate. Raising it does NOT compensate for the weak warm start --
#    that was the first thing tried here and it caused the collapse above. For
#    the 12-layer model the ordering was 1e-4 (collapse) < 3e-5 (69.8%) <
#    1e-5 (80.6%): on small data, lower is better. The script default of 3e-5
#    is left alone.
ARCH_FLAGS="--layers 4 --hidden 256 --heads 4 --intermediate 1024"
if [[ "$FULL" == 1 ]]; then
  MLM_EPOCHS=20; FT_EPOCHS=6
  TASKS=(compiler opt_hl opt4 opt_o0o1 opt_o2o3)
else
  MLM_EPOCHS=5; FT_EPOCHS=2
  TASKS=(compiler opt4)
  echo "NOTE: quick mode -- fewer epochs and two tasks. Use FULL=1 for all five."
fi

echo "############ 1/4  corpus ############"
if [[ ! -f "$CORPUS/text.u8" ]]; then
  if [[ ! -d "$LOCAL_DATA" ]]; then
    echo "no dataset at $LOCAL_DATA; build one first:" >&2
    echo "  $PY scripts/build_local_dataset.py --out $LOCAL_DATA --source synthetic gnu" >&2
    exit 1
  fi
  $PY scripts/build_corpus.py --root "$LOCAL_DATA" --out "$CORPUS" \
      --arch x86_64 --compiler gcc clang --opt O0 O1 O2 O3 --workers 32
else
  echo "reusing $CORPUS"
fi

echo
echo "############ 2/4  MLM pre-training ############"
# --split-name train, not the default 'pretrain'. The paper's rule (one program
# per package, all its variants) yields a large set on BinKit's 51 packages, but
# only ~2k sequences on a local corpus this size -- far too little for MLM to get
# past unigram statistics. On a small corpus, pre-train on everything in train.
# shellcheck disable=SC2086
$PY scripts/pretrain_mlm.py --corpus "$CORPUS" --out "$CKPT/mlm" \
    --split-name train $ARCH_FLAGS \
    --epochs "$MLM_EPOCHS" --batch-size 32 --workers 4 --log-every 100

echo
echo "############ 3/4  fine-tuning ############"
for task in "${TASKS[@]}"; do
  echo "--- $task ---"
  # shellcheck disable=SC2086
  $PY scripts/finetune.py --corpus "$CORPUS" --task "$task" \
      --init-from "$CKPT/mlm" --out "$CKPT/$task" \
      --epochs "$FT_EPOCHS" --batch-size 32 --workers 4 --log-every 100
done

echo
echo "############ 4/4  evaluation with voting ############"
CKPT_ARGS=()
for task in "${TASKS[@]}"; do CKPT_ARGS+=(--ckpt "$task=$CKPT/$task"); done
$PY scripts/evaluate.py --corpus "$CORPUS" --out "$RESULTS" \
    --batch-size 64 --workers 4 "${CKPT_ARGS[@]}"

echo
echo "pipeline OK. tables -> $RESULTS/tables.md"
echo "Remember: these numbers validate the code, not the paper."
