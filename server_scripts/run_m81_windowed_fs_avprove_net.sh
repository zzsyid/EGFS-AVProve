#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/path/to/EGFS-AVProve}"
PYTHON="${PYTHON:-/path/to/runtime/anaconda3/envs/iap/bin/python}"
OUT_DIR="${OUT_DIR:-${ROOT}/rq1_eval/artifacts_m81_windowed_fs_avprove_net}"

cd "${ROOT}"

# Keep this on physical GPU 0. GPU 1 may be reserved for other training.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PYTHON}" rq1_eval/train_m81_windowed_fs_avprove_net.py \
  --out-dir "${OUT_DIR}" \
  --device cuda:0 \
  --epochs "${M81_EPOCHS:-30}" \
  --batch-size "${M81_BATCH_SIZE:-128}" \
  --hidden-dim "${M81_HIDDEN_DIM:-128}" \
  --num-heads "${M81_NUM_HEADS:-4}" \
  --layers "${M81_LAYERS:-2}" \
  --predicate-loss-weight "${M81_PREDICATE_LOSS_WEIGHT:-0.35}" \
  --counterfactual-dropout "${M81_COUNTERFACTUAL_DROPOUT:-0.15}" \
  --limit-carrier "${M81_LIMIT_CARRIER:-12000}"
