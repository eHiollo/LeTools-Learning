#!/usr/bin/env bash
# Offline Stage-A ACT execute-first eval (no ROS required).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
CKPT="${1:-data/rl_runs/checkpoints/005000/pretrained_model}"
OUT="${ROOT}/data/rl_runs/act_execute_first_eval"
mkdir -p "$OUT"

echo "[eval-act] ckpt=$CKPT"
docker run --rm --gpus all \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -w /workspace/LeTools-Learning \
  "$IMAGE" \
  bash -lc "
source /opt/conda/etc/profile.d/conda.sh
set +u; conda activate letools; set -u
export PYTHONPATH=/workspace/LeTools-Learning
python scripts/rl/eval_act_execute_first.py \
  --checkpoint ${CKPT} \
  --dataset-root data/lerobot/lerobot_merged \
  --device cuda \
  --manifest data/rl_runs/act_execute_first_eval/manifest.json
"
echo "[eval-act] done -> data/rl_runs/act_execute_first_eval/manifest.json"
