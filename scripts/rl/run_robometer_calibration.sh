#!/usr/bin/env bash
# Offline Robometer calibration + VRAM probe inside hilserl image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
STUB="${STUB:-0}"
OUT_DIR="${OUT_DIR:-data/reward_calibration}"
mkdir -p "${OUT_DIR}"

EXTRA_ARGS=()
if [[ "${STUB}" == "1" ]]; then
  EXTRA_ARGS+=(--stub)
fi

echo "[robometer] image=${IMAGE} stub=${STUB} out=${OUT_DIR}"

docker run --rm -i --gpus all --network host \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -e HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-15}" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/LeTools-Learning \
  "$IMAGE" \
  bash -s <<INNER
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
set +u; conda activate letools; set -u
export PYTHONPATH="/workspace/LeTools-Learning:\${PYTHONPATH:-}"

# Fail fast on unreachable hubs instead of multi-minute retries
export HF_HUB_DOWNLOAD_TIMEOUT="\${HF_HUB_DOWNLOAD_TIMEOUT:-15}"

python scripts/rl/score_rollouts_robometer.py \
  --dataset data/lerobot/lerobot_merged \
  --out ${OUT_DIR}/offline_scores.json \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

python scripts/rl/probe_robometer_vram.py \
  --out ${OUT_DIR}/vram_budget.json \
  ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}

echo "ROBOMETER_CALIBRATION_DONE out=${OUT_DIR}"
INNER
