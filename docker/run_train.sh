#!/usr/bin/env bash
# 本机无 docker compose 时使用此脚本启动训练容器
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${LETOOLS_IMAGE_LOCAL:-letools-train:lerobot-0.4.2}"

docker run --rm -it --gpus all \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -e HF_HOME=/workspace/LeTools-Learning/data/hf_cache \
  -e HF_LEROBOT_HOME=/workspace/LeTools-Learning/data/hf_cache \
  -e TRANSFORMERS_CACHE=/workspace/LeTools-Learning/data/hf_cache \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -v "${REPO_ROOT}:/workspace/LeTools-Learning" \
  -v "${REPO_ROOT}/data:/workspace/LeTools-Learning/data" \
  -w /workspace/LeTools-Learning \
  "${IMAGE}" \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate letools && $*"
