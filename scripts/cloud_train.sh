#!/usr/bin/env bash
# 云端完整训练（Docker 可用时用 run_train.sh，否则回退 native_train.sh）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

POLICY="${1:-diffusion}"
CONFIG_ROOT="${2:-configs/train/cloud}"

if [[ -f "${REPO_ROOT}/deploy.env" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/deploy.env"
fi

if docker info &>/dev/null; then
  chmod +x docker/run_train.sh
  exec ./docker/run_train.sh python kuavo_model/train.py \
    --policy "${POLICY}" --mode simple --config-root "${CONFIG_ROOT}"
else
  echo "WARN: Docker 不可用，使用本机 conda 训练"
  exec bash scripts/native_train.sh \
    --policy "${POLICY}" --mode simple --config-root "${CONFIG_ROOT}"
fi
