#!/usr/bin/env bash
# 本机 conda 训练（无 Docker 权限时使用）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/deploy.env" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/deploy.env"
else
  echo "WARN: deploy.env 不存在，使用 deploy.env.example"
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/deploy.env.example"
fi

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
set +u
conda activate letools
set -u

export HF_HOME="${HF_CACHE}"
export HF_LEROBOT_HOME="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

_resolve_config_root() {
  local config_root="$1"
  if [[ "${config_root}" != /* ]]; then
    config_root="${REPO_ROOT}/${config_root}"
  fi
  local tmp
  tmp="$(mktemp -d)"
  cp -r "${config_root}/." "${tmp}/"
  find "${tmp}" -name '*.yaml' -exec sed -i "s|/workspace/LeTools-Learning|${REPO_ROOT}|g" {} +
  if [[ -n "${LEROBOT_DATA:-}" ]]; then
    find "${tmp}" -name '*_total.yaml' -exec sed -i \
      "s|root: \"${REPO_ROOT}/data/lerobot/[^\"]*\"|root: \"${LEROBOT_DATA}\"|g" {} +
  fi
  echo "${tmp}"
}

ARGS=()
CONFIG_ROOT=""
RESOLVED_CONFIG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-root)
      CONFIG_ROOT="$2"
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "${CONFIG_ROOT}" ]]; then
  RESOLVED_CONFIG="$(_resolve_config_root "${CONFIG_ROOT}")"
  trap 'rm -rf "${RESOLVED_CONFIG}"' EXIT
  ARGS+=(--config-root "${RESOLVED_CONFIG}")
fi

exec python kuavo_model/train.py "${ARGS[@]}"
