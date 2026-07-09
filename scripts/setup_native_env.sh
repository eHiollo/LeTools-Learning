#!/usr/bin/env bash
# 无 Docker 权限时的本机 conda 训练环境（与 docker/setup_env_docker.sh 对齐）
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_SH="${HOME}/miniforge3/etc/profile.d/conda.sh"
if [[ ! -f "${CONDA_SH}" ]]; then
  echo "ERROR: miniforge 未安装，请先运行 scripts/install_miniforge.sh"
  exit 1
fi
# shellcheck source=/dev/null
source "${CONDA_SH}"
set +u
conda activate letools
set -u

export CUDA_VER="${CUDA_VER:-12.8}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${REPO_ROOT}/data/hf_cache"
export HF_LEROBOT_HOME="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"

if ! python -c "import torch" 2>/dev/null; then
  echo "==> 安装 cuda-toolkit ${CUDA_VER} + 项目依赖（首次较慢）..."
  conda install -y -c nvidia "cuda-toolkit=${CUDA_VER}"
  bash docker/setup_env_docker.sh 2>&1 | tee "${REPO_ROOT}/native_env_setup.log"
fi

echo "==> 验证 CUDA..."
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
