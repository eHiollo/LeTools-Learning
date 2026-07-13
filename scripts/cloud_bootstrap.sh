#!/usr/bin/env bash
# 云端基建自检：Docker 权限、数据集、镜像、smoke
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck source=/dev/null
source "${REPO_ROOT}/deploy.env"

echo "==> LeTools 云端 bootstrap"
echo "    REPO_ROOT=${REPO_ROOT}"
echo "    CUDA_VER=${CUDA_VER}"
echo "    LEROBOT_DATA=${LEROBOT_DATA}"

# 1. Docker 权限
if ! docker info &>/dev/null; then
  echo ""
  echo "ERROR: 当前用户无法访问 Docker（permission denied）。"
  echo "请在本机终端执行（需输入 sudo 密码）："
  echo "  sudo usermod -aG docker ${USER}"
  echo "  newgrp docker"
  echo "然后重新运行本脚本。"
  exit 1
fi

# 2. NVIDIA Container Toolkit
if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "WARN: docker info 未显示 nvidia runtime，--gpus all 可能失败。"
  echo "若 smoke 报 GPU 错误，请管理员安装 nvidia-container-toolkit 并重启 docker。"
fi

# 3. 数据目录
mkdir -p "${DATA_ROOT}"/{lerobot,models,hf_cache,outputs,rosbag}
mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${LEROBOT_DATA}/meta/info.json" ]]; then
  echo ""
  echo "ERROR: 数据集未就绪: ${LEROBOT_DATA}/meta/info.json 不存在"
  echo "请从本地 rsync 数据，例如："
  echo "  rsync -avP <本地路径>/lerobot_v3.0/ ${USER}@<云电脑IP>:${LEROBOT_DATA}/"
  exit 1
fi
echo "OK: 数据集 meta/info.json 存在"

# 4. 构建镜像（若不存在）
if ! docker image inspect "${LETOOLS_IMAGE_LOCAL}" &>/dev/null; then
  echo "==> 构建镜像 ${LETOOLS_IMAGE_LOCAL}（约 40GB，耗时较长）..."
  docker build -f docker/Dockerfile.letools \
    --build-arg CUDA_VER="${CUDA_VER}" \
    -t "${LETOOLS_IMAGE_LOCAL}" \
    . 2>&1 | tee docker/build_cloud.log
else
  echo "OK: 镜像 ${LETOOLS_IMAGE_LOCAL} 已存在"
fi

# 5. CUDA 验证
echo "==> 验证容器内 CUDA..."
chmod +x docker/run_train.sh
./docker/run_train.sh python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# 6. ACT smoke（复用 configs/train/smoke）
echo "==> ACT smoke test（10 步）..."
./docker/run_train.sh python kuavo_model/train.py \
  --policy act --mode simple --config-root configs/train/smoke

echo ""
echo "==> Bootstrap 完成。完整训练示例（tmux 内执行）："
echo "  tmux new -s train"
echo "  cd ${REPO_ROOT} && source deploy.env"
echo "  bash scripts/cloud_train.sh diffusion"
echo ""
echo "若无 Docker 权限，改用："
echo "  bash scripts/native_train.sh --policy diffusion --mode simple --config-root configs/train/cloud"
