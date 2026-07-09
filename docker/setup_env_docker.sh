#!/bin/bash
# 非交互版 setup_env.sh，供 Docker 镜像构建使用。
set -euo pipefail

echo "==> pip 换源"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

echo "==> 跳过 requirements_ros_env.txt（训练镜像不需要；部署/仿真运行时 source ROS 即可）"

echo "==> 主项目依赖"
pip install -r requirements.txt

echo "==> lerobot 基础 + 训练"
python -m pip install -e "third_party/lerobot[training,dataset]"

echo "==> diffusion / pi 系列依赖（ACT/Diffusion/PI0/PI05 lerobot 版）"
python -m pip install -e "third_party/lerobot[diffusion]"
python -m pip install -e "third_party/lerobot[pi,peft]"

echo "==> ffmpeg / pyarrow / pyaudio"
conda install -y ffmpeg=6.1.1
pip uninstall -y pyarrow 2>/dev/null || true
pip install pyarrow==21.0.0
conda install -y pyaudio

echo "==> 重新安装 lerobot + 本项目"
python -m pip install -e "third_party/lerobot[training,dataset]"
pip install -e .

echo "==> 跳过 flash-attn（gr00t 按需另建镜像时再装）"

echo "==> HF 镜像"
grep -q 'HF_ENDPOINT=https://hf-mirror.com' /root/.bashrc 2>/dev/null || \
  echo 'export HF_ENDPOINT=https://hf-mirror.com' >> /root/.bashrc

echo "✅ Docker 环境安装完成"
