#!/usr/bin/env bash
# Install / verify HIL-SERL deps inside letools-train and commit as letools-train:hilserl.
#
# NOTE: plain `pip install -e "third_party/lerobot[hilserl]"` may fail building `evdev`
# (kernel headers missing KEY_* macros). We install core deps manually and allow
# keyboard gym_hil path without gamepad/evdev.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE_IN="${LETOOLS_IMAGE_LOCAL:-letools-train:lerobot-0.4.2}"
IMAGE_OUT="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
NAME="letools-hilserl-setup"

echo "[phase0] base image: $IMAGE_IN"
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -w /workspace/LeTools-Learning \
  "$IMAGE_IN" sleep infinity

docker exec "$NAME" bash -lc 'set -e
source /opt/conda/etc/profile.d/conda.sh && conda activate letools
python -V
python -m pip install -e "third_party/lerobot"
python -m pip install "grpcio>=1.73.1,<2.0.0" "protobuf>=6.31.1,<8.0.0"
# gym-hil 0.1.14 wants mujoco<3.9; pin if possible
python -m pip install "mujoco>=2.3.7,<3.9.0" || python -m pip install mujoco
python -m pip install glfw pygame imageio-ffmpeg pettingzoo placo pynput hidapi etils pyopengl || true
python -m pip install "gym-hil>=0.1.14,<0.2.0" --no-deps
python -m pip install "gym-hil>=0.1.14,<0.2.0" || true
python - <<"PY"
import grpc, mujoco, torch
import gym_hil
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
print("HIL-SERL imports OK; CUDA:", torch.cuda.is_available())
PY
python -m lerobot.rl.actor --help >/dev/null
python -m lerobot.rl.learner --help >/dev/null
echo PHASE0_OK
'

echo "[phase0] committing $IMAGE_OUT"
docker commit "$NAME" "$IMAGE_OUT"
docker rm -f "$NAME" >/dev/null
echo "[phase0] done -> $IMAGE_OUT"
