#!/usr/bin/env bash
# Host ROS bridge for Stage-B Docker actor (requires live Kuavo-Sim + RGB cams).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate data

set +u
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
# shellcheck disable=SC1091
source /home/fulin/VSCode/kuavo-ros-control/devel/setup.bash
set -u

KUAVO_PY=/home/fulin/VSCode/kuavo-ros-control/devel/lib/python3/dist-packages
APP_PY=/home/fulin/VSCode/new_pkg/kuavo_ros_application/devel/lib/python3/dist-packages
SDK=/home/fulin/VSCode/kuavo-ros-control/src/kuavo_humanoid_sdk
export PYTHONPATH="${ROOT}:${SDK}:${KUAVO_PY}:${APP_PY}:${PYTHONPATH:-}"

PORT="${KUAVO_ROS_BRIDGE_PORT:-8877}"
DEPLOY_CFG="${DEPLOY_CFG:-configs/deploy/total/deploy_sim_smoke_cams_total.yaml}"
IMAGE_H="${IMAGE_H:-128}"
IMAGE_W="${IMAGE_W:-128}"

exec python -u scripts/rl/kuavo_ros_bridge_server.py \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --deploy-config "${DEPLOY_CFG}" \
  --image-h "${IMAGE_H}" \
  --image-w "${IMAGE_W}"
