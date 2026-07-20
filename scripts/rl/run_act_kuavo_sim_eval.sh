#!/usr/bin/env bash
# Host ROS + Kuavo-Sim ACT execute-first eval (remote infer in Docker).
# Prerequisites:
#   1) ROS sim with native cameras:
#        export ROBOT_VERSION=62
#        roslaunch humanoid_controllers load_kuavo_mujoco_sim_wheel.launch publish_camera:=true
#   2) bash scripts/rl/run_act_infer_server.sh
# Note: host-side run_sim_rgb_cameras.sh is obsolete when publish_camera:=true.
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

STEPS="${1:-10}"
DEPLOY_CFG="${DEPLOY_CFG:-configs/deploy/total/deploy_sim_mujoco_native_cams.yaml}"
RL_CFG="${RL_CFG:-configs/rl/kuavo_hilserl_sim_act.yaml}"
HOST="${ACT_INFER_HOST:-127.0.0.1}"
PORT="${ACT_INFER_PORT:-8765}"
HIL_ARGS=()
if [[ "${HIL_RECORDING:-0}" == "1" ]]; then
  HIL_ARGS+=(--hil-recording)
  if [[ "${HIL_LIVE_ROSBAG:-0}" == "1" ]]; then
    HIL_ARGS+=(--hil-recording-live-rosbag)
    HIL_ARGS+=(--hil-topics-profile "${HIL_TOPICS_PROFILE:-configs/rl/hil_topics_sim_v002.yaml}")
  fi
fi
SHADOW_ARGS=()
if [[ "${SHADOW:-0}" == "1" ]]; then
  SHADOW_ARGS+=(--shadow)
  DEFAULT_MANIFEST="data/rl_runs/act_kuavo_sim_shadow/manifest.json"
else
  DEFAULT_MANIFEST="data/rl_runs/act_kuavo_sim_eval/manifest.json"
fi
MANIFEST="${MANIFEST:-$DEFAULT_MANIFEST}"

exec python -u scripts/rl/eval_act_execute_first.py \
  --kuavo-env \
  --policy remote \
  --infer-host "${HOST}" \
  --infer-port "${PORT}" \
  --deploy-config "${DEPLOY_CFG}" \
  --config "${RL_CFG}" \
  --steps "${STEPS}" \
  --manifest "${MANIFEST}" \
  --record-dir data/rl_runs/hilserl_sim_verify \
  --record-experiment hilserl_vr \
  "${HIL_ARGS[@]}" \
  "${SHADOW_ARGS[@]}"
