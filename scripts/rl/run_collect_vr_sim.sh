#!/usr/bin/env bash
# Kuavo-Sim VR teaching — LeRobot-style long session.
# Episodes have NO step timeout; each kept episode must end with B (success/fail/abort).
# After B success/failure: accepted_replay → auto Brain CvtRosbag2Lerobot → LeRobot v3.
#
# Prerequisites:
#   ROS sim + Quest teleop (/quest_joystick_data, /kuavo_arm_traj)
#
# Usage:
#   bash scripts/rl/run_collect_vr_sim.sh           # up to 50 episodes
#   bash scripts/rl/run_collect_vr_sim.sh 20        # up to 20 episodes
#   LIVE_ROSBAG=0 bash scripts/rl/run_collect_vr_sim.sh   # dry_run bag (no LeRobot export)
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

# Back-compat: old "MAX_STEPS EPISODES" — if two args, second is episodes.
if [[ "${1:-}" =~ ^[0-9]+$ ]] && [[ "${2:-}" =~ ^[0-9]+$ ]]; then
  EPISODES="${2}"
elif [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  EPISODES="${1}"
else
  EPISODES="${EPISODES:-50}"
fi

COL_CFG="${COL_CFG:-configs/rl/hil_collection_sim_v001.yaml}"
DEPLOY_CFG="${DEPLOY_CFG:-configs/deploy/total/deploy_sim_mujoco_native_cams.yaml}"
ENV_CFG="${ENV_CFG:-configs/rl/kuavo_hilserl_sim.yaml}"
OPERATOR="${OPERATOR:-fulin}"

EXTRA=()
# Default ON: real rosbag required for Brain → LeRobot v3. Set LIVE_ROSBAG=0 to disable.
if [[ "${LIVE_ROSBAG:-1}" == "1" ]]; then
  EXTRA+=(--live-rosbag)
fi
if [[ "${SINGLE:-0}" == "1" ]]; then
  EXTRA+=(--single-episode)
fi

echo "[run_collect_vr_sim] episodes=${EPISODES} operator=${OPERATOR}"
echo "[run_collect_vr_sim] No step limit — end each episode with B (click=ok, ×2=fail, hold=abort)"
echo "[run_collect_vr_sim] Session: RESET → RECORD → RESET …  (Y+↓ in RESET to quit)"
if [[ "${LIVE_ROSBAG:-1}" == "1" ]]; then
  echo "[run_collect_vr_sim] live rosbag ON → stage bags; session end: batch CvtRosbag2Lerobot → one LeRobot v3"
else
  echo "[run_collect_vr_sim] LIVE_ROSBAG=0 → dry_run bag; LeRobot auto-export will be skipped"
fi

exec python -u scripts/rl/collect_hil_dataset.py \
  --config "${COL_CFG}" \
  --mode vr_only \
  collect \
  --vr-sim \
  --confirm-live \
  --deploy-config "${DEPLOY_CFG}" \
  --env-config "${ENV_CFG}" \
  --episodes "${EPISODES}" \
  --operator "${OPERATOR}" \
  "${EXTRA[@]}"
