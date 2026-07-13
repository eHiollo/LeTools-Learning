#!/usr/bin/env bash
# Host ROS + Kuavo-Sim ACT execute-first eval (remote infer in Docker).
# Prerequisites:
#   1) Kuavo-Sim v62 wheel launch running (in Kuavo docker)
#   2) bash scripts/rl/run_sim_rgb_cameras.sh
#   3) bash scripts/rl/run_act_infer_server.sh
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
DEPLOY_CFG="${DEPLOY_CFG:-configs/deploy/total/deploy_sim_smoke_cams_total.yaml}"
RL_CFG="${RL_CFG:-configs/rl/kuavo_hilserl_sim_act.yaml}"
HOST="${ACT_INFER_HOST:-127.0.0.1}"
PORT="${ACT_INFER_PORT:-8765}"
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
  "${SHADOW_ARGS[@]}"
