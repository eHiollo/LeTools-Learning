#!/usr/bin/env bash
# Host-side Kuavo-Sim smoke for kuavo_rl (requires live roslaunch in Kuavo docker).
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
# Joint-only fallback: DEPLOY_CFG=configs/deploy/total/deploy_sim_smoke_total.yaml
exec python -u scripts/rl/run_kuavo_sim_smoke.py \
  --kuavo-env \
  --deploy-config "${DEPLOY_CFG}" \
  --steps "${STEPS}"
