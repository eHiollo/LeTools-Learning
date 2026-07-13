#!/usr/bin/env bash
# Host-side RGB camera publisher synced to live Kuavo-Sim joint state.
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

export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Prefer EGL (NVIDIA present); override with MUJOCO_GL=glfw if you have a working DISPLAY.
KUAVO_PY=/home/fulin/VSCode/kuavo-ros-control/devel/lib/python3/dist-packages
export PYTHONPATH="${ROOT}:${KUAVO_PY}:${PYTHONPATH:-}"

exec python -u scripts/rl/publish_sim_rgb_cameras.py "$@"
