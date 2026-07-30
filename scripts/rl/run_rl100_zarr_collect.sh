#!/usr/bin/env bash
# RL-100 zarr live collect on Kuavo real + Quest (upper cams).
#
# Usage:
#   bash scripts/rl/run_rl100_zarr_collect.sh preflight --check-ros
#   bash scripts/rl/run_rl100_zarr_collect.sh collect --confirm-live --build-after
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate letools-rl

set +u
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
if [[ -f "${HOME}/kuavo_ros_application/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/kuavo_ros_application/devel/setup.bash"
fi
set -u

# Keep workspace kuavo_msgs first (sensorsData MD5). Missing footPose6D is
# injected in Python by kuavo_rl.ros_msg_compat — do NOT prepend SDK msg path.
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://kuavo_master:11311}"
export ROS_IP="${ROS_IP:-192.168.26.12}"

CFG="${RL100_CFG:-configs/rl/rl100_zarr_collect_upper_cams.yaml}"
echo "[rl100] config=${CFG} ROS_MASTER_URI=${ROS_MASTER_URI} ROS_IP=${ROS_IP}"
exec python -u scripts/rl/collect_rl100_zarr.py --config "${CFG}" "$@"
