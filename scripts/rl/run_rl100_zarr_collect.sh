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

# Preload libgomp so torch's bundled libgomp-a49a47f9.so.1.0.0 can reuse an
# already-reserved static TLS slot instead of failing with
#   "cannot allocate memory in static TLS block"
# Must happen before python starts.
#
# IMPORTANT: torch's bundled libgomp has a custom SONAME
# (libgomp-a49a47f9.so.1.0.0) that differs from the system libgomp.so.1, so
# preloading the system libgomp does NOT prevent torch from loading its own —
# they occupy separate TLS slots. After sourcing ROS noetic + kuavo workspace,
# other ROS libs fill the static TLS block and torch's own libgomp then fails
# to allocate. The fix is to preload torch's OWN libgomp so it grabs the TLS
# slot first; torch reuses it on later load.
#
# We OVERRIDE any pre-existing LD_PRELOAD that does not already point at torch's
# libgomp: a stale system-libgomp preload (e.g. left in the shell from a prior
# `export LD_PRELOAD=/usr/lib/.../libgomp.so.1`) silently breaks torch here
# because it makes `import torch` work in isolation (no ROS sourced → TLS free)
# but fails inside this script (ROS libs fill TLS). Set RL100_KEEP_LD_PRELOAD=1
# to opt out and keep your own LD_PRELOAD verbatim.
_torch_libs="${CONDA_PREFIX:-}/lib/python3.10/site-packages/torch.libs"
_rl100_gomp=""
for _gomp in \
  "${_torch_libs}/libgomp-a49a47f9.so.1.0.0" \
  "${_torch_libs}/libgomp.so.1" \
  "/usr/lib/aarch64-linux-gnu/libgomp.so.1" \
  "/usr/lib/x86_64-linux-gnu/libgomp.so.1"; do
  if [[ -f "$_gomp" ]]; then
    _rl100_gomp="$_gomp"
    break
  fi
done
unset _torch_libs _gomp

if [[ -n "${RL100_KEEP_LD_PRELOAD:-}" ]]; then
  echo "[rl100] RL100_KEEP_LD_PRELOAD=1 → keeping LD_PRELOAD=[${LD_PRELOAD:-}]"
elif [[ -n "$_rl100_gomp" ]]; then
  if [[ -n "${LD_PRELOAD:-}" && "$LD_PRELOAD" != "$_rl100_gomp" ]]; then
    echo "[rl100] overriding stale LD_PRELOAD=[${LD_PRELOAD}] → [${_rl100_gomp}]"
  fi
  export LD_PRELOAD="$_rl100_gomp"
  echo "[rl100] LD_PRELOAD=${_rl100_gomp} (libgomp static-TLS workaround)"
else
  echo "[rl100] WARNING: no libgomp candidate found; torch may fail to import"
fi
unset _rl100_gomp

exec python -u scripts/rl/collect_rl100_zarr.py --config "${CFG}" "$@"
