#!/usr/bin/env bash
# RL-100 zarr live collect on Kuavo real + Quest (upper cams).
#
# Usage:
#   bash scripts/rl/run_rl100_zarr_collect.sh preflight --check-ros
#   bash scripts/rl/run_rl100_zarr_collect.sh collect --confirm-live
#   bash scripts/rl/run_rl100_zarr_collect.sh build --overwrite
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# Machine-local overrides (ROS_MASTER_URI, ROS_IP, wrist camera serials, ...).
# Copy configs/rl/local/env.sh.example → configs/rl/local/env.sh and edit.
# env.sh is gitignored; only the .example is committed. Sourced before conda so
# its values feed the ${VAR:-default} fallbacks below.
if [[ -f "${ROOT}/configs/rl/local/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/configs/rl/local/env.sh"
fi

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

# Preload libgomp so torch's bundled libgomp can reuse an already-reserved
# static TLS slot instead of failing with
#   "cannot allocate memory in static TLS block"
# Must happen before python starts.
#
# IMPORTANT: torch's bundled libgomp has a custom SONAME (e.g.
# libgomp-a49a47f9.so.1.0.0) that differs from the system libgomp.so.1, so
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
#
# Detection is fully dynamic: glob the conda env for any
# `python*/site-packages/torch.libs/libgomp*.so*` so neither the Python version
# nor the torch libgomp hash is hard-coded — survives torch/python upgrades.
_rl100_gomp=""
for _d in "${CONDA_PREFIX:-}"/lib/python*/site-packages/torch.libs; do
  [[ -d "$_d" ]] || continue
  for _f in "$_d"/libgomp*.so*; do
    [[ -f "$_f" ]] && { _rl100_gomp="$_f"; break 2; }
  done
done
unset _d _f
# Fallback: system libgomp (only if torch didn't ship its own).
if [[ -z "$_rl100_gomp" ]]; then
  for _gomp in /usr/lib/aarch64-linux-gnu/libgomp.so.1 /usr/lib/x86_64-linux-gnu/libgomp.so.1; do
    [[ -f "$_gomp" ]] && { _rl100_gomp="$_gomp"; break; }
  done
  unset _gomp
fi

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
