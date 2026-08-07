#!/usr/bin/env bash
# RL-100 real-robot deploy (inspect / ros-preflight / shadow / live).
#
# Usage:
#   bash scripts/rl/run_rl100_real.sh inspect-checkpoint --config configs/rl/rl100_real_deploy.yaml
#   bash scripts/rl/run_rl100_real.sh ros-preflight --config configs/rl/rl100_real_deploy.yaml --duration-s 60
#   bash scripts/rl/run_rl100_real.sh shadow --config configs/rl/rl100_real_deploy.yaml --max-steps 20
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Machine-local overrides (ROS_MASTER_URI / ROS_IP / CUDA staging, ...).
# Copy configs/rl/local/env.sh.example → configs/rl/local/env.sh and edit.
if [[ -f "${ROOT_DIR}/configs/rl/local/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/configs/rl/local/env.sh"
fi

# shellcheck disable=SC1091
source "${HOME}/miniforge3/etc/profile.d/conda.sh"
conda activate letools-rl

# Prefer conda libstdc++ before ROS pulls older system CXXABI (same as collect).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Torch 2.2.0 on Orin was built against CUDA 11.4; host JetPack often ships
# CUDA 12 only. Auto-prepend staged CUDA 11.4 + cuDNN 8 when present.
# env.sh may already have set this; keep staging first either way.
_CUDA114="${ROOT_DIR}/.runtime_staging/cuda11.4"
_CUDA114_LIB="${_CUDA114}/usr/local/cuda-11.4/targets/sbsa-linux/lib"
_CUDNN_LIB="${_CUDA114}/usr/lib/aarch64-linux-gnu"
if [[ -d "${_CUDA114_LIB}" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${_CUDA114_LIB}:"*) ;;
    *)
      export LD_LIBRARY_PATH="${_CUDA114_LIB}:${_CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
      ;;
  esac
  echo "[rl100-real] staged CUDA 11.4 runtime: ${_CUDA114}"
fi
unset _CUDA114 _CUDA114_LIB _CUDNN_LIB

set +u
# shellcheck disable=SC1091
source /opt/ros/noetic/setup.bash
if [[ -f "${HOME}/kuavo_ros_application/devel/setup.bash" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/kuavo_ros_application/devel/setup.bash"
fi
set -u

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://kuavo_master:11311}"
export ROS_IP="${ROS_IP:-192.168.26.12}"
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/third_party/RL-100/RL-100${PYTHONPATH:+:$PYTHONPATH}"

echo "[rl100-real] ROS_MASTER_URI=${ROS_MASTER_URI} ROS_IP=${ROS_IP}"

exec python scripts/rl/run_rl100_real.py "$@"
