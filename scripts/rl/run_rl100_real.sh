#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /opt/ros/noetic/setup.bash
if [[ -f "$HOME/kuavo_ros_application/devel/setup.bash" ]]; then
  source "$HOME/kuavo_ros_application/devel/setup.bash"
fi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/third_party/RL-100/RL-100${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"
exec python3 scripts/rl/run_rl100_real.py "$@"
