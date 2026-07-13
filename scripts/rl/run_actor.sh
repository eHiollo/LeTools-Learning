#!/usr/bin/env bash
# Stage-B actor launcher (start AFTER learner). Uses kuavo_rl runtime patches.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/rl/gym_hil_baseline.json}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
echo "[run_actor] Starting actor AFTER learner with $CONFIG"
echo "[run_actor] python -m kuavo_rl.hilserl_cli actor --config_path $CONFIG"
if python -c "import lerobot.rl.actor" 2>/dev/null; then
  python -m kuavo_rl.hilserl_cli actor --config_path "$CONFIG"
else
  echo "[run_actor] lerobot.rl.actor not importable yet — install hilserl extra first."
  exit 2
fi
