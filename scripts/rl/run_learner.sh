#!/usr/bin/env bash
# Stage-B learner launcher. Uses kuavo_rl runtime patches (does not edit third_party).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/rl/gym_hil_baseline.json}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
echo "[run_learner] Starting learner FIRST with $CONFIG"
echo "[run_learner] python -m kuavo_rl.hilserl_cli learner --config_path $CONFIG"
if python -c "import lerobot.rl.learner" 2>/dev/null; then
  python -m kuavo_rl.hilserl_cli learner --config_path "$CONFIG"
else
  echo "[run_learner] lerobot.rl.learner not importable yet — install hilserl extra first."
  exit 2
fi
