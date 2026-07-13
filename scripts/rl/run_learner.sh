#!/usr/bin/env bash
# Stage-B learner launcher (requires lerobot[hilserl]).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/rl/gym_hil_baseline.json}"
echo "[run_learner] Starting learner FIRST with $CONFIG"
echo "[run_learner] Target command (only after hilserl install):"
echo "  python -m lerobot.rl.learner --config_path $CONFIG"
if python -c "import lerobot.rl.learner" 2>/dev/null; then
  python -m lerobot.rl.learner --config_path "$CONFIG"
else
  echo "[run_learner] lerobot.rl.learner not importable yet — install hilserl extra first."
  exit 2
fi
