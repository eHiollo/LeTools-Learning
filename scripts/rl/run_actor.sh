#!/usr/bin/env bash
# Stage-B actor launcher (start AFTER learner).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:-configs/rl/gym_hil_baseline.json}"
echo "[run_actor] Starting actor AFTER learner with $CONFIG"
echo "[run_actor] For Kuavo real/sim, switch to kuavo_rl adapter once wired:"
echo "  # future: python -m kuavo_rl.actor --config-path configs/rl/kuavo_hilserl_sim.yaml"
echo "[run_actor] Current upstream smoke target:"
echo "  python -m lerobot.rl.actor --config_path $CONFIG"
if python -c "import lerobot.rl.actor" 2>/dev/null; then
  python -m lerobot.rl.actor --config_path "$CONFIG"
else
  echo "[run_actor] lerobot.rl.actor not importable yet — install hilserl extra first."
  exit 2
fi
