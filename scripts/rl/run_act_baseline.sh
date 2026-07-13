#!/usr/bin/env bash
# Stage-A ACT helpers.
# - Mock execute-first smoke (no checkpoint)
# - Full train: bash scripts/rl/train_act_stage_a.sh
# - Eval checkpoint: docker ... python scripts/rl/eval_act_execute_first.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
source deploy.env 2>/dev/null || true

CONFIG="${1:-configs/rl/act_kuavo_bc.yaml}"
echo "[run_act_baseline] config=$CONFIG"
echo "[run_act_baseline] Train:  bash scripts/rl/train_act_stage_a.sh"
echo "[run_act_baseline] Smoke:  bash scripts/rl/train_act_stage_a.sh configs/rl/act_stage_a_smoke.json"
echo "[run_act_baseline] Eval:   (in hilserl) python scripts/rl/eval_act_execute_first.py"
echo "[run_act_baseline] Use ActExecuteFirstRunner (chunk=10, execute first step only)."
echo "[run_act_baseline] Example mock smoke:"
PYTHONPATH=. python - <<'PY'
from kuavo_rl.act_runner import ActExecuteFirstRunner, ConstantChunkPolicy
from kuavo_rl.adapter import make_kuavo_hilserl_env
from kuavo_rl.config import ActRunnerConfig
import numpy as np
env = make_kuavo_hilserl_env(use_stub_robometer=True)
chunk = np.zeros((10, 16), dtype=np.float32)
runner = ActExecuteFirstRunner(ConstantChunkPolicy(chunk), ActRunnerConfig())
out = runner.run_episode(env, max_steps=3)
print({"steps": out["n"], "discarded_per_step": out["steps"][0]["discarded"]})
env.close()
PY
