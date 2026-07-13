#!/usr/bin/env bash
# Phase 1 smoke: learner then actor on gym_hil (no human input).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
CONFIG="${1:-configs/rl/gym_hil_smoke.json}"
RUN_ID="gym_hil_smoke_$(date +%Y%m%d_%H%M%S)"
# Do NOT mkdir the run output dir here: TrainPipelineConfig.validate() refuses an
# existing output_dir when resume=false. Learner/actor create their own dirs.
mkdir -p "${ROOT}/data/rl_runs"

echo "[phase1] image=$IMAGE config=$CONFIG out=${RUN_ID}"

# Inner script runs inside the container; host expands only RUN_ID/CONFIG via env.
docker run --rm -i --gpus all --network host \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -e MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
  -e PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}" \
  -e LEROBOT_GYM_HIL_RENDER_MODE="${LEROBOT_GYM_HIL_RENDER_MODE:-rgb_array}" \
  -e LEROBOT_GYM_HIL_HEADLESS="${LEROBOT_GYM_HIL_HEADLESS:-1}" \
  -e RUN_ID="${RUN_ID}" \
  -e CONFIG="${CONFIG}" \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -w /workspace/LeTools-Learning \
  "$IMAGE" \
  bash -s <<'INNER'
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
# conda activate may reference unset NVCC_* vars under set -u
set +u
conda activate letools
set -u
set -o pipefail
export PYTHONPATH="/workspace/LeTools-Learning:${PYTHONPATH:-}"

python - <<'PY'
import grpc, gym_hil, torch
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
print("precheck OK cuda=", torch.cuda.is_available())
PY

OUT="data/rl_runs/${RUN_ID}"
OUT_ACTOR="data/rl_runs/${RUN_ID}_actor"
LEARNER_LOG="data/rl_runs/${RUN_ID}_learner.log"
ACTOR_LOG="data/rl_runs/${RUN_ID}_actor.log"

python -m kuavo_rl.hilserl_cli learner --config_path "${CONFIG}" --output_dir "${OUT}" >"${LEARNER_LOG}" 2>&1 &
LPID=$!
echo "learner_pid=${LPID} out=${OUT}"

ready=0
for _ in $(seq 1 45); do
  if python -c 'import socket; s=socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1", 50051)); s.close()' 2>/dev/null; then
    echo learner_grpc_ready
    ready=1
    break
  fi
  # Fail fast if learner already exited
  if ! kill -0 "${LPID}" 2>/dev/null; then
    echo "learner exited early; log:"
    tail -n 120 "${LEARNER_LOG}" || true
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "learner gRPC not ready after timeout; log:"
  tail -n 120 "${LEARNER_LOG}" || true
  kill "${LPID}" 2>/dev/null || true
  exit 1
fi

set +e
timeout 300 python -m kuavo_rl.hilserl_cli actor --config_path "${CONFIG}" --output_dir "${OUT_ACTOR}" >"${ACTOR_LOG}" 2>&1
ACODE=$?
set -e

kill "${LPID}" 2>/dev/null || true
wait "${LPID}" 2>/dev/null || true
echo "actor_exit=${ACODE}"

if [[ -d "${OUT}" ]]; then
  mv -f "${LEARNER_LOG}" "${OUT}/learner.log" 2>/dev/null || true
  mv -f "${ACTOR_LOG}" "${OUT}/actor.log" 2>/dev/null || true
  echo "${OUT_ACTOR}" >"${OUT}/actor_output_dir.txt"
  echo "----- learner.log (tail) -----"
  tail -n 80 "${OUT}/learner.log" || true
  echo "----- actor.log (tail) -----"
  tail -n 80 "${OUT}/actor.log" || true
  test -s "${OUT}/learner.log"
  test -s "${OUT}/actor.log"
else
  echo "learner output dir missing; dumping sidecar logs"
  tail -n 120 "${LEARNER_LOG}" || true
  tail -n 120 "${ACTOR_LOG}" || true
  exit 1
fi

# Success: policy rolled out and learner optimized. Ignore shutdown-time gRPC/queue races.
if grep -Eq "Policy loop finished|Episode reward" "${OUT}/actor.log" \
  && grep -Eq "Number of optimization step|gRPC server started" "${OUT}/learner.log"; then
  if grep -Eq "Unhandled exception in act_with_policy" "${OUT}/actor.log"; then
    echo "PHASE1_SMOKE_FAIL actor crashed in policy loop; see ${OUT}/actor.log"
    exit 1
  fi
  echo "PHASE1_SMOKE_OK out=${OUT} actor_out=${OUT_ACTOR} actor_exit=${ACODE}"
  # Lightweight manifest for progress tracking
  python - <<PY
import json, re, pathlib
out = pathlib.Path("${OUT}")
actor = (out / "actor.log").read_text(errors="replace")
learner = (out / "learner.log").read_text(errors="replace")
opt_steps = [int(x) for x in re.findall(r"Number of optimization step: (\\d+)", learner)]
rewards = re.findall(r"Episode reward: ([\\-0-9.]+)", actor)
manifest = {
  "status": "ok",
  "run_dir": str(out),
  "actor_exit": int("${ACODE}"),
  "max_optimization_step": max(opt_steps) if opt_steps else 0,
  "episode_rewards": [float(x) for x in rewards],
  "headless": True,
  "mujoco_gl": "osmesa",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")
print("manifest", manifest)
PY
  exit 0
fi

if [[ "${ACODE}" -eq 124 ]] && grep -Eq "Episode reward|Policy loop finished" "${OUT}/actor.log"; then
  echo "PHASE1_SMOKE_OK_TIMEOUT out=${OUT} actor_exit=124"
  exit 0
fi

echo "PHASE1_SMOKE_FAIL actor_exit=${ACODE}"
exit 1
INNER
