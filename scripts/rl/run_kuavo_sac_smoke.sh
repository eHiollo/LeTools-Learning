#!/usr/bin/env bash
# Stage-B smoke: KuavoHILSerlEnv (MockBackend) + gaussian_actor + SAC (learner then actor).
# Does NOT start ACT runner. Replay actions must remain single-step 16-D.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
CONFIG="${1:-configs/rl/kuavo_sac_smoke.json}"
RUN_ID="kuavo_sac_smoke_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${ROOT}/data/rl_runs"

echo "[stage-b] image=$IMAGE config=$CONFIG out=${RUN_ID}"

docker run --rm -i --gpus all --network host \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -e KUAVO_HILSERL_BACKEND="${KUAVO_HILSERL_BACKEND:-mock}" \
  -e RUN_ID="${RUN_ID}" \
  -e CONFIG="${CONFIG}" \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -w /workspace/LeTools-Learning \
  "$IMAGE" \
  bash -s <<'INNER'
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
set +u
conda activate letools
set -u
set -o pipefail
export PYTHONPATH="/workspace/LeTools-Learning:${PYTHONPATH:-}"

python - <<'PY'
import grpc, torch
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
for _ in $(seq 1 60); do
  if python -c 'import socket; s=socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1", 50051)); s.close()' 2>/dev/null; then
    echo learner_grpc_ready
    ready=1
    break
  fi
  if ! kill -0 "${LPID}" 2>/dev/null; then
    echo "learner exited early; log:"
    tail -n 160 "${LEARNER_LOG}" || true
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "learner gRPC not ready after timeout; log:"
  tail -n 160 "${LEARNER_LOG}" || true
  kill "${LPID}" 2>/dev/null || true
  exit 1
fi

set +e
timeout 420 python -m kuavo_rl.hilserl_cli actor --config_path "${CONFIG}" --output_dir "${OUT_ACTOR}" >"${ACTOR_LOG}" 2>&1
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
  tail -n 100 "${OUT}/learner.log" || true
  echo "----- actor.log (tail) -----"
  tail -n 100 "${OUT}/actor.log" || true
else
  echo "learner output dir missing"
  tail -n 160 "${LEARNER_LOG}" || true
  tail -n 160 "${ACTOR_LOG}" || true
  exit 1
fi

if grep -Eq "Policy loop finished|Episode reward" "${OUT}/actor.log" \
  && grep -Eq "Number of optimization step|gRPC server started" "${OUT}/learner.log"; then
  if grep -Eq "Unhandled exception in act_with_policy" "${OUT}/actor.log"; then
    echo "STAGEB_SMOKE_FAIL actor crashed; see ${OUT}/actor.log"
    exit 1
  fi
  python - <<PY
import json, re, pathlib
out = pathlib.Path("${OUT}")
actor = (out / "actor.log").read_text(errors="replace")
learner = (out / "learner.log").read_text(errors="replace")
opt_steps = [int(x) for x in re.findall(r"Number of optimization step: (\\d+)", learner)]
rewards = re.findall(r"Episode reward: ([\\-0-9.]+)", actor)
# Sanity: no ACT chunk shapes in logs
bad = bool(re.search(r"chunk_size|action chunk|\\(10,\\s*16\\)", actor, re.I))
manifest = {
  "status": "ok",
  "mode": "kuavo_sac_smoke",
  "backend": "mock",
  "policy": "gaussian_actor",
  "algorithm": "sac",
  "run_dir": str(out),
  "actor_exit": int("${ACODE}"),
  "max_optimization_step": max(opt_steps) if opt_steps else 0,
  "episode_rewards": [float(x) for x in rewards],
  "act_chunk_not_used": (not bad),
  "action_dim": 16,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\\n")
# Convenience pointer
latest = pathlib.Path("data/rl_runs/kuavo_sac_smoke_latest")
if latest.exists() or latest.is_symlink():
    latest.unlink()
latest.symlink_to(out.name)
print("manifest", manifest)
PY
  echo "STAGEB_SMOKE_OK out=${OUT}"
  exit 0
fi

if [[ "${ACODE}" -eq 124 ]] && grep -Eq "Episode reward|Policy loop finished" "${OUT}/actor.log"; then
  echo "STAGEB_SMOKE_OK_TIMEOUT out=${OUT}"
  exit 0
fi

echo "STAGEB_SMOKE_FAIL actor_exit=${ACODE}"
exit 1
INNER
