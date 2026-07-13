#!/usr/bin/env bash
# Stage-B Kuavo-Sim SAC smoke: host ROS bridge + Docker learner/actor (proxy backend).
# Prerequisites:
#   1) Kuavo-Sim v62 running
#   2) bash scripts/rl/run_sim_rgb_cameras.sh
#   3) bash scripts/rl/run_kuavo_ros_bridge.sh   # this script can start it if AUTO_BRIDGE=1
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
CONFIG="${1:-configs/rl/kuavo_sac_smoke.json}"
RUN_ID="kuavo_sac_sim_$(date +%Y%m%d_%H%M%S)"
BRIDGE_PORT="${KUAVO_ROS_BRIDGE_PORT:-8877}"
mkdir -p "${ROOT}/data/rl_runs"

if [[ "${AUTO_BRIDGE:-1}" == "1" ]]; then
  if ! ss -ltn 2>/dev/null | grep -q ":${BRIDGE_PORT} "; then
    echo "[stage-b-sim] starting ROS bridge on :${BRIDGE_PORT}"
    bash scripts/rl/run_kuavo_ros_bridge.sh >"data/rl_runs/${RUN_ID}_bridge.log" 2>&1 &
    BPID=$!
    echo "${BPID}" >"data/rl_runs/${RUN_ID}_bridge.pid"
    for _ in $(seq 1 60); do
      if ss -ltn 2>/dev/null | grep -q ":${BRIDGE_PORT} "; then
        echo "[stage-b-sim] bridge ready"
        break
      fi
      if ! kill -0 "${BPID}" 2>/dev/null; then
        echo "[stage-b-sim] bridge died; log:"
        tail -n 80 "data/rl_runs/${RUN_ID}_bridge.log" || true
        exit 1
      fi
      sleep 1
    done
  else
    echo "[stage-b-sim] reuse existing bridge on :${BRIDGE_PORT}"
  fi
fi

echo "[stage-b-sim] image=$IMAGE config=$CONFIG out=${RUN_ID} backend=proxy"

# Free learner port if leftover
fuser -k 50051/tcp 2>/dev/null || true

docker run --rm -i --gpus all --network host \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -e KUAVO_HILSERL_BACKEND=proxy \
  -e KUAVO_ROS_BRIDGE_HOST=127.0.0.1 \
  -e KUAVO_ROS_BRIDGE_PORT="${BRIDGE_PORT}" \
  -e KUAVO_HILSERL_SHADOW="${KUAVO_HILSERL_SHADOW:-0}" \
  -e RUN_ID="${RUN_ID}" \
  -e CONFIG="${CONFIG}" \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/LeTools-Learning \
  "$IMAGE" \
  bash -s <<'INNER'
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
set +u; conda activate letools; set -u
set -o pipefail
export PYTHONPATH="/workspace/LeTools-Learning:${PYTHONPATH:-}"

OUT="data/rl_runs/${RUN_ID}"
OUT_ACTOR="data/rl_runs/${RUN_ID}_actor"
LEARNER_LOG="data/rl_runs/${RUN_ID}_learner.log"
ACTOR_LOG="data/rl_runs/${RUN_ID}_actor.log"

python -m kuavo_rl.hilserl_cli learner --config_path "${CONFIG}" --output_dir "${OUT}" >"${LEARNER_LOG}" 2>&1 &
LPID=$!
echo "learner_pid=${LPID}"

ready=0
for _ in $(seq 1 60); do
  if python -c 'import socket; s=socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1", 50051)); s.close()' 2>/dev/null; then
    echo learner_grpc_ready
    ready=1
    break
  fi
  if ! kill -0 "${LPID}" 2>/dev/null; then
    echo "learner exited early:"; tail -n 160 "${LEARNER_LOG}" || true; exit 1
  fi
  sleep 1
done
[[ "${ready}" == "1" ]] || { echo "learner not ready"; tail -n 160 "${LEARNER_LOG}"; exit 1; }

set +e
timeout 420 python -m kuavo_rl.hilserl_cli actor --config_path "${CONFIG}" --output_dir "${OUT_ACTOR}" >"${ACTOR_LOG}" 2>&1
ACODE=$?
set -e
kill "${LPID}" 2>/dev/null || true
wait "${LPID}" 2>/dev/null || true

if [[ -d "${OUT}" ]]; then
  mv -f "${LEARNER_LOG}" "${OUT}/learner.log" 2>/dev/null || true
  mv -f "${ACTOR_LOG}" "${OUT}/actor.log" 2>/dev/null || true
  echo "----- actor.log (tail) -----"; tail -n 80 "${OUT}/actor.log" || true
else
  tail -n 120 "${LEARNER_LOG}" || true; tail -n 120 "${ACTOR_LOG}" || true; exit 1
fi

if grep -Eq "Policy loop finished|Episode reward" "${OUT}/actor.log" \
  && grep -Eq "Number of optimization step|gRPC server started" "${OUT}/learner.log"; then
  if grep -Eq "Unhandled exception in act_with_policy|ros bridge error" "${OUT}/actor.log"; then
    echo "STAGEB_SIM_FAIL"; exit 1
  fi
  python - <<PY
import json, re, pathlib
out = pathlib.Path("${OUT}")
actor = (out / "actor.log").read_text(errors="replace")
learner = (out / "learner.log").read_text(errors="replace")
opt_steps = [int(x) for x in re.findall(r"Number of optimization step: (\\d+)", learner)]
rewards = re.findall(r"Episode reward: ([\\-0-9.]+)", actor)
manifest = {
  "status": "ok",
  "mode": "kuavo_sac_sim",
  "backend": "proxy",
  "policy": "gaussian_actor",
  "algorithm": "sac",
  "run_dir": str(out),
  "actor_exit": int("${ACODE}"),
  "max_optimization_step": max(opt_steps) if opt_steps else 0,
  "episode_rewards": [float(x) for x in rewards],
  "act_chunk_not_used": True,
  "action_dim": 16,
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\\n")
latest = pathlib.Path("data/rl_runs/kuavo_sac_sim_latest")
if latest.exists() or latest.is_symlink():
    latest.unlink()
latest.symlink_to(out.name)
print("manifest", manifest)
PY
  echo "STAGEB_SIM_OK out=${OUT}"
  exit 0
fi
echo "STAGEB_SIM_FAIL actor_exit=${ACODE}"
exit 1
INNER
