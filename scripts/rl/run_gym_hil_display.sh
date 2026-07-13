#!/usr/bin/env bash
# Phase 1 with display: Keyboard HIL on gym_hil (requires local X11).
#
# Usage:
#   bash scripts/rl/run_gym_hil_display.sh
#   bash scripts/rl/run_gym_hil_display.sh configs/rl/gym_hil_display.json
#
# Keyboard (focus the MuJoCo window):
#   Arrows / Shift / Ctrl  — EE + gripper
#   Space                 — start/stop intervention
#   Enter / Backspace     — success / failure end episode
#   Esc                   — exit
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
CONFIG="${1:-configs/rl/gym_hil_display.json}"
RUN_ID="gym_hil_display_$(date +%Y%m%d_%H%M%S)"
ACTOR_TIMEOUT_S="${ACTOR_TIMEOUT_S:-900}"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "ERROR: DISPLAY is empty. Run this on a graphical session." >&2
  exit 1
fi

# Allow container (often root) to talk to the host X server.
if command -v xhost >/dev/null 2>&1; then
  xhost +local:root >/dev/null 2>&1 || xhost +local:docker >/dev/null 2>&1 || true
fi

mkdir -p "${ROOT}/data/rl_runs"
echo "[phase1-display] image=$IMAGE config=$CONFIG out=${RUN_ID} DISPLAY=${DISPLAY}"

# NOTE: do not use -t with a heredoc stdin ("the input device is not a TTY").
# Keyboard HIL goes through X11/pynput via DISPLAY, not the docker TTY.
docker run --rm -i --gpus all --network host \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -e MUJOCO_GL="${MUJOCO_GL:-glfw}" \
  -e PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-}" \
  -e LEROBOT_GYM_HIL_HEADLESS=0 \
  -e LEROBOT_GYM_HIL_RENDER_MODE=human \
  -e RUN_ID="${RUN_ID}" \
  -e CONFIG="${CONFIG}" \
  -e ACTOR_TIMEOUT_S="${ACTOR_TIMEOUT_S}" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -w /workspace/LeTools-Learning \
  "$IMAGE" \
  bash -s <<'INNER'
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
set +u
conda activate letools
set -u
set -o pipefail

python - <<'PY'
import os, grpc, gym_hil, torch
from lerobot.rl.train_rl import TrainRLServerPipelineConfig
print(
    "precheck OK cuda=", torch.cuda.is_available(),
    "DISPLAY=", os.environ.get("DISPLAY"),
    "HEADLESS=", os.environ.get("LEROBOT_GYM_HIL_HEADLESS"),
    "MUJOCO_GL=", os.environ.get("MUJOCO_GL"),
)
PY

OUT="data/rl_runs/${RUN_ID}"
OUT_ACTOR="data/rl_runs/${RUN_ID}_actor"
LEARNER_LOG="data/rl_runs/${RUN_ID}_learner.log"
ACTOR_LOG="data/rl_runs/${RUN_ID}_actor.log"

python -m lerobot.rl.learner --config_path "${CONFIG}" --output_dir "${OUT}" >"${LEARNER_LOG}" 2>&1 &
LPID=$!
echo "learner_pid=${LPID} out=${OUT}"
echo "Keyboard tip: focus the MuJoCo window, press Space to intervene."

ready=0
for _ in $(seq 1 60); do
  if python -c 'import socket; s=socket.socket(); s.settimeout(0.5); s.connect(("127.0.0.1", 50051)); s.close()' 2>/dev/null; then
    echo learner_grpc_ready
    ready=1
    break
  fi
  if ! kill -0 "${LPID}" 2>/dev/null; then
    echo "learner exited early; log:"
    tail -n 120 "${LEARNER_LOG}" || true
    exit 1
  fi
  sleep 1
done
if [[ "${ready}" != "1" ]]; then
  echo "learner gRPC not ready; log:"
  tail -n 120 "${LEARNER_LOG}" || true
  kill "${LPID}" 2>/dev/null || true
  exit 1
fi

set +e
# Actor logs go to file AND console so keyboard tips / errors are visible.
timeout "${ACTOR_TIMEOUT_S}" \
  python -m lerobot.rl.actor --config_path "${CONFIG}" --output_dir "${OUT_ACTOR}" \
  2>&1 | tee "${ACTOR_LOG}"
ACODE=${PIPESTATUS[0]}
set -e

kill "${LPID}" 2>/dev/null || true
wait "${LPID}" 2>/dev/null || true
echo "actor_exit=${ACODE}"

if [[ -d "${OUT}" ]]; then
  cp -f "${LEARNER_LOG}" "${OUT}/learner.log" 2>/dev/null || true
  cp -f "${ACTOR_LOG}" "${OUT}/actor.log" 2>/dev/null || true
  echo "${OUT_ACTOR}" >"${OUT}/actor_output_dir.txt"
else
  echo "learner output dir missing"
  tail -n 120 "${LEARNER_LOG}" || true
  exit 1
fi

if grep -Eq "Policy loop finished|Episode reward|make_env online" "${OUT}/actor.log" \
  && grep -Eq "gRPC server started|Number of optimization step" "${OUT}/learner.log"; then
  if grep -Eq "Unhandled exception in act_with_policy" "${OUT}/actor.log"; then
    echo "PHASE1_DISPLAY_FAIL actor crashed in policy loop"
    exit 1
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
  "mode": "display_keyboard",
  "run_dir": str(out),
  "actor_exit": int("${ACODE}"),
  "max_optimization_step": max(opt_steps) if opt_steps else 0,
  "episode_rewards": [float(x) for x in rewards],
  "headless": False,
  "mujoco_gl": "glfw",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\\n")
print("manifest", manifest)
PY
  echo "PHASE1_DISPLAY_OK out=${OUT}"
  exit 0
fi

echo "PHASE1_DISPLAY_FAIL actor_exit=${ACODE}"
tail -n 80 "${OUT}/actor.log" || true
exit 1
INNER

ln -sfn "${RUN_ID}" "${ROOT}/data/rl_runs/gym_hil_display_latest" 2>/dev/null || true
