#!/usr/bin/env bash
# Start ACT infer server in hilserl Docker (GPU). Host ROS client connects to localhost:8765.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
CKPT="${1:-data/rl_runs/checkpoints/005000/pretrained_model}"
PORT="${ACT_INFER_PORT:-8765}"
NAME="${ACT_INFER_NAME:-kuavo-act-infer}"

docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[act-infer] starting $NAME ckpt=$CKPT port=$PORT"
docker run -d --name "$NAME" --gpus all \
  -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
  -p "${PORT}:8765" \
  -v "${ROOT}:/workspace/LeTools-Learning" \
  -v "${HOME}/.cache/torch:/root/.cache/torch" \
  -w /workspace/LeTools-Learning \
  "$IMAGE" \
  bash -lc "
source /opt/conda/etc/profile.d/conda.sh
set +u; conda activate letools; set -u
export PYTHONPATH=/workspace/LeTools-Learning:/workspace/LeTools-Learning/third_party/lerobot/src
exec python -u scripts/rl/act_infer_server.py \
  --host 0.0.0.0 --port 8765 \
  --checkpoint ${CKPT} --device cuda
"

echo "[act-infer] waiting for listen..."
for i in $(seq 1 60); do
  if docker logs "$NAME" 2>&1 | grep -q "listening on"; then
    echo "[act-infer] ready (docker logs $NAME)"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "[act-infer] container exited:"
    docker logs "$NAME" 2>&1 | tail -40
    exit 1
  fi
  sleep 2
done
echo "[act-infer] timeout waiting for ready; last logs:"
docker logs "$NAME" 2>&1 | tail -40
exit 1
