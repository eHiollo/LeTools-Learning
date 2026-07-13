#!/usr/bin/env bash
# Stage-A ACT training on lerobot_merged (chunk_size=10, n_action_steps=1).
#
# Local Docker (default):
#   bash scripts/rl/train_act_stage_a.sh
#   bash scripts/rl/train_act_stage_a.sh configs/rl/act_stage_a_smoke.json
#
# Cloud / bare-metal (no Docker), after installing lerobot 0.6.x with CUDA:
#   USE_DOCKER=0 bash scripts/rl/train_act_stage_a.sh configs/rl/act_stage_a_train.json
#
# Env overrides:
#   LETOOLS_RL_IMAGE   docker image (default letools-train:hilserl)
#   USE_DOCKER=0|1
#   NVIDIA_VISIBLE_DEVICES
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
IMAGE="${LETOOLS_RL_IMAGE:-letools-train:hilserl}"
CONFIG="${1:-configs/rl/act_stage_a_train.json}"
USE_DOCKER="${USE_DOCKER:-1}"
STAMP="$(date +%Y%m%d_%H%M%S)"
# Unique output dir so validate() does not refuse an existing path.
OUT_DIR="data/rl_runs/act_stage_a_${STAMP}"

mkdir -p "${ROOT}/data/rl_runs"
echo "[act-stage-a] use_docker=$USE_DOCKER config=$CONFIG out=$OUT_DIR"

run_train() {
  python - <<'PY'
import torch
from lerobot.policies.act.configuration_act import ACTConfig
print("precheck OK cuda=", torch.cuda.is_available(), "ACTConfig OK")
PY

  lerobot-train \
    --config_path "${CONFIG}" \
    --output_dir "${OUT_DIR}" \
    --job_name "act_stage_a_${STAMP}" \
    --wandb.enable=false \
    --policy.push_to_hub=false

  python - <<PY
import json, pathlib
out = pathlib.Path("${OUT_DIR}")
ckpts = sorted((out / "checkpoints").glob("*")) if (out / "checkpoints").exists() else []
manifest = {
  "status": "ok",
  "stage": "A",
  "policy": "act",
  "chunk_size": 10,
  "n_action_steps": 1,
  "dataset": "data/lerobot/lerobot_merged",
  "output_dir": str(out),
  "checkpoints": [str(p) for p in ckpts],
  "note": "Use ActExecuteFirstRunner for closed-loop eval; do not write ACT chunks into SAC replay.",
}
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("PHASE4_ACT_TRAIN_OK", manifest)
PY
}

if [[ "${USE_DOCKER}" == "1" ]]; then
  echo "[act-stage-a] image=$IMAGE"
  docker run --rm --gpus all --shm-size=16g \
    -e NVIDIA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES:-0}" \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
    -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}" \
    -v "${ROOT}:/workspace/LeTools-Learning" \
    -w /workspace/LeTools-Learning \
    "$IMAGE" \
    bash -lc "
set -eo pipefail
source /opt/conda/etc/profile.d/conda.sh
set +u
conda activate letools
set -u
set -o pipefail
CONFIG='${CONFIG}' OUT_DIR='${OUT_DIR}' STAMP='${STAMP}'
$(declare -f run_train)
run_train
"
else
  echo "[act-stage-a] native python (USE_DOCKER=0)"
  if ! command -v lerobot-train >/dev/null 2>&1; then
    echo "ERROR: lerobot-train not found. Install first, e.g.:" >&2
    echo "  pip install -e \"third_party/lerobot[dataset,training]\"" >&2
    exit 1
  fi
  run_train
fi

ln -sfn "$(basename "$OUT_DIR")" "${ROOT}/data/rl_runs/act_stage_a_latest" 2>/dev/null || true
echo "[act-stage-a] latest -> data/rl_runs/act_stage_a_latest"
echo "[act-stage-a] After cloud training, sync back:"
echo "  rsync -avz cloud:PATH/${OUT_DIR}/ data/rl_runs/\$(basename ${OUT_DIR})/"
echo "  # then on this machine:"
echo "  PYTHONPATH=. python scripts/rl/eval_act_execute_first.py --checkpoint data/rl_runs/act_stage_a_latest/checkpoints/last/pretrained_model"
