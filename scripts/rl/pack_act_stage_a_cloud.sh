#!/usr/bin/env bash
# Pack Stage-A ACT training bundle for a remote / cloud GPU machine.
#
# Creates: data/rl_runs/act_stage_a_cloud_bundle_YYYYMMDD_HHMMSS.tar
# Contains code configs + lerobot_merged dataset (no third_party full git history).
#
# On cloud:
#   tar xf act_stage_a_cloud_bundle_*.tar
#   cd LeTools-Learning   # or the extracted root
#   bash scripts/rl/train_act_stage_a.sh configs/rl/act_stage_a_train.json
#
# Or without Docker (conda/venv with lerobot 0.6.x + CUDA):
#   USE_DOCKER=0 bash scripts/rl/train_act_stage_a.sh configs/rl/act_stage_a_train.json
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/data/rl_runs"
mkdir -p "$OUT_DIR"
BUNDLE="${OUT_DIR}/act_stage_a_cloud_bundle_${STAMP}.tar"

echo "[pack] root=$ROOT"
echo "[pack] writing $BUNDLE"

# Staging list (relative to ROOT)
LIST="$(mktemp)"
cat >"$LIST" <<'EOF'
configs/rl/act_stage_a_train.json
configs/rl/act_stage_a_smoke.json
configs/rl/act_kuavo_bc.yaml
scripts/rl/train_act_stage_a.sh
scripts/rl/eval_act_execute_first.py
scripts/rl/run_act_baseline.sh
scripts/rl/preflight.py
third_party/lerobot
data/lerobot/lerobot_merged
EOF

# Exclude bulky / irrelevant paths from third_party/lerobot and dataset QC videos if needed.
# Keep QC out of critical path; videos are inside lerobot_merged and ARE needed for training.
cd "$ROOT"
tar \
  --exclude='third_party/lerobot/.git' \
  --exclude='third_party/lerobot/**/__pycache__' \
  --exclude='**/__pycache__' \
  --exclude='data/lerobot/lerobot_merged/per_bag_qc' \
  -cf "$BUNDLE" -T "$LIST"
rm -f "$LIST"

# Convenience: gzip if under reasonable size after tar (optional, user can gzip)
SIZE="$(du -h "$BUNDLE" | awk '{print $1}')"
echo "[pack] done size=$SIZE path=$BUNDLE"
echo
echo "Upload example:"
echo "  scp $BUNDLE user@cloud:/data/"
echo "  # on cloud:"
echo "  mkdir -p ~/robot-il && tar xf /data/$(basename "$BUNDLE") -C ~/robot-il"
echo "  # ensure layout is ~/robot-il/LeTools-Learning/... or adjust paths"
echo "  # If tar extracted flat keys, recreate:"
echo "  mkdir -p ~/LeTools-Learning && cd ~/LeTools-Learning && tar xf /data/$(basename "$BUNDLE")"
echo "  bash scripts/rl/train_act_stage_a.sh configs/rl/act_stage_a_train.json"
