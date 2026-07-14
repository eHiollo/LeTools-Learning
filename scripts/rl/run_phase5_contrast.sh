#!/usr/bin/env bash
# Phase-5 contrast smoke (MockBackend, deterministic reward).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  conda activate data 2>/dev/null || true
fi

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
STEPS="${1:-20}"
MANIFEST="${MANIFEST:-data/rl_runs/phase5_contrast_latest/manifest.json}"

exec python -u scripts/rl/run_phase5_contrast.py \
  --steps "${STEPS}" \
  --manifest "${MANIFEST}"
