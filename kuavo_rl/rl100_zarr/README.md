# Kuavo → RL-100 zarr collection

Parallel feature on branch `dev/rl100_record`. Does **not** change HIL recording
defaults (`hil_topics_*.yaml`, `collect_hil_dataset.py`, ACT runners).

## What it produces

```text
data/rl100/<task>/episodes/*.npz   # labeled episode staging
data/rl100/<task>/demo.zarr          # RL-100 training dataset
```

Zarr fields (aligned with RL-100 `data_prepare` / FoldingDataset):

| field | shape | note |
|-------|-------|------|
| `state` / `next_state` | `(T, 16)` | Kuavo joint+claw; training maps to `agent_pos` |
| `action` / `next_action` | `(T, 16)` | absolute joint targets |
| `point_cloud` / `next_*` | `(T, 1024, 3)` | fused multi-cam FPS |
| `reward` / `return` / `done` / `timeout` | `(T, 1)` | sparse success + penalties |
| `meta/episode_ends` | `(E,)` | episode boundaries |

## Labels (same as HIL)

- B click → **success** (terminal reward ≈ 1 − length penalty)
- B double → **failure** (kept by default, terminal reward 0)
- B hold / rerecord → **abort** (discarded, not written)

## Commands

```bash
# Config / optional ROS depth check
python scripts/rl/collect_rl100_zarr.py preflight --config configs/rl/rl100_zarr_collect.yaml
python scripts/rl/collect_rl100_zarr.py preflight --check-ros

# No-robot smoke test
python scripts/rl/collect_rl100_zarr.py smoke --task kuavo_demo

# Merge staged NPZs → zarr
python scripts/rl/collect_rl100_zarr.py build --config configs/rl/rl100_zarr_collect.yaml --overwrite

# Live VR collect (ROS + depth/tf + confirm)
python scripts/rl/collect_rl100_zarr.py collect \
  --config configs/rl/rl100_zarr_collect.yaml \
  --confirm-live --build-after

# Inspect
python scripts/rl/collect_rl100_zarr.py inspect --task kuavo_demo
```

## Dependencies

- Required for build/smoke: `numpy`, `pyyaml`, `zarr` (prefer `zarr>=2.12,<3`)
- Optional better FPS: `fpsample`
- Live collect: ROS + `cv_bridge` + `tf2_ros` + existing Kuavo sim/deploy stack

## Isolation notes

- New package: `kuavo_rl/rl100_zarr/`
- New CLI: `scripts/rl/collect_rl100_zarr.py`
- New config: `configs/rl/rl100_zarr_collect.yaml`
- Does not modify `contracts.py` semantics or HIL topic profiles
- `data/` is gitignored; only code/config are committed
