# Kuavo → RL-100 zarr collection（真机对齐）

Parallel feature on branch `dev/rl100_record`. Does **not** change HIL recording
defaults (`hil_topics_*.yaml`, `collect_hil_dataset.py`, ACT runners).

默认配置对齐真机采集：`deploy_total.yaml` + `kuavo_hilserl_real_mvp.yaml`，
深度为 `compressedDepth`（与 `kuavo_deploy` / 模仿学习一致）。

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
# Config / ROS depth+tf check（上机前必跑）
python scripts/rl/collect_rl100_zarr.py preflight --config configs/rl/rl100_zarr_collect.yaml
python scripts/rl/collect_rl100_zarr.py preflight --check-ros --timeout-s 10

# No-robot smoke test
python scripts/rl/collect_rl100_zarr.py smoke --task box_to_chest_v1

# Merge staged NPZs → zarr
python scripts/rl/collect_rl100_zarr.py build --config configs/rl/rl100_zarr_collect.yaml --overwrite

# Live VR collect on real robot (Kuavo-Real + Quest)
python scripts/rl/collect_rl100_zarr.py collect \
  --config configs/rl/rl100_zarr_collect.yaml \
  --confirm-live --build-after

# Inspect
python scripts/rl/collect_rl100_zarr.py inspect --task box_to_chest_v1
```

## Real-robot notes

- `deploy_config`: `configs/deploy/total/deploy_total.yaml` → `Kuavo-Real`
- `env_config`: `configs/rl/kuavo_hilserl_real_mvp.yaml`（teleop `reward_button: right_second_button_pressed`）
- Episode 结束靠 B；`live_max_steps` / `live_max_duration_s` 仅作安全上限
- 三相机深度缺一或 workspace 裁空点云会 **硬失败**（避免全零点云进训练）
- 若机上只有 raw `Image` 深度，把 yaml 里 `depth_msg_type: image` 并改 topic

## Dependencies

- Required for build/smoke: `numpy`, `pyyaml`, `zarr` (prefer `zarr>=2.12,<3`)
- Optional better FPS: `fpsample`
- Live collect: ROS + `cv_bridge` + `tf2_ros` + Kuavo real deploy stack

## Isolation notes

- New package: `kuavo_rl/rl100_zarr/`
- New CLI: `scripts/rl/collect_rl100_zarr.py`
- New config: `configs/rl/rl100_zarr_collect.yaml`
- Does not modify `contracts.py` semantics or HIL topic profiles
- `data/` is gitignored; only code/config are committed
