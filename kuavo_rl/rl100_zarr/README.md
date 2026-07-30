# Kuavo → RL-100 zarr collection（真机对齐）

Parallel feature on branch `dev/rl100_record`. Does **not** change HIL recording
defaults (`hil_topics_*.yaml`, `collect_hil_dataset.py`, ACT runners).

真机 upper cams 用专用配置：
`configs/rl/rl100_zarr_collect_upper_cams.yaml`
（深度 topic = `/camera` + `/left|right_wrist_camera`；gym state = `deploy_real_rl100_upper.yaml`）。

Brain `/cam_h|l|r` 深度仍用 `configs/rl/rl100_zarr_collect.yaml`。

## What it produces

```text
data/rl100/<task>/episodes/*.npz   # labeled episode staging
data/rl100/<task>/demo.zarr          # RL-100 training dataset
```

| field | shape | note |
|-------|-------|------|
| `state` / `next_state` | `(T, 16)` | Kuavo joint+claw |
| `action` / `next_action` | `(T, 16)` | absolute joint targets |
| `point_cloud` / `next_*` | `(T, 1024, 3)` | fused multi-cam FPS |
| `reward` / `return` / `done` / `timeout` | `(T, 1)` | sparse success + penalties |
| `meta/episode_ends` | `(E,)` | episode boundaries |

## Labels (same as HIL)

- B click → **success**
- B double → **failure**（默认保留）
- B hold / rerecord → **abort**（丢弃）

## 相机稳定配置（已验证）

`configs/launch/hil_upper_cams.launch` 默认：

| 相机 | 型号 | 流 | 分辨率 / FPS |
|------|------|----|--------------|
| 头 | Orbbec Gemini | color+depth | 驱动默认 @ **30 Hz** |
| 左/右腕 | RealSense D405 | **depth-only** | **640×480 @ 15 Hz** |

原因：
- RL-100 zarr 只需要深度做点云，关腕部 color 可减半带宽
- 左腕常在 **USB 2.1**；右腕 USB 3.2；两台固件不同（5.13 / 5.16）
- 左腕 848×480 depth **最高仅 10/5 Hz**；右腕无 10 Hz profile（会回退 30 Hz）
- 共同可用且稳定的 profile：**640×480 @ 15 Hz depth-only**
- `initial_reset:=true` + `respawn:=true` 缓解 D405 nodelet bond-break

启动时可能出现 `rs2_set_region_of_interest` / `hwmon 0x75` 报错：**可忽略**（D405 不支持 AE ROI，已被 catch）。

**不要重复 `roslaunch hil_upper_cams.launch`**，否则节点名冲突互相踢掉。

本机腕部序列号（也写在 `~/.bashrc`）：

```bash
export LEFT_WRIST_CAMERA_SERIAL_NO=412622270881
export RIGHT_WRIST_CAMERA_SERIAL_NO=260522279592
```

## 终端顺序（真机 upper cams）

### 终端 A — 相机（保持运行，只开一次）

```bash
export ROS_MASTER_URI=http://kuavo_master:11311
export ROS_IP=192.168.26.12
source /opt/ros/noetic/setup.bash
source ~/kuavo_ros_application/devel/setup.bash
cd ~/wjy/robot-il/LeTools-Learning
export LEFT_WRIST_CAMERA_SERIAL_NO=412622270881
export RIGHT_WRIST_CAMERA_SERIAL_NO=260522279592
roslaunch configs/launch/hil_upper_cams.launch
```

确认三路有深度后再开 B：

```bash
# 头 ~30Hz，腕 ~15Hz
rostopic hz /camera/depth/image_raw/compressedDepth
rostopic hz /left_wrist_camera/depth/image_rect_raw/compressedDepth
rostopic hz /right_wrist_camera/depth/image_rect_raw/compressedDepth
```

### 终端 B — 采集

```bash
conda activate letools-rl
cd ~/wjy/robot-il/LeTools-Learning

bash scripts/rl/run_rl100_zarr_collect.sh preflight --check-ros --timeout-s 10
# 目标: ros_pointcloud.ok == true（三路 depth+info+tf）

bash scripts/rl/run_rl100_zarr_collect.sh collect --confirm-live --build-after
```

### 其它

```bash
python scripts/rl/collect_rl100_zarr.py smoke --task box_to_chest_v1
python scripts/rl/collect_rl100_zarr.py build --config configs/rl/rl100_zarr_collect_upper_cams.yaml --overwrite
python scripts/rl/collect_rl100_zarr.py inspect --task box_to_chest_v1
```

## Real-robot notes

- Upper cams deploy（RL-100）: `configs/deploy/total/deploy_real_rl100_upper.yaml`
  - **只等 joint_q + gripper**（RGB 不进 zarr，避免腕相机 RGB 掉线卡住）
  - 点云来自深度 hub，不是 gym RGB
- `env_config`: `configs/rl/kuavo_hilserl_real_mvp.yaml`
- Episode 结束靠 B；`live_max_steps` / `live_max_duration_s` 仅安全上限
- 三相机深度缺一或 workspace 裁空点云会 **硬失败**
- gripper 若 3s 无消息会 idle-seed 零帧，避免永远卡在 buffer
- `kuavo_rl/ros_msg_compat.py`：只注入 SDK 缺的 footPose6D，**不要**把整包 SDK `kuavo_msgs` 前置到 `PYTHONPATH`（会弄坏 `/sensors_data_raw` MD5）
- 若腕相机进程退出（`Bond broken` / `finished cleanly`），先停干净再只开一次终端 A，再 collect
- ROS master：`ROS_MASTER_URI=http://kuavo_master:11311`，本机 `ROS_IP=192.168.26.12`

## Dependencies

- build/smoke: `numpy`, `pyyaml`, `zarr` (`zarr>=2.12,<3`)
- Optional: `fpsample`
- Live: ROS + `cv_bridge` + `tf2_ros` + Kuavo real stack

## Isolation notes

- Package: `kuavo_rl/rl100_zarr/`
- CLI: `scripts/rl/collect_rl100_zarr.py` / `scripts/rl/run_rl100_zarr_collect.sh`
- Configs: `rl100_zarr_collect.yaml` / `rl100_zarr_collect_upper_cams.yaml`
- Launch: `configs/launch/hil_upper_cams.launch`
- Does not modify HIL topic profiles; `data/` is gitignored
