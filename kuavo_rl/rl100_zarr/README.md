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

## Labels（与 main HIL 对齐）

- **B 短按** → **success**
- **B 长按**（≥ `chord_long_press_s`，默认 0.8s）→ **failure**（默认保留）
- **Y 单击** → 开始录制（仅 RESET）
- **Y 双击** → rerecord（丢弃当前，回 RESET）
- **Y 长按** → 结束整场采集（仅 RESET；录制中请先用 B 结束）
- 右摇杆留给腰/底盘，不占用 episode 控制（`episode_control: quest_y_button`）

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

## libgomp static-TLS workaround（aarch64 / Tegra）

真机 `import torch` 可能直接挂掉：

```
ImportError: .../torch/lib/../../torch.libs/libgomp-a49a47f9.so.1.0.0:
cannot allocate memory in static TLS block
```

**根因**：torch 自带的 `libgomp-a49a47f9.so.1.0.0` 用 static TLS（线程局部存储
静态块），且其 SONAME 是 `libgomp-a49a47f9.so.1.0.0`，**和系统的
`libgomp.so.1` 不同**。glibc 在进程启动晚期才加载它时，static TLS block 已被
先加载的库（ROS noetic + kuavo workspace 引入的大量 .so）分配完，没有剩余槽位
—— 与内存大小无关，是 TLS 预留空间问题。

`live_collect.py:46` 的 `import torch  # load before cv_bridge` 只能保证 torch
先于 `cv_bridge` 加载，**当 torch 自身就加载失败时它救不了**，必须在 Python
启动前就 preload。

**关键坑**：preload **系统的** `libgomp.so.1` **没用** —— SONAME 不同，torch
仍会加载自己的 `libgomp-a49a47f9.so`，两者各占一个 TLS 槽位，问题依旧。必须
preload **torch 自己的** `libgomp-a49a47f9.so.1.0.0`，让它先占 TLS 槽位，torch
后续加载时复用。

**修复**：`scripts/rl/run_rl100_zarr_collect.sh` 已内置自动检测，按优先级查找
torch 自带的 libgomp 并注入 `LD_PRELOAD`。若 shell 里已有 `LD_PRELOAD` 但
不指向 torch 的 libgomp（典型：之前手动 `export` 过系统的 libgomp），脚本会
**覆盖**它并打印 `overriding stale LD_PRELOAD=[...]`；想保留自己的设置可设
`RL100_KEEP_LD_PRELOAD=1`。

```bash
# 脚本会自动做，这里仅供手动复现（注意：必须是 torch 自己的那个）
GOMP=$CONDA_PREFIX/lib/python3.10/site-packages/torch.libs/libgomp-a49a47f9.so.1.0.0
LD_PRELOAD="$GOMP" python -c "import torch, cv_bridge; print(torch.__version__)"
# 应输出 2.9.1+cpu 且 cv_bridge OK
```

注意：

- 升级 torch 后 `libgomp-a49a47f9.so.1.0.0` 的 hash 段（`a49a47f9`）可能变化，
  脚本里也兜底匹配 `torch.libs/libgomp.so.1` 与系统 libgomp，但**首选**带 hash
  的那个；若哪天 torch 不再自带 libgomp，兜底链仍能工作。
- 若在别的 shell 直接跑 `python scripts/rl/collect_rl100_zarr.py`（绕过启动
  脚本），需自己先 `export LD_PRELOAD` 指向 torch 自带的 libgomp，否则会复现
  该报错。
- 升级 torch / glibc 后若问题消失，这段检测是幂等的，可保留无副作用。
- 候选路径（按优先级）：
  - `$CONDA_PREFIX/lib/python3.10/site-packages/torch.libs/libgomp-a49a47f9.so.1.0.0`（推荐，精确匹配 torch SONAME）
  - `$CONDA_PREFIX/lib/python3.10/site-packages/torch.libs/libgomp.so.1`
  - `/usr/lib/aarch64-linux-gnu/libgomp.so.1`（系统兜底）
  - `/usr/lib/x86_64-linux-gnu/libgomp.so.1`（x86_64 兜底）

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
