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
data/rl100/<task>/<name>.zarr      # RL-100 training dataset
```

当前强脑手重采配置输出：

```text
data/rl100/grasp_8_4_v2/episodes/*.npz
data/rl100/grasp_8_4_v2/grasp_8_4_v2.zarr
```

旧的 `third_party/RL-100/data/grasp_8_4.zarr` 存在 `action == state`、
夹爪维恒定且超出 `[0,1]` 的问题，禁止用于 BC 训练。

| field | shape | note |
|-------|-------|------|
| `state` / `next_state` | `(T, 16)` | Kuavo joint+claw |
| `action` / `next_action` | `(T, 16)` | absolute joint targets |
| `point_cloud` / `next_*` | `(T, 1024, 3)` | fused multi-cam FPS |
| `reward` / `return` / `done` / `timeout` | `(T, 1)` | sparse success + penalties |
| `meta/episode_ends` | `(E,)` | episode boundaries |

## Labels（upper cams 采集）

- **右摇杆下推** → **success**（`reward_gesture: right_stick_ud`）
- **右摇杆上推** → **failure**
- **Y 单击** → 开始录制（仅 RESET）
- **Y 双击** → rerecord（丢弃当前，回 RESET）
- **Y 长按** → 结束整场采集（仅 RESET；录制中请先打标签结束）
- 右摇杆左右仍留给腰/底盘；上下仅在 RECORD 时打标签
- 若以后用 B：yaml 设 `reward_gesture: button`（短按 success / 长按 failure）

## 相机稳定配置（已验证）

`configs/launch/hil_upper_cams.launch` 默认：

| 相机 | 型号 | 流 | 分辨率 / FPS |
|------|------|----|--------------|
| 头 | Orbbec Gemini | color+depth | 驱动默认 @ **30 Hz** |
| 左/右腕 | RealSense D405 | **depth-only** | **848×480 @ 30 Hz** |

原因：
- RL-100 zarr 只需要深度做点云，关腕部 color 可减半带宽
- 当前 launch 统一使用 **848×480 @ 30 Hz depth-only**
- `initial_reset:=true` + `respawn:=true` 缓解 D405 nodelet bond-break

启动时可能出现 `rs2_set_region_of_interest` / `hwmon 0x75` 报错：**可忽略**（D405 不支持 AE ROI，已被 catch）。

**不要重复 `roslaunch hil_upper_cams.launch`**，否则节点名冲突互相踢掉。

腕部相机序列号等机器特定配置统一放在 `configs/rl/local/env.sh`（gitignored），
模板见 `configs/rl/local/env.sh.example`。首次使用：

```bash
cp configs/rl/local/env.sh.example configs/rl/local/env.sh
# 编辑 env.sh：填 ROS_MASTER_URI / ROS_IP / 两个腕部相机序列号
# 查序列号：rs-enumerate-devices --compact | grep Serial
```

`run_rl100_zarr_collect.sh` 会自动 source 它；终端 A 的 roslaunch 需手动 `source`。

## 终端顺序（真机 upper cams）

### 终端 A — 相机（保持运行，只开一次）

```bash
source /opt/ros/noetic/setup.bash
source ~/kuavo_ros_application/devel/setup.bash
cd ~/wjy/robot-il/LeTools-Learning
source configs/rl/local/env.sh   # ROS_MASTER_URI / ROS_IP / 腕部相机序列号
roslaunch configs/launch/hil_upper_cams.launch
```

确认三路有深度后再开 B：

```bash
# 头、双腕目标均为 ~30Hz
rostopic hz /camera/depth/image_raw/compressedDepth
rostopic hz /left_wrist_camera/depth/image_rect_raw/compressedDepth
rostopic hz /right_wrist_camera/depth/image_rect_raw/compressedDepth
```


查看头相机画面（在自己电脑浏览器打开；SSH 到机器人后启动服务）

```bash
source /opt/ros/noetic/setup.bash
source ~/kuavo_ros_application/devel/setup.bash
source ~/.bashrc
# 若 8080 已被占用说明已在跑，无需重复启动
rosrun web_video_server web_video_server _port:=8080
```

Orbbec gemini_330 实际 topic 为 `cam2`/`cam3`（非 `/camera/color/...`）：

```bash
rostopic hz /camera/cam2/image_raw/compressed   # 应 ~30–43 Hz
```

浏览器（优先用有线 IP `192.168.26.12`，WiFi 为 `10.10.31.37`）：

```
http://192.168.26.12:8080/stream_viewer?topic=/camera/cam2/image_raw/compressed
# 或 cam3：.../camera/cam3/image_raw/compressed
```

### 终端 B — 采集

```bash
conda activate letools-rl
cd ~/wjy/robot-il/LeTools-Learning
# env.sh 会被脚本自动 source（设 ROS_MASTER_URI / ROS_IP）

bash scripts/rl/run_rl100_zarr_collect.sh preflight --check-ros --timeout-s 10
# 目标: ros_pointcloud.ok == true（三路 depth+info+tf）

bash scripts/rl/run_rl100_zarr_collect.sh collect --confirm-live --build-after
```

采集进程启动时还会自动检查下面两路 action 命令流，二者必须持续更新：

```text
/kuavo_arm_traj                 # 14-D 机械臂目标，消息内为 degree
/control_robot_hand_position    # 强脑手左右各 6-D，范围 0..100
```

强脑手 state 从 `/dexhand/state` 读取。12 维顺序为：

```text
l_thumb, l_thumb_aux, l_index, l_middle, l_ring, l_pinky,
r_thumb, r_thumb_aux, r_index, r_middle, r_ring, r_pinky
```

RL-100 的 16-D 契约为
`[L7, left_gripper, R7, right_gripper]`。当前 1-DoF 强脑手映射取
`/dexhand/state.position[0]` 和 `[6]`，并除以 100 归一化到 `[0,1]`；
action 同样取左右 `robotHandPosition` 的第 0 维。

以下情况会直接丢弃当前 episode，不写入 staging：

- 机械臂或强脑手命令缺失/超过 0.2 秒未更新；
- 机械臂所有 action 与同帧 state 无有效差异；
- 左右手均没有至少 0.05 的归一化开合范围；
- action 出现 NaN/Inf，或夹爪 action 超出 `[0,1]`。

### 采集后验收

```bash
bash scripts/rl/run_rl100_zarr_collect.sh inspect \
  --zarr-path data/rl100/grasp_8_4_v2/grasp_8_4_v2.zarr
```

必须满足：

```text
action_quality.ok: true
arm_action_state_max_abs_rad > 0.0001
left/right 至少一个 gripper_action_range >= 0.05
gripper_action_min/max 均在 [0,1]
```

训练前将验收通过的数据复制到 RL-100 数据目录（旧数据保留作问题样本）：

```bash
cp -a data/rl100/grasp_8_4_v2/grasp_8_4_v2.zarr \
  third_party/RL-100/data/grasp_8_4_v2.zarr
```

### 其它

```bash
python scripts/rl/collect_rl100_zarr.py smoke --task box_to_chest_v1
python scripts/rl/collect_rl100_zarr.py --config configs/rl/rl100_zarr_collect_upper_cams.yaml \
  build --overwrite
python scripts/rl/collect_rl100_zarr.py --config configs/rl/rl100_zarr_collect_upper_cams.yaml \
  inspect --zarr-path data/rl100/grasp_8_4_v2/grasp_8_4_v2.zarr
python scripts/rl/verify_joint_map.py --live
```

## Real-robot notes

- Upper cams deploy（RL-100）: `configs/deploy/total/deploy_real_rl100_upper.yaml`
  - **只等 joint_q + gripper**（RGB 不进 zarr，避免腕相机 RGB 掉线卡住）
  - 点云来自深度 hub，不是 gym RGB
- `env_config`: `configs/rl/kuavo_hilserl_real_mvp.yaml`
- Episode 结束靠 B（短按=success，长按≥0.8s=failure）；`live_max_steps` / `live_max_duration_s` 仅安全上限
- **不要**在 RECORD 中双击 Y（那是 rerecord/丢弃）；Y 单击在 RECORD 中会被忽略
- 三相机深度缺一或 workspace 裁空点云会 **硬失败**
- 末端类型必须为 `qiangnao`；观测话题为 `/dexhand/state`，不是 `/leju_claw_state`
- action 命令流不再回退为 measured state；缺失时硬失败，避免再次生成 hold-policy 数据
- `kuavo_rl/ros_msg_compat.py`：只注入 SDK 缺的 footPose6D，**不要**把整包 SDK `kuavo_msgs` 前置到 `PYTHONPATH`（会弄坏 `/sensors_data_raw` MD5）
- 若腕相机进程退出（`Bond broken` / `finished cleanly`），先停干净再只开一次终端 A，再 collect
- ROS master：见 `configs/rl/local/env.sh`（`ROS_MASTER_URI` / `ROS_IP`），脚本与终端 A 均自动/手动 source

## libgomp static-TLS workaround（aarch64 / Tegra）

真机 `import torch` 可能直接挂掉：

```
ImportError: .../torch/lib/../../torch.libs/libgomp-<hash>.so.1.0.0:
cannot allocate memory in static TLS block
```

**根因**：torch 自带的 `libgomp-<hash>.so` 用 static TLS（线程局部存储静态块），
且其 SONAME（如 `libgomp-a49a47f9.so.1.0.0`）**和系统的 `libgomp.so.1` 不同**。
glibc 在进程启动晚期才加载它时，static TLS block 已被先加载的库（ROS noetic +
kuavo workspace 引入的大量 .so）分配完，没有剩余槽位 —— 与内存大小无关，是 TLS
预留空间问题。

`live_collect.py:46` 的 `import torch  # load before cv_bridge` 只能保证 torch
先于 `cv_bridge` 加载，**当 torch 自身就加载失败时它救不了**，必须在 Python
启动前就 preload。

**关键坑**：preload **系统的** `libgomp.so.1` **没用** —— SONAME 不同，torch
仍会加载自己的 `libgomp-<hash>.so`，两者各占一个 TLS 槽位，问题依旧。必须
preload **torch 自己的** `libgomp-<hash>.so`，让它先占 TLS 槽位，torch 后续
加载时复用。

**修复**：`scripts/rl/run_rl100_zarr_collect.sh` 在 Python 启动前 `LD_PRELOAD`
torch 自带的 libgomp。检测完全动态 —— glob 匹配
`$CONDA_PREFIX/lib/python*/site-packages/torch.libs/libgomp*.so*`，**不写死
Python 版本与 libgomp hash**，升级 torch / python 无需改脚本；找不到时兜底系统
libgomp。若 shell 里已有 `LD_PRELOAD` 但不指向 torch 的 libgomp（典型：之前手动
`export` 过系统的 libgomp），脚本会**覆盖**它并打印
`overriding stale LD_PRELOAD=[...]`；想保留自己的设置可设 `RL100_KEEP_LD_PRELOAD=1`。

```bash
# 脚本会自动做，这里仅供手动复现（注意：必须是 torch 自己的那个）
GOMP=$(ls "$CONDA_PREFIX"/lib/python*/site-packages/torch.libs/libgomp*.so* 2>/dev/null | head -1)
LD_PRELOAD="$GOMP" python -c "import torch, cv_bridge; print(torch.__version__)"
# 应输出 torch 版本且 cv_bridge OK
```

注意：

- 若在别的 shell 直接跑 `python scripts/rl/collect_rl100_zarr.py`（绕过启动脚本），
  需自己先 `export LD_PRELOAD` 指向 torch 自带的 libgomp，否则会复现该报错。
- 升级 torch / glibc 后若问题消失，这段检测是幂等的，可保留无副作用。

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
