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

旧的 `third_party/RL-100/data/grasp_8_4.zarr` 是旧的 16/16 数据，禁止与
`rl100_topic_native_v1` 数据混合训练。

| field | shape | note |
|-------|-------|------|
| `state` / `next_state` | `(T, 32)` | raw `joint_q20` (rad) + dexhand feedback12 (0..100) |
| `action` / `next_action` | `(T, 26)` | arm topic degree14 + Qiangnao command raw12 (0..100) |
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

bash scripts/rl/run_rl100_zarr_collect.sh preflight --check-ros --timeout-s 10 --profile-s 60
# 目标: ros_pointcloud.ok 与 ros_state.ok 均为 true；profile 输出两路状态频率、p50/p95/p99、min/max

bash scripts/rl/run_rl100_zarr_collect.sh collect --confirm-live --build-after
```

采集进程订阅下面两路事件驱动 command。它们不要求每个采样周期都有新消息，首次合法消息后永久
hold-last；任意一路先到都可以开始写样本：

```text
/kuavo_arm_traj                 # 14-D 机械臂目标，消息内为 degree
/control_robot_hand_position    # 强脑手左右各 6-D，范围 0..100
```

state 从两个原始状态 topic 读取，不用 command 覆盖 state。`/sensors_data_raw` 必须是完整 20 维
rad；强脑手 state 从 `/dexhand/state` 读取，必须严格是 12 维。顺序为：

```text
l_thumb, l_thumb_aux, l_index, l_middle, l_ring, l_pinky,
r_thumb, r_thumb_aux, r_index, r_middle, r_ring, r_pinky
```

RL-100 topic-native 契约为 `state32/action26`：`state = joint_q20 + dexhand_state12`，
`action = arm_traj.position14_deg + left_hand6 + right_hand6`。不做 16 维重排、手部标量压缩或单位归一化。
录制开始前会检查 dexhand feedback 是否接近固定默认值
`[0,99,0,0,0,0,0,99,0,0,0,0]`；手臂先动时，首条手部 command 前 action 使用默认 hold，
手部先动时，首条手臂 command 前使用录制起点实测双臂的 degree hold。

以下情况会直接丢弃当前 episode，不写入 staging：

- 整个 episode 没有收到任一路合法 command；
- raw state 维度/名称/范围、command 维度/范围不合法，或数据出现 NaN/Inf；
- 点云、joint state、dexhand state 连续超过 freshness/skew 阈值；
- 配置要求手部动作但 12 个手指 action 的最大 range 不足；
- command `received_at` 晚于采样 cutoff，或有效 header stamp 倒退。

每个 staging NPZ 还保存逐帧 `audit__*` 数组，包括 sample cutoff、两个状态时间、两个命令时间、
changed/seen/source、age/skew 和点云 freshness；这些字段不是塞进单个 `meta_json`。

### 采集后验收

```bash
bash scripts/rl/run_rl100_zarr_collect.sh inspect \
  --zarr-path data/rl100/grasp_8_4_v2/grasp_8_4_v2.zarr
```

必须满足：

```text
root.attrs.contract == rl100_topic_native_v1
data/state.shape[-1] == 32
data/action.shape[-1] == 26
action_quality.ok: true
causality_violation_count == 0
action_hand12_raw min/max 均在 [0,100]
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

## RL-100 真机部署

部署实现位于 `kuavo_rl/rl100_policy.py` 与 `kuavo_rl/rl100_real_runner.py`，topic-native 设计见
[`docs/RL100_TOPIC_NATIVE_COLLECTION_DEPLOYMENT_PLAN.md`](../../docs/RL100_TOPIC_NATIVE_COLLECTION_DEPLOYMENT_PLAN.md)。
默认配置 `configs/rl/rl100_real_deploy.yaml` 是 shadow 模式；现场批准的 14 维关节限位和
手部默认姿态未确认时，`live` 会拒绝启动。部署端复用同一 `TopicStateHub` 和三相机点云契约，
模型输出经安全门后直接发布 `/kuavo_arm_traj`（degree + `arm_joint_0..13`）与
`/control_robot_hand_position`（左右各 6 个 uint8）。

```bash
bash scripts/rl/run_rl100_real.sh inspect-checkpoint \
  --config configs/rl/rl100_real_deploy.yaml

bash scripts/rl/run_rl100_real.sh ros-preflight \
  --config configs/rl/rl100_real_deploy.yaml --duration-s 60

bash scripts/rl/run_rl100_real.sh shadow \
  --config configs/rl/rl100_real_deploy.yaml --max-steps 500
```

只有完成部署方案中的逐级验收后，才可运行 `live`；它还要求 `--confirm-live`、
`--physical-estop-ready` 和一次性 `--live-token`。不要用旧的
`third_party/RL-100/data/grasp_8_4.zarr` 做训练或回放。
- Launch: `configs/launch/hil_upper_cams.launch`
- Does not modify HIL topic profiles; `data/` is gitignored
