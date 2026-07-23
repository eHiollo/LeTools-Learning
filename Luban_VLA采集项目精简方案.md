# 鲁班全身 VLA 数据采集项目 — 精简迁移方案

> 执行说明：本方案交给 AI/工程师独立执行。**当前仓库（LeTools-Learning）一律不改**，
> 只读取、拷贝。产出一个新的独立项目文件夹。

## 1. 目标

从 `LeTools-Learning` 仓库中抽取已跑通的 HIL 采集脚本链路，生成一个**纯 bag 采集**
的新项目 `luban-vla-collect`，用于鲁班机器人全身 VLA 数据采集。

核心约束：

1. **交付物是原始 rosbag**，topic 布局对齐参考 bag（见 §7），**不是 RL 格式**。
   不写 `transitions.jsonl`、不导 `frames/`、没有 reward / intervention / obs-action
   transition 概念。
2. 保留采集脚本的质量保障能力：录前 gate、录制 watchdog、质检、staging→accepted
   原子发布、quarantine 隔离、SQLite + manifest 审计、Quest 手柄 episode 控制。
3. 保留相机 topic relay 机制（下位机 topic 名 → 参考 bag 规范名，不改下位机）。
4. 不依赖本仓库其余部分：不带 `kuavo_deploy/`、`kuavo_model/`、`kuavo_data/`、
   `kuavo_server/`、RL 训练与评估代码。

## 2. 新项目位置与结构

新项目创建在 `/home/leju_kuavo/wjy/robot-il/luban-vla-collect/`：

```text
luban-vla-collect/
├── README.md                              # 上手指南（见 §8 验收里要求的内容）
├── requirements.txt                       # pyyaml、numpy；ROS(noetic) 用系统环境
├── configs/
│   ├── collect_luban_v001.yaml            # 采集主配置
│   └── topics_luban_fullbody_v001.yaml    # 全身 topic 清单（含 source→name 相机映射）
├── vla_collect/
│   ├── __init__.py
│   ├── recording/                         # 录制引擎（从 kuavo_rl/hil_recording/ 平移）
│   │   ├── __init__.py
│   │   ├── session.py
│   │   ├── gate.py
│   │   ├── rosbag_recorder.py
│   │   ├── topic_relay.py
│   │   ├── watchdog.py
│   │   ├── quality.py
│   │   ├── publish.py                     # ← publish_replay.py 改名+瘦身
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── topics.py
│   │   ├── timebase.py
│   │   ├── result_events.py
│   │   ├── audit_publisher.py             # 简化：只发 /collect/episode_event
│   │   └── cli.py
│   ├── episode_control.py                 # ← kuavo_rl/quest_episode_control.py
│   ├── orchestrator.py                    # ← kuavo_rl/hil_collection.py 瘦身
│   └── collect_live.py                    # ← kuavo_rl/hil_collect_live.py 瘦身
├── scripts/
│   ├── collect.py                         # CLI 入口：preflight/collect/recover/list
│   └── check_bag_quality.py               # 离线单条 bag 质检
└── tests/
    └── test_recording.py                  # ← test_hil_recording.py 适配（dry-run 无 ROS）
```

## 3. 源文件 → 目标文件映射

源路径均相对 `/home/leju_kuavo/wjy/robot-il/LeTools-Learning/`。

### 3.1 直接平移（改 import 路径 `kuavo_rl.hil_recording` → `vla_collect.recording`，逻辑基本不动）

| 源 | 目标 | 说明 |
|---|---|---|
| `kuavo_rl/hil_recording/session.py` | `vla_collect/recording/session.py` | 状态机 + relay + rosbag 生命周期；见 §4.3 小改 |
| `kuavo_rl/hil_recording/gate.py` | `vla_collect/recording/gate.py` | 已支持探测 `spec.bus_name`（source），保持 |
| `kuavo_rl/hil_recording/rosbag_recorder.py` | `vla_collect/recording/rosbag_recorder.py` | 原样 |
| `kuavo_rl/hil_recording/topic_relay.py` | `vla_collect/recording/topic_relay.py` | 原样（topic_tools relay 管理） |
| `kuavo_rl/hil_recording/watchdog.py` | `vla_collect/recording/watchdog.py` | 原样 |
| `kuavo_rl/hil_recording/topics.py` | `vla_collect/recording/topics.py` | 原样（含 TopicSpec.source / relay_pairs） |
| `kuavo_rl/hil_recording/timebase.py` | `vla_collect/recording/timebase.py` | 原样 |
| `kuavo_rl/hil_recording/database.py` | `vla_collect/recording/database.py` | 原样（SQLite WAL） |
| `kuavo_rl/hil_recording/models.py` | `vla_collect/recording/models.py` | 见 §4.1 删字段 |
| `kuavo_rl/hil_recording/result_events.py` | `vla_collect/recording/result_events.py` | 原样 |
| `kuavo_rl/hil_recording/config.py` | `vla_collect/recording/config.py` | 目录名改动见 §4.2 |
| `kuavo_rl/hil_recording/cli.py` | `vla_collect/recording/cli.py` | 原样 |
| `kuavo_rl/quest_episode_control.py` | `vla_collect/episode_control.py` | 原样（事件源已是抽象接口） |

### 3.2 瘦身后迁移（删 RL 分支）

| 源 | 目标 | 要删/要改 |
|---|---|---|
| `kuavo_rl/hil_recording/quality.py` | `vla_collect/recording/quality.py` | 删 sidecar(transitions.jsonl) 与 bag audit 步数对账（`sidecar_step_count`/`bag_audit_step_count`/`sidecar_bag_match`）；改为"export 必需 topic 的消息数 ≥ 阈值"（阈值进 topics yaml，可选字段 `min_msgs`，默认 1）。保留 bag 可读性检查、export topic 存在性软告警 |
| `kuavo_rl/hil_recording/publish_replay.py` | `vla_collect/recording/publish.py` | 发布对象从 staging replay 目录改为 **bag 本体**：质检通过 → 把 `sessions/<eid>/bags/original.bag` 硬链接/rename 到 `accepted_bags/<eid>.bag`，并写 `publish_manifest.json`（含 quality_report sha256、label、时间戳）；失败/中止/重录 → episode 目录移入 `quarantine/`。`TRAIN_READY`/`REVIEW_READY` 标记文件机制保留（放在 `accepted_bags/<eid>.meta/` 下） |
| `kuavo_rl/hil_recording/audit_publisher.py` | `vla_collect/recording/audit_publisher.py` | 删 `/hil/transition_audit`（逐步审计，RL 专用）；保留结果事件，topic 改名 `/collect/episode_event`（std_msgs/String JSON：episode_id、event_type、stamps、payload） |
| `kuavo_rl/hil_collection.py`（990 行） | `vla_collect/orchestrator.py` | 保留：CollectionConfig / preflight / recover / episode 生命周期 / B 键标签流转（success/failure→accepted，abort/rerecord→quarantine）/ collection_manifest / collection_index / collection_events。删除：`dry_run_collect_episode` 里 HILReplayWriter/TransitionRecord 相关（约 757-830 行区域）、`auto_export_lerobot`、robometer、ACT 相关字段 |
| `kuavo_rl/hil_collect_live.py`（800 行） | `vla_collect/collect_live.py` | 删除：第 23-24 行 `ActExecuteFirstRunner`/`ActRunnerConfig` import 及 `HoldStatePolicy` runner（约 215 行区域，纯遥操不需要策略 runner）；第 42 行 `HILReplayWriter, TransitionRecord` import 及 369/403 行区域的 staging transition 写入（整段 replay writer 逻辑删掉，录制只靠 rosbag）。保留：长驻 session 循环（RESET→RECORD→保存）、Y+摇杆/B 键操作卡、estop 处理、phase 播报 |

### 3.3 新写（少量）

| 文件 | 内容 |
|---|---|
| `scripts/collect.py` | argparse CLI，子命令 `preflight` / `collect`(调 collect_live) / `recover` / `list`。参考 `scripts/rl/collect_hil_dataset.py` 的结构（该文件本身不拷贝，太多 HIL 子命令） |
| `scripts/check_bag_quality.py` | 离线质检一条 bag：`rosbag info` 可读性、各 topic 频率/最大断流、JPEG 魔数抽检、**全身关节运动量检查**（`/sensors_data_raw` joint_q 每维 span，低于阈值告警"疑似摆拍无动作"）。本仓库对话中已验证过该检查逻辑有效（能识别出无动作的 success episode） |
| `configs/collect_luban_v001.yaml` | 从 `configs/rl/hil_collection_real_v001.yaml` 改：删 `runner:`（policy/checkpoint/deploy_config/env_config 全删）、删 `auto_export_lerobot`/`lerobot_topic_profile`；`root` 改 `data/luban_vla_episodes`；`task_id` 等留占位 |
| `configs/topics_luban_fullbody_v001.yaml` | 从 `configs/rl/hil_topics_real_upper_cams_v001.yaml` 改，见 §5 |
| `tests/test_recording.py` | 从 `kuavo_rl/tests/test_hil_recording.py` 挑选并适配：状态机迁移、gate（含 `test_gate_probes_source_not_canonical_name`）、topics relay（`test_upper_cams_profile_relays_to_vla_canonical_names` 改到新 yaml）、dry-run 录制、publish/quarantine 流转。删除所有 HILReplayWriter/TransitionRecord 用例 |

### 3.4 明确不带

`kuavo_rl/` 其余全部（env、sac、actor/learner、act_runner、config、contracts、recording.py、
robometer、lerobot_patches、ros_adapter、backend 等）、`kuavo_deploy/`、`kuavo_model/`、
`kuavo_data/`、`kuavo_server/`、`scripts/`（除上述参考）、所有 HILSERL 文档 md。

## 4. 具体瘦身细节

### 4.1 `models.py`

- 删 `REPLAY_SCHEMA_VERSION` 及 replay 相关常量（若 config.py 引用需同步删）。
- `ResultEvent`/`EpisodeControlEvent`/状态常量（`STATE_*`、`PHASE_*`、`FINALIZED_OK`、
  `EXPORT_*`、`REVIEW_READY_MARKER` 等）全部保留 —— 状态机不动。
- 结果类型保留：`success` / `failure` / `abort` / `estop` / `rerecord` / `record_error`。

### 4.2 目录布局（`config.py` 中的路径生成）

```text
data/luban_vla_episodes/<collection_root>/
├── recording.db                       # SQLite（原 hil_recording.db）
├── collection_manifest.json           # 本次采集元信息（任务/操作员/配置hash/git head）
├── collection_index.json              # episode 索引
├── collection_events.jsonl            # 采集事件流水
├── sessions/<episode_id>/             # 录制现场（gate/watchdog/quality/日志/bags）
│   └── bags/original.bag
├── accepted_bags/                     # ★ 质检通过的交付 bag（下游只读这里）
│   ├── <episode_id>.bag
│   └── <episode_id>.meta/             # publish_manifest.json / TRAIN_READY / 质检报告副本
└── quarantine/<episode_id>/           # 质检失败/中止/重录（保留证据，不交付）
```

与原版的差别：没有 `accepted_replay/`（RL replay 目录）、没有 `staging/` 下的
transitions+frames（staging 目录本身可保留用于 bag 落盘前的临时性隔离，也可以直接
去掉 staging 概念，让 quality 直接检 `sessions/<eid>/bags/original.bag` —— 推荐后者，
更简单）。

### 4.3 `session.py` 小改

- 删除对 quality 里 sidecar 对账参数的传递（配合 §3.2 quality 瘦身）。
- `topic_relay` 启停逻辑保持现状（gate 探 source → 起 relay → rosbag 录 canonical 名
  → 停录后关 relay；relay 启动失败则 episode 进 `Failed(record_error)`）。
- audit_publisher 换成简化版（只发 `/collect/episode_event`）。

## 5. 鲁班全身 topic 配置模板

`configs/topics_luban_fullbody_v001.yaml` 结构沿用现有 schema（`TopicSpec` 字段：
`name/source/role/mode/required_for_start/required_for_export/min_hz/freshness_s/profiles`），
内容按鲁班实际改。模板：

```yaml
version: luban-fullbody-v001
robot_type: Luban
eef_type: TODO            # 按鲁班末端改
control_profile: teleop
topics:
  # ---- 全身状态/指令（名字按鲁班下位机实际 topic 填）----
  - name: /sensors_data_raw          # 全身关节状态
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 25
    freshness_s: 1.0
  - name: /joint_cmd                 # 全身关节指令
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 25
    freshness_s: 1.0
  - name: /kuavo_arm_traj            # 若鲁班有独立手臂轨迹 topic，改名
    role: training
    mode: streaming
    required_for_export: true
    min_hz: 25
    freshness_s: 1.0
  # TODO：鲁班全身特有——腿部/腰部/头部状态与指令 topic 逐条列出
  # ---- 末端 ----
  - name: /leju_claw_state           # 或鲁班灵巧手 state
    role: training
    mode: streaming
    required_for_export: true
  - name: /leju_claw_command         # ★ 末端指令必须录（参考 bag 有）
    role: training
    mode: streaming
    required_for_export: true
  # ---- TF / 标定 ----
  - name: /tf
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 10
    freshness_s: 2.0
  - name: /tf_static
    role: calibration
    mode: latched
    required_for_start: true
    required_for_export: true
  - name: /kuavo/arm_zeros           # 有则录（latched）
    role: calibration
    mode: latched
  - name: /kuavo/offset
    role: calibration
    mode: latched
  # ---- 相机：bag 内名字对齐参考布局，source 填鲁班驱动实际名 ----
  - name: /cam_h/color/image_raw/compressed
    source: "${ENV:LUBAN_HEAD_CAM_TOPIC:/camera/color/image_raw/compressed}"
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 10
    freshness_s: 1.0
  - name: /cam_l/color/image_raw/compressed
    source: "${ENV:LUBAN_WRIST_L_CAM_TOPIC:/left_wrist_camera/color/image_raw/compressed}"
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 10
    freshness_s: 1.0
  - name: /cam_r/color/image_raw/compressed
    source: "${ENV:LUBAN_WRIST_R_CAM_TOPIC:/right_wrist_camera/color/image_raw/compressed}"
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 10
    freshness_s: 1.0
  # ---- depth + camera_info：参考 bag 有；驱动有发就录（可选，不阻塞开录）----
  - name: /cam_h/depth/image_raw/compressedDepth
    source: "${ENV:LUBAN_HEAD_DEPTH_TOPIC:/camera/depth/image_raw/compressedDepth}"
    role: training
    mode: streaming
    required_for_start: false
    required_for_export: false
  - name: /cam_h/color/camera_info
    source: "${ENV:LUBAN_HEAD_CAMINFO_TOPIC:/camera/color/camera_info}"
    role: calibration
    mode: streaming
    required_for_start: false
    required_for_export: false
  # （cam_l / cam_r 的 depth + camera_info 同理各加一组）
  # ---- 采集事件审计 ----
  - name: /collect/episode_event
    role: audit
    mode: streaming
    required_for_export: true
```

## 6. RL 痕迹自查清单（执行后逐项确认）

- [ ] 新项目全文搜索无 `transitions.jsonl`、`HILReplayWriter`、`TransitionRecord`、
      `reward`、`intervention`、`accepted_replay`、`act_runner`、`robometer`、`lerobot`
- [ ] 录一条 dry-run，episode 目录内**只有** bag + 元数据 json/log，无 frames/、无 jsonl 数据文件
      （`collection_events.jsonl` 是采集流水审计，允许存在）
- [ ] `accepted_bags/` 下是 `.bag` 文件本体
- [ ] bag 内 topic 与 §7 参考布局一致（相机为 `/cam_h|l|r`）

## 7. 参考 bag 布局（数据形式对齐目标）

参考文件：`LeTools-Learning/A10-A15-I-L-05-TQ_01_01-5W_59-leju_claw-20260416145832-62-ec6791-v003.bag`
（如需查看先复制再 `rosbag reindex`）。topic 布局：

```text
/cam_h/color/image_raw/compressed            sensor_msgs/CompressedImage (848x480 jpeg)
/cam_h/color/camera_info                     sensor_msgs/CameraInfo
/cam_h/depth/image_raw/compressedDepth       sensor_msgs/CompressedImage (16UC1 png)
/cam_l/color/image_raw/compressed            + camera_info + depth(image_rect_raw)   同上
/cam_r/color/image_raw/compressed            + camera_info + depth(image_rect_raw)   同上
/joint_cmd                                   kuavo_msgs/jointCmd
/kuavo_arm_traj                              sensor_msgs/JointState (deg)
/leju_claw_command                           kuavo_msgs/lejuClawCommand
/leju_claw_state                             kuavo_msgs/lejuClawState
/sensors_data_raw                            kuavo_msgs/sensorsData
/kuavo/arm_zeros, /kuavo/offset              std_msgs/Float32MultiArray (latched)
/tf, /tf_static                              tf2_msgs/TFMessage
```

注意：`/cam_*/color/metadata`（orbbec/realsense Metadata）为驱动私有消息，非必需，
不列入 required。

## 8. 执行步骤与验收

1. 创建目录结构（§2），按 §3 拷贝/瘦身/新写文件，统一改 import 前缀为 `vla_collect.`
2. 写 `configs/` 两个 yaml（§3.3、§5）
3. 写 README：环境要求（ROS noetic、`topic_tools`、python3 + pyyaml/numpy）、
   一条命令开采示例、手柄操作卡（从 `hil_collect_live.py` 的
   `EPISODE_CONTROL_OPERATOR_CARD` 摘录）、目录布局说明（§4.2）
4. 跑测试：`pytest tests/ -q`（dry-run，无 ROS 依赖，全部通过）
5. dry-run 整链路验收（无 ROS）：
   ```bash
   python scripts/collect.py preflight --config configs/collect_luban_v001.yaml
   # dry_run_recorder: true 时跑一条假 episode：
   python scripts/collect.py collect --config configs/collect_luban_v001.yaml --dry-run
   ```
   预期：sessions/ 生成 episode 目录、gate.json Pass、假 bag 增长、B 标签后
   accepted_bags/ 出现产物、rerecord 进 quarantine/
6. 过一遍 §6 RL 痕迹自查清单
7. （真机阶段，非本次执行范围）鲁班上把 topics yaml 的 TODO/source 填实，
   `preflight` 通过后录一条真 bag，用 `scripts/check_bag_quality.py` 验证

## 9. 已知坑（来自本仓库真机联调经验）

1. **摆拍数据**：手柄按了 success 但机器人没动过 —— bag 完全健康也没用。
   `check_bag_quality.py` 的关节运动量检查（joint_q 每维 span 阈值）必须做，
   建议采集现场每条录完立即跑。
2. **相机频率不齐**：真机上头/左腕/右腕帧率可能是 60/10/30 不等，这不算错误，
   但 README 里要注明，下游对齐由转换器的主时间线机制处理。
3. **rosbag 未 stop 完就断电** → bag unindexed。recover 命令要保留
   （orchestrator 的 `recover_interrupted` 逻辑），并在 README 写明
   `rosbag reindex` 补救方式。
4. **relay 与 gate 顺序**：gate 探测的是 `source`（下位机真名），relay 起在
   gate 之后、rosbag 之前 —— 迁移时别改这个顺序，否则 required_for_start
   的 canonical 名探测不到会误报 Block。
5. **磁盘**：全身+三相机 bag 体积大（本仓库真机 30s ≈ 2.3GB 未压缩），
   gate 的磁盘阈值（start_block/hard_stop）要保留，README 注明预估每分钟体积。
