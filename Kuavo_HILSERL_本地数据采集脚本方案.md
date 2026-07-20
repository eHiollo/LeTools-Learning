# Kuavo HIL-SERL 本地数据采集脚本方案

> 目标：提供一个无需 KuavoBrain 平台、云端 task、NAS 或内部网关的本地采集入口，用于 ACT/VR HIL 数据采集。  
> 基础：复用已实现的 `kuavo_rl/hil_recording/`，不重新实现 session、gate、rosbag、watchdog、质量检查或 staging 发布。  
> 安全边界：脚本默认只做 preflight；只有明确指定真机模式并通过 Gate 后，才允许 runner 下发动作。

## 1. 为什么需要独立采集脚本

当前 `scripts/rl/eval_act_execute_first.py` 已可运行单次 ACT episode，并可通过 `--hil-recording` 接入 recorder。但它仍是“策略评估脚本”，不适合操作者持续采集：

- episode/task 命名、操作者、场景、物体初始条件没有统一入口；
- 缺少一次采集前的显式 preflight/recover；
- 缺少 accepted/quarantine 的采集汇总和离线标注复核队列；
- 不适合把 ACT、ACT+VR、VR-only 统一为同一套数据协议；
- 真机操作需要把“检查”和“实际下发动作”做成明显不同的命令。

新脚本只负责编排已有模块，不替代：

```text
RosTeleopAdapter        # VR、B 初始标记/中止、左手双键急停
KuavoHILSerlEnv         # action 仲裁、SafetyGate、reward
HILRecordingSession     # session、rosbag、watchdog、质量与发布
HILReplayWriter         # staging transition/frame 写入
ActExecuteFirstRunner   # ACT 每步重推理、execute-first
```

## 2. 用户入口

新增脚本：

```text
scripts/rl/collect_hil_dataset.py
```

新增可复用编排层：

```text
kuavo_rl/hil_collection.py
```

不建议让采集脚本直接 import `eval_act_execute_first.py`；后者是 CLI 评估入口，采集流程应放到可测试的库层。

### 2.1 子命令

```text
collect_hil_dataset.py preflight
collect_hil_dataset.py collect
collect_hil_dataset.py batch
collect_hil_dataset.py inspect
collect_hil_dataset.py recover
```

| 子命令 | 是否可能下发机器人动作 | 用途 |
|---|---:|---|
| `preflight` | 否 | 加载配置、recover、检查 ROS/topic/磁盘/profile/控制冲突。 |
| `collect` | 是，需明确确认 | 安全地采集一个 episode。 |
| `batch` | 是，需明确确认 | 交互式连续采集多个 episode；每条之间必须 reset 和人工确认。 |
| `inspect` | 否 | 查看 session、quality、accepted/quarantine、结果分布。 |
| `recover` | 否 | 恢复中断 session、处理孤儿 recorder 和 `.active` 文件。 |

`batch` 的 episode 状态机沿用本地 `third_party/lerobot` 的 reset / rerecord / end 原则，但操控入口改为 Quest **右手摇杆方向事件**。这三种采集事件与既有 VR 接管、B 标签建议和急停语义隔离。

## 3. 采集模式

```text
act        ACT 自主执行；VR 不接管或不启用
act_vr     ACT 默认执行；双 grip 接管对应手臂；松开回到 ACT
vr_only    不运行 ACT；仅用于人工示教/诊断采集
shadow     只观察/录制，不允许 policy 下发动作
```

`act_vr` 是 R0 和 HIL-SERL 的默认模式。它保留现有控制行为；任务成败以采集后的离线复核为准：

```text
未按 grip      → ACT 控制
按左/右 grip   → 对应手臂 VR 接管
松开 grip      → 对应手臂回到 ACT
B 单击         → success_candidate（操作者即时建议）
B 双击         → failure_candidate（操作者即时建议）
B 长按         → abort（立即停止原因，不等同任务失败标签）
左手双键       → estop
```

### 3.1 默认：按住 Y + 右摇杆（`quest_y_stick`）

真机遥操作里右摇杆平时用于腰部/底盘；**采集时用左手 `Y` 作修饰键**——只有按住 `Y` 时，右摇杆方向才变成 episode 控制。松 `Y` 后摇杆仍归遥操作。逻辑事件仍映射内部 `right_stick_*`。

按键约定：`Y=left_second`；`B` 仍只作成败标注。腰部总开关继续是 `X + 摇杆`（与 `Y + 摇杆` 区分）。

| 当前阶段 | 操作 | 行为 | 对机器人与数据的影响 |
|---|---|---|---|
| `RESETTING` | 按住 `Y` + 右摇杆 `→` | 结束 reset，开始录制 | Gate 通过后创建 episode，启动 rosbag 与 runner。 |
| `RECORDING` | 按住 `Y` + 右摇杆 `→` | 提前结束当前条 | `end_reason=early_end` → `pending_review`；可继续下一条。 |
| `RECORDING` | 按住 `Y` + 右摇杆 `←` | 放弃并重录 | finalize 后 quarantine(`rerecord`)，证据保留；回到 reset。 |
| `RESETTING` / `RECORDING` | 按住 `Y` + 右摇杆 `↓` | 结束整个 collection | 若在录制则先 finalize → `pending_review`；不再进入 reset。 |
| 任意阶段 | 终端 Ctrl-C | 结束整个采集 session | 安全 stop/finalize；不自动下一条。 |
| 任意阶段 | 左手双键（既有） | 急停 | 既有路径；当前条安全审计 quarantine。 |
| 任意阶段 | `B` 单击/双击/长按（既有） | 成败建议 / abort | 不改变 episode 控制；离线标注可覆盖。 |

要点：

- 未按 `Y` 时：采集脚本**不消费**右摇杆，腰部/底盘映射保持。
- 按住 `Y` 推摇杆瞬间：部分旧遥操作节点仍可能读到轴值；推完请马上松 `Y`。
- 可选备用：`episode_control: quest_y_chord`（`Y+A` / `Y+X`，完全不用摇杆）。

```text
RESETTING --Y+摇杆→--> RECORDING --Y+摇杆→--> FINALIZING --> PENDING_REVIEW --> RESETTING
    │                       │
    ├--Y+摇杆↓------------> COLLECTION_ENDED
    │                       ├--Y+摇杆←--> FINALIZING --> QUARANTINE(rerecord) --> RESETTING
    │                       ├--Y+摇杆↓--> FINALIZING --> PENDING_REVIEW --> COLLECTION_ENDED
    │                       └--Ctrl-C---> FINALIZING --> QUARANTINE(abort) --> COLLECTION_ENDED
    └--Ctrl-C--------------> COLLECTION_ENDED
```

边沿规则：越过阈值触发一次 → 摇杆回中或松开 `Y` 后才能再次触发。轴正负号建议用 `calibrate-stick` 写入本机 override。

### 3.2 遗留模式：右手摇杆独占（`quest_right_stick`）

仅当显式配置 `episode_control: quest_right_stick` 时启用（不按 Y，裸摇杆）。此时必须满足右摇杆独占与轴校准。默认 `quest_y_stick` **不**要求 `collection_mode_ack`。

## 4. 单 episode 协议

```text
0. 操作者摆放物体/确认工作区
1. preflight + recover（不下发动作）
2. 进入 RESETTING：机器人保持已知安全姿态，操作者摆放场景
3. 操作者右手摇杆推 `→`；创建 episode_id，冻结 policy/config/topic profile/git 元数据
4. 启动 HILRecordingSession；Gate 通过后启动 rosbag
5. recorder ready 后才启动 ACT/VR runner，进入 RECORDING
6. 运行期间记录 /hil/transition_audit 与 staging replay
7. 右手摇杆 `←` / `→` / `↓`、Ctrl-C、B 初始标记、SafetyGate、watchdog 或超时产生 EpisodeControlEvent 或 StopRequest
8. 安全 hold/stop，记录最终状态，完成 post-roll
9. 停 rosbag，异步质量检查
10. quality 通过 → pending_review；摇杆 `←` 则标记 rerecord 并转 quarantine；安全/质量失败 → quarantine
11. 摇杆 `→`：输出 summary 后 batch 返回 RESETTING；摇杆 `↓`：输出 summary 后结束整个 collection；单条模式均返回 shell
```

采集脚本不应自动把“文件存在”认定为采集成功。终端必须明确显示：

```text
episode_id
stop reason / B initial suggestion
record/session state
quality status
staging path / pending-review path 或 quarantine reason
```

## 5. 关键命令设计

### 5.1 preflight

```bash
python scripts/rl/collect_hil_dataset.py preflight \
  --config configs/rl/hil_collection_real_v001.yaml \
  --mode act_vr \
  --task-id box_to_chest_v1
```

必做事项：

- 执行 `HILRecordingSession.recover_interrupted()`；
- 加载 `hil_topics_v002.yaml` 并按 robot/eef/mode resolve；
- 检查 ROS master、必要 streaming topic、latched topic、磁盘和目录；
- 检查当前是否有活动 session、未登记控制 producer 或 recorder；
- 打印 resolved topic 与阈值；
- 输出 `Pass` 或 `Block`；`Block` 时退出码非零。

`preflight` 不加载 ACT checkpoint、不启动 rosbag、不发布控制消息。

### 5.2 collect

```bash
python scripts/rl/collect_hil_dataset.py collect \
  --config configs/rl/hil_collection_real_v001.yaml \
  --mode act_vr \
  --task-id box_to_chest_v1 \
  --scene-id table_a \
  --operator fulin \
  --max-steps 300 \
  --confirm-live
```

安全规则：

- 真机模式必须显式给出 `--confirm-live`；缺失时只允许 `--shadow` 或 dry-run。
- 每次 `collect` 只采一个 episode；episode 结束后必须返回 shell，避免无意连续下发动作。
- 单条模式支持右摇杆 `→` 提前结束、`↓` 完成当前条并结束 collection、`←` 安全结束后标记 `rerecord`；三者完成 finalize 后都返回 shell。Ctrl-C 记录 abort 并安全结束。
- `--episode-id` 可选；默认生成 `YYYYMMDDTHHMMSS_<task>_<uuid8>`。
- `--operator`、`--scene-id`、对象/目标初始条件进入 session metadata。
- `--max-steps` 与最大时长同时存在，任一达到都产生 timeout StopRequest。
- Ctrl-C 不是直接 kill：脚本捕获后写 `abort` 事件并走同一条安全 stop/finalize 流程；二次 Ctrl-C 才允许强制退出，后续由 `recover` 处理。

### 5.3 batch

```bash
python scripts/rl/collect_hil_dataset.py batch \
  --config configs/rl/hil_collection_real_v001.yaml \
  --mode act_vr \
  --task-id box_to_chest_v1 \
  --episodes 20 \
  --episode-time-s 90 \
  --reset-time-s 60 \
  --confirm-live
```

启动后终端固定打印：

```text
[RESETTING]  摆放物体并复位；右摇杆 → 开始录制；↓ 结束整个采集
[RECORDING]  右摇杆 ← 放弃重录；→ 提前结束本条并继续；↓ 完成本条并结束整个采集
```

`batch` 不使用每条 episode 的 y/n 确认，而是按 LeRobot 的 reset phase 循环。上一条 finalize 完成后进入 `RESETTING`；操作者在 `--reset-time-s` 内摆放场景。右手摇杆 `→` 可提前开始；超时后只提示并保持等待，**不会自动启动真机动作**。每条之间必须：

```text
机器人回到已知安全初始姿态
→ 人工摆放/确认物体
→ 新 session + 新 episode_id
→ 右手摇杆 `→` 后再次轻量 Gate
```

如出现 estop、SafetyGate fault、record_error、质量 Failed 或 watchdog stop，batch 默认立即停止，不自动进入下一条。`←` 产生的是受控 `rerecord` quarantine：它是唯一允许返回 `RESETTING` 的 quarantine 原因；不能绕过 safety stop 或质量失败。

## 6. 配置文件

新增：

```text
configs/rl/hil_collection_real_v001.yaml
configs/rl/hil_collection_sim_v001.yaml
```

示例：

```yaml
collection:
  task_id: box_to_chest_v1
  task_text: 将物料框搬运到胸前目标位置
  root: data/rl_runs/hilserl_episodes/hilserl_vr
  mode: act_vr
  default_max_steps: 300
  default_max_duration_s: 90
  reset_time_s: 60
  batch_stop_on_quarantine: true     # 除受控 rerecord 外，任意 quarantine 均停止 batch
  episode_control: quest_right_stick
  right_stick_trigger_threshold: 0.80
  right_stick_rearm_neutral_threshold: 0.20
  right_stick_debounce_s: 0.25
  right_stick_exclusive: true
  require_collection_mode_ack: true
  auto_start_after_reset: false

metadata:
  robot_type: Kuavo
  eef_type: leju_claw
  robot_version: unknown
  lower_commit: unknown
  scene_id: unset
  task_variant: default

recording:
  topics_profile: configs/rl/hil_topics_v002.yaml
  live_rosbag: true
  post_roll_s: 0.5
  start_block_disk_percent: 90
  hard_stop_disk_percent: 95
  allow_degraded_export: false

runner:
  policy: act
  checkpoint: <现场填写>
  shadow_mode: false
  ros_teleop: true
```

真实 checkpoint、机器人序列号和对象位姿不要硬编码进代码。敏感/机器专属值放本地 override 文件，并被 gitignore。

## 7. metadata 与采集索引

每条 session 已由 SQLite 保存运行状态；采集脚本还应写一个不可变 collection index：

```text
data/rl_runs/hilserl_episodes/hilserl_vr/
├── hil_recording.db
├── collection_manifest.json
├── collection_events.jsonl
└── collection_index.json
```

`collection_manifest.json`：一次 batch 的固定上下文。

```json
{
  "collection_id": "20260717_box_to_chest_a",
  "task_id": "box_to_chest_v1",
  "task_text": "...",
  "operator": "...",
  "scene_id": "table_a",
  "config_path": "...",
  "config_sha256": "...",
  "topics_profile_sha256": "...",
  "git_head": "...",
  "started_at_wall_ns": 0
}
```

`collection_events.jsonl` 是追加式事实日志，同一 episode 可以有多行，不允许原地覆盖。事件至少包括：

```text
collection_started / episode_started / episode_stopped
quality_finalized / export_transition
label_created / label_reviewed / label_rejected
collection_ended
```

`collection_index.json` 是可重建的查询快照，包含 episode、stop reason、label、质量和路径；它不是真实来源，损坏时必须能从 SQLite 与 `collection_events.jsonl` 重建。`RecordRequest.metadata` 必须真正写入 SQLite 的 `metadata_json`，不能只存在于进程内。

## 8. 两阶段数据治理：先采集、后标注、再入训练集

采集阶段不把 B 按键、timeout 或操作者主观感觉直接写成最终 `success/failure`。每条数据先完整保存可复核证据：rosbag、相机帧、transition audit、执行 action、VR intervention、stop reason 与安全事件。

```text
collect
  → recorder quality gate
  → publish_pending_review（数据可用但任务标签待定）
  → 离线回放/人工标注
  → label validation
  → publish_accepted（可导入训练）或 quarantine（不可用/安全审计）
```

`session_state` 只描述录制生命周期，保持 `Preparing → Recording → Stopping → Finalizing → Finalized(...)`，不把 `PendingReview` 混入 session state。数据发布状态单独扩展为：

```text
NotStarted → Staged → PendingReview → Published
                         └──────────→ Quarantined
NotStarted / Staged ─────────────────→ Quarantined
```

目录及完成标记：

```text
sessions/<episode_id>/staging/          # 正在写，任何 importer 禁止读取
pending_review/<episode_id>/REVIEW_READY
accepted_replay/<episode_id>/TRAIN_READY
quarantine/<episode_id>/
```

`publish_pending_review()` 只能接收 Finalized Healthy（或明确允许的 degraded）数据，并原子移动 staging；`publish_accepted()` 只能接收 `label_status=reviewed` 且未被安全规则否决的数据。训练 importer 只扫描 `accepted_replay/*/TRAIN_READY`，不以“目录存在”作为可用条件。

### 8.1 采集时写入的标签字段

```yaml
label_status: pending                 # pending | labeled | reviewed | rejected
operator_label_hint: unknown          # unknown | success_candidate | failure_candidate
stop_reason: unknown                  # success_button | failure_button | abort | timeout | estop | safety_fault | record_error
final_label: null                     # success | failure | abort | unsafe | invalid
failure_reason: null                  # empty_grasp | grasp_offset | drop | stuck | placement_error | ...
labeler: null
label_version: null
labeled_at_wall_ns: null
reviewer: null
reviewed_at_wall_ns: null
```

`stop_reason` 还必须支持 `early_end / collection_complete / rerecord`。这些属于 episode 控制原因，不进入 `ResultEvent` 的 success/failure 优先级。

`operator_label_hint` 是即时证据，离线标注可以覆盖它，但绝不删除原始按键事件。`stop_reason` 是运行事实，绝不被最终任务标签覆盖。

### 8.2 离线标注入口

此阶段不连接机器人、不发布 ROS 控制：

```text
collect_hil_dataset.py inspect --pending-review
collect_hil_dataset.py label <episode_id> --label success --reason verified_goal
collect_hil_dataset.py label <episode_id> --label failure --reason empty_grasp
collect_hil_dataset.py review <episode_id> --approve
collect_hil_dataset.py review <episode_id> --reject --reason camera_occluded
```

第一版 CLI 必须支持定位 rosbag/帧目录、记录标注者和版本，并以追加式审计记录保存每次变更。所有进入 accepted 的数据都必须执行一次显式 `review`；普通训练数据允许 reviewer 与 labeler 相同，reward 校准集和门禁评测集强制 reviewer 与 labeler 不同。

SQLite 新增两张表，不将所有可变标注字段堆入 `hil_sessions`：

```text
hil_episode_labels   # 每个 episode 当前有效标签快照
hil_label_events     # append-only：创建、修改、复核、拒绝及原因
```

同时将数据库 schema 升级为新版本并提供显式 migration；禁止直接用新 DDL 打开旧 `hil-db-v001` 后静默失败。

### 8.3 数据去向

| 采集/标注状态 | 质量 Healthy | 默认去向 | 说明 |
|---|---:|---|---|
| `pending` / `labeled` 未复核 | 是 | pending_review | 可保存和查看，不能被训练 importer 读取。 |
| `reviewed + success` | 是 | accepted_replay | 正任务正样本，可用于 success reward / classifier。 |
| `reviewed + failure` | 是 | accepted_replay | 保留为 RL failure/reward 数据；是否进入 BC 由 importer 决定。 |
| `reviewed + abort` | 是 | quarantine | 默认不作任务成败标签；可留作操作诊断。 |
| `reviewed + unsafe` / estop / safety_fault | 任意 | quarantine | 进入安全审计集，不进普通任务 replay。 |
| timeout | 是 | pending_review | 必须人工判定失败、abort 或 success，不能自动当失败。 |
| record_error / self_check fail | 否 | quarantine | 永不进入 accepted。 |

SAC importer 只能读取 `reviewed` 的 accepted 数据，并可按 success/failure/intervention mask 选择采样方式。线上稀疏 reward 若暂时需要即时信号，可单独消费 B 事件；它不得取代离线最终标签，也不得自动回填历史 replay。

### 8.4 事件类型边界

现有 `ResultEvent` 继续只表达 `success / failure / abort / estop / fault`。新增独立 `EpisodeControlEvent`：

```text
right_stick_left  → rerecord
right_stick_right → early_end
right_stick_down  → collection_complete
timeout           → timeout
B 单/双击         → success_candidate / failure_candidate
```

候选标签和 episode 控制事件写入 append-only event log，不直接覆盖 `result_type`。安全事件优先级仍为 `estop/fault > abort > failure > success`；任何安全事件都可否决后续发布，但不会删除原始数据。

## 9. 与现有 HIL 代码的接线

```text
collect_hil_dataset.py
  → HILCollectionOrchestrator
      → QuestEpisodeControlEventSource
      → verify_right_stick_exclusive / collection_mode_ack
      → HILRecordingSession.create/start
      → register ACT / VR producer
      → make_kuavo_hilserl_env(..., teleop=RosTeleopAdapter)
      → ActExecuteFirstRunner.run_episode(..., on_step=...)
          → HILReplayWriter(staging_dir=...)
          → session.update_transition(...)
      → session.record_event(...)
      → session.request_stop / wait_finalized
      → publish_pending_review
      → label / review
      → publish_accepted 或 quarantine
```

`on_step` 必须写入：

- policy action；
- executed action；
- raw VR audit action；
- SafetyGate 后 replay action；
- intervention mask、segment id、segment step；
- reward、fault、terminated/truncated；
- `TimeStamps` 和观测 source header stamp。

`collect` 不得修改 `KuavoHILSerlEnv` 的手臂 action 仲裁；它负责外层生命周期以及右摇杆采集事件的独占。若外部腰部/底盘控制无法确认已屏蔽右摇杆，Gate 必须拒绝启动。

## 10. 输出和操作者体验

每次 `collect` 结束打印：

```text
Collection: 20260717_box_to_chest_a
Episode:    20260717T103012_box_to_chest_1a2b3c4d
End:        collection_complete (quest_right_stick_down)
Hint:       success_candidate (quest_b)
Label:      pending_review
Record:     Finalized(Healthy)
Quality:    Healthy
Export:     Pending review
Pending:    data/.../pending_review/<episode_id>
Duration:   18.4 s
Intervention: left=2.1s right=0.0s
```

若失败，必须改成显眼的非零退出码与原因：

```text
Record: Failed(record_error)
Export: Quarantined
Reason: /sensors_data_raw stale for 2.4s
Recovery: python -m kuavo_rl.hil_recording.cli ... recover
```

## 11. 实施阶段

### C0：数据模型、脚本骨架与 migration（无 ROS）

- 新建 `kuavo_rl/hil_collection.py` 和 `scripts/rl/collect_hil_dataset.py`；
- 实现 `preflight/recover/inspect`；
- 定义可 mock 的 `EpisodeControlEvent` 与事件状态机，不订阅 ROS；
- 扩展 replay export 状态 `PendingReview`，新增 `pending_review_dir`、`REVIEW_READY`、`TRAIN_READY`；
- 新增 `hil_episode_labels`、`hil_label_events`、`metadata_json` 和 `hil-db-v001 → v002` migration；
- 拆分 `publish_pending_review()` 与 `publish_accepted()`，禁止现有 `publish_replay()` 绕过标签门禁；
- 生成 collection manifest/index；
- 单元测试参数、ID、metadata、退出码；
- 不修改机器人控制代码。

### C1：Quest 摇杆探测与独占 Gate（ROS，只读，不下发动作）

- 实现 `QuestEpisodeControlEventSource` 订阅 `/quest_joystick_data`；
- 实现左/右/下阈值、去抖、回中解锁和方向校准；
- 将校准结果写入本地 override，未校准 Block 真机 collect；
- 检查腰部/底盘旧消费者已关闭或返回 `collection_mode_ack=true`；
- 验证摇杆事件不会发布任何机器人 action。

### C2：单条 dry-run / MuJoCo

- 复用现有 `--hil-recording` 路径；
- 连接 ACT runner、HILReplayWriter staging 和 `wait_finalized`；
- 测试右摇杆 `→` early_end 并继续、`↓` collection_complete 并结束、`←` rerecord quarantine、Ctrl-C abort；
- 测试 success/failure candidate 与 abort/estop/fault/record_error 的事件边界；
- 断言质量通过后只生成 `pending_review/*/REVIEW_READY`，不会直接出现 accepted。

### C3：离线标注闭环（不连接机器人）

- 实现 `inspect --pending-review`、`label`、`review` 和追加式 label audit；
- 将质量 Healthy 的录制发布至 `pending_review`，禁止 importer 读取；
- 测试 success/failure/abort/unsafe、候选按键覆盖和双人复核；
- 生成按 final label / failure_reason / intervention 分布的采集报表。

### C4：batch 与交互保护

- 实现 reset hook、按原因 stop-on-quarantine（受控 rerecord 例外）；
- 实现 LeRobot 风格 RESETTING 状态、右摇杆 `→` 开始/提前结束、`←` 重录、`↓` 完成当前条并结束整个 collection；
- 实现 collection index；
- 测试 Ctrl-C 一次 abort、二次强退和 recover；验证 reset 超时绝不自动启动 runner。

### C5：仿真 live rosbag

- 使用 `--hil-recording-live-rosbag`；
- 确认 bag 包含 `/hil/transition_audit`、`/hil/result_event`；
- 对比 bag audit count 与 sidecar step count；
- 通过真机前置清单 Day A。

### C6：真机受控采集

- 按 `Kuavo_HILSERL_真机录制上线前置清单.md` 做 Day B/Day C；
- 真机只允许 `collect` 单条模式开始；
- 完成故障注入后才允许 `batch`；
- 先正常采集并进入 `pending_review`，完成离线标注后才形成 R0 reward 数据集。

## 12. 验收标准

- [ ] `preflight` 不下发控制、不启动 rosbag，能准确 Block；
- [ ] `recover` 可在新进程中恢复中断 session；
- [ ] `collect` 在未给 `--confirm-live` 时拒绝真机控制；
- [ ] reset 阶段无 ACT action、无 rosbag、无 replay transition；
- [ ] 右手摇杆 `←` 放弃当前条、`→` 提前结束当前条并继续、`↓` 完成当前条并结束整个 collection；每次触发后必须回中才能再次触发；
- [ ] 右手摇杆方向在 preflight 校准；未校准或没有 `collection_mode_ack` 时真机 collect Block；
- [ ] 采集模式下右摇杆不会同时进入腰部/底盘 action，退出后才恢复原映射；
- [ ] recorder ready 后 runner 才开始动作；
- [ ] B/急停/SafetyGate/watchdog 走统一 stop 流程；
- [ ] episode 结束后等待 `wait_finalized`，不抢跑 publish；
- [ ] `pending_review` 仅含质量 Healthy、带 `REVIEW_READY` 的 episode；
- [ ] 未经 `reviewed` 的数据不能生成 `TRAIN_READY`，训练 importer 不能读取；
- [ ] 旧 `hil-db-v001` 可显式迁移至 v002，metadata 与 label audit 不丢失；
- [ ] failure 与 success 的 recorder 质量语义相同；
- [ ] estop/fault/record_error 不进入 accepted；
- [ ] 离线标注可保留 B 初始建议并可审计地覆盖；
- [ ] batch 在 safety/quality quarantine 后自动停止并保留证据；
- [ ] rerecord 只取消训练资格、完整保留原始审计证据；
- [ ] collection index 可从 episode 回溯 config、topic profile、policy 和操作者；
- [ ] MuJoCo live rosbag 和真机前置清单通过；
- [ ] 不需要 KuavoBrain 平台、云端连接或 NAS。

## 13. 第一批 AI 实施范围

第一步仅执行 C0，允许改动：

```text
kuavo_rl/hil_collection.py
kuavo_rl/hil_recording/models.py
kuavo_rl/hil_recording/config.py
kuavo_rl/hil_recording/database.py
kuavo_rl/hil_recording/publish_replay.py
scripts/rl/collect_hil_dataset.py
kuavo_rl/tests/test_hil_collection.py
kuavo_rl/tests/test_hil_recording.py
configs/rl/hil_collection_sim_v001.yaml
```

禁止改动：

```text
third_party/kuavo_brain/
kuavo_rl/env.py
kuavo_rl/ros_teleop.py
kuavo_rl/safety.py
真实机器人启动脚本
```

C0 测试通过后再单独执行 C1；先完成 C3 离线标注闭环，再做 batch、真 rosbag 和真机控制，避免未标注数据进入训练集。C1 才允许新增 Quest ROS 事件源；C0 禁止连接 ROS 或启动机器人。
