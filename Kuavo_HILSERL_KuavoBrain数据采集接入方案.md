# Kuavo HIL-SERL 接入 KuavoBrain 数据采集框架方案（v2 修订版）

> 目标：将 `third_party/kuavo_brain/kuavobrain-v3.x` 中经过验证的本地数据采集可靠性机制迁移到当前 HIL-SERL 项目。
> 原则：只迁移本地录制、质量检查和恢复能力；不依赖公司云端、NAS、OTA、内部网关或内部推理服务。
> 本版修订要点：ACT 推理是合法并行 producer 而非冲突项；统一双时间基准；replay 采用 staging→原子发布；停止顺序保证终止时刻不丢失；topic 分级 profile；stop/export 异步化；HIL 关键事件进入 rosbag 审计。

## 0. 全局架构不变量（AI 实现时必须遵守）

以下不变量贯穿全部模块，任何实现与此冲突即为错误：

1. **单一可写事实源**：SQLite 是 session 状态的唯一可写事实源。`session.json` 只是 SQLite 的导出快照（只读、可再生），二者不允许同时作为可写状态。
2. **图像事实源**：H.265 rosbag 是原始审计源（source of truth）；`HILReplayWriter` 写出的 JPEG/NPY 是训练派生物（derived artifact）。任何数据争议以 bag 为准。
3. **合法并行**：HIL 的正常运行形态就是 `ACT 推理 + rosbag 录制 + VR 随时接管` 三者同时存在。录制 Gate 不得把 ACT/SAC runner 判定为冲突。
4. **双时间基准**：所有事件必须同时携带 ROS 时间与本机单调时间（见 §7），不允许把 `time.time()` 或相对启动时间直接当作 ROS 时间使用。
5. **两条判定链分离**：控制安全（机器人是否安全停止）与数据质量（episode 是否可训练）是两条独立判定链；`failure` 是任务结果，`Failed` 是数据/系统状态。
6. **未发布即不可训练**：只有通过质量检查并被原子发布到 `accepted_replay/` 的 episode 才可能进入训练。staging 与 quarantine 中的数据对下游训练代码不可见。
7. **单线程停止协调**：只有 episode controller（主控制线程）可以执行停止序列。Watchdog、监控线程只能提交 stop request，不得直接调用 `env.step()`、robot 控制或 recorder 停止。
8. **只读边界**：迁移过程中不修改 `third_party/kuavo_brain/kuavobrain-v3.x` 的任何文件。

## 1. 接入目标

当前 HIL 已经具备 VR 接管、ACT 执行、人工 reward 和 HIL replay 写入，但录制可靠性仍由 Python runner 直接负责。接入后要形成下面的闭环：

```text
HIL episode 控制器（唯一停止协调者）
  ├── ACT policy（登记为 session 合法 producer）
  ├── VR teleop / B reward / 左手双键急停
  ├── SafetyGate
  └── HIL recording session
        ├── 录前 gate（互斥检查不包含 ACT）
        ├── rosbag recorder（含 /hil/* 审计 topic）
        ├── 录中 watchdog（只发 stop request）
        ├── 录后 bag quality（异步 finalize）
        ├── SQLite session state（WAL + 事务）
        └── staging → accepted_replay 原子发布
```

最终只有同时满足以下条件的 episode 才能进入训练：

```text
控制结果明确
  ∧ rosbag 正常结束（含 post-roll）
  ∧ 必要 topic 按各自 profile 规则质量合格
  ∧ state/action/image 以 ROS 时间可对齐
  ∧ reward/result 事件已双通道落盘（sidecar + /hil/result_event bag）
  ∧ replay 已从 staging 原子发布到 accepted_replay
```

## 2. 不迁移的内容

当前项目直接在本地运行，因此不复制 KuavoBrain 的以下模块：

| 模块 | 处理方式 |
|---|---|
| `kb_cloud` | 不接入平台 API；本地保存导出状态。 |
| `kb_upload` | 不上传 NAS/云端；保留 `ReadyToExport`，由本地脚本处理。 |
| `kb_ota` / `kb_ota_agent` | 不影响 HIL 录制。 |
| `kb_gateway_http/ws` | 初期用 Python API/CLI；不引入内部远程命令协议。 |
| `kb_inference` | 继续使用当前 ACT/SAC runner。 |
| Debian packaging | 使用当前项目环境和脚本。 |
| 内部云端 task 协议 | 使用本地 `episode_id` 和 HIL 配置生成最小元数据。 |

迁移的是 `kb_storage + kb_ros + kb_record + kb_record_monitor` 的语义和必要实现。

## 3. 当前实现对照

| KuavoBrain 能力 | 当前 HIL 对应物 | 接入动作 |
|---|---|---|
| `RecordSessionManager` | `ActExecuteFirstRunner` + `HILReplayWriter` | 新增 `HILRecordingSession`，把 runner 生命周期纳入 session。 |
| `StorageLayout` / SQLite | `data/rl_runs/hilserl_episodes` + JSONL | 增加轻量 SQLite（WAL、外键、schema version），JSONL 继续作为逐步审计文件。 |
| `topics.yaml` resolver | `RosTeleopAdapter` / `KuavoHILSerlEnv` 的 ROS topic | 新增分级 topic profile（role/mode/required 分离）和 resolved topic 快照。 |
| `RecordGateEvaluator` | 当前没有统一录前 gate | 新增 gate：ROS、核心 topic、时间、磁盘、**recorder/session 互斥（不含 ACT）**。 |
| `RecorderWatchdog` | 当前没有 recorder 级 watchdog | 新增 PID、bag 增长、topic freshness、磁盘和限额监控；只向 controller 提交 stop request。 |
| `BagQualityChecker` | 当前仅写 transition/replay | 新增停止后异步 bag 质量报告，作为 replay 发布前置条件。 |
| `RecoverInterruptedSessions` | 当前 runner 异常后无统一恢复 | 启动时扫描活动 session、孤儿进程和 `.active` 文件，staging 数据移入 quarantine。 |
| `B` reward 事件 | `RosTeleopAdapter.poll()` | 保留事件语义，增加双时间戳、持久化关联，并发布到 `/hil/result_event`。 |
| VR action 规范化 | `KuavoHILSerlEnv.step()` | 保持当前 raw audit / mask / SafetyGate 后 action 语义不变，并发布到 `/hil/transition_audit`。 |

现状确认（AI 实现前应知道的代码事实）：

- `kuavo_rl/recording.py` 中 `HILReplayWriter` 目前在 episode 运行时就把 JPEG/NPY/JSONL 写进正式 replay 目录 —— 本方案改为写入 session staging 目录（§6、§10）。
- `kuavo_rl/backend.py` 中 `MockBackend.get_observation()` 的 `timestamp_s` 是相对进程启动的 wall time，不是 ROS 时间 —— 本方案要求所有时间字段显式声明时间基（§7）。

## 4. 目标目录

```text
kuavo_rl/hil_recording/
├── __init__.py
├── config.py                 # HIL recording 配置和版本
├── models.py                 # Session/Bag/Gate/Watchdog/TimeStamps 数据结构
├── timebase.py               # ROS/monotonic/wall 双时间基准采样与换算
├── topics.py                 # robot_type + eef_type + control_profile → resolved topics
├── database.py               # SQLite schema、PRAGMA、事务、状态迁移
├── session.py                # create/start/request_stop/wait_finalized/cancel/recover
├── rosbag_recorder.py        # rosbag 子进程、PID、日志和 active 文件
├── audit_publisher.py        # /hil/transition_audit 与 /hil/result_event 发布
├── gate.py                   # 录前 gate（互斥不含 ACT）
├── watchdog.py               # 录中检查；只产生 StopRequest，不直接控制
├── quality.py                # 录后 bag 检查（异步 finalize 线程内执行）
├── result_events.py          # success/failure/abort/estop 事件
├── publish_replay.py         # staging → accepted_replay 原子发布 / quarantine
└── cli.py                    # 本地诊断和录制命令
```

现有文件保持职责不变：

```text
kuavo_rl/recording.py          # TransitionRecord/HILReplayWriter（输出根目录改为 staging）
kuavo_rl/act_runner.py         # episode 执行和 on_step 回调
kuavo_rl/env.py                # reward、intervention mask、SafetyGate action
kuavo_rl/ros_teleop.py         # VR 与 B/急停事件解析
scripts/rl/eval_act_execute_first.py  # 入口脚本
```

## 5. Session 数据模型

### 5.1 双状态机：session_state 与 replay_export_status 分离

录制状态机（`session_state`，描述"这次录制发生了什么"）：

```text
Preparing
  ↓ gate_pass
Recording
  ↓ stop_request(user | limit | watchdog | terminal_event)
Stopping            # 执行 §9.2 停止序列：terminal event → 安全保持 → post-roll → 停 bag
  ↓ recorder_exit
Finalizing          # 异步：bag 可读性 + 质量检查
  ↓ quality_pass
Finalized(Healthy)
  ↓ quality_degraded_accepted
Finalized(ManuallyAcceptedDegraded)
```

异常状态：

```text
Failed(record_error)   # recorder 崩溃、进程退出、bag 不可读、磁盘 hard-stop
Failed(self_check)     # bag 存在但 topic/频率/质量不合格
Canceled               # 用户主动取消且明确不作为任务失败
Deleted                # 明确删除，不再参与训练
```

导出状态机（`replay_export_status`，描述"训练数据发布到哪一步"）：

```text
NotStarted → Staged → Published        # 原子发布到 accepted_replay/
                   → Quarantined       # 质量失败/急停/record_error
```

两套状态独立迁移：`ReadyToReplay` 之类的导出概念不允许混入录制状态机。状态迁移必须通过 `database.py` 中的白名单迁移函数完成（非法迁移直接抛错），且每次迁移在单个 SQLite 事务内同时更新状态字段与写入 `hil_events` 审计行。

`failure` 是任务结果，`Failed` 是数据/系统状态，二者必须分开。

### 5.2 SQLite 设置与最小表

数据库打开时必须执行：

```sql
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
```

并维护 `schema_version` 表（首版 `hil-db-v001`），启动时校验版本，不匹配即拒绝启动而不是静默继续。

```sql
CREATE TABLE schema_version (
  version TEXT NOT NULL
);

CREATE TABLE hil_sessions (
  episode_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  session_state TEXT NOT NULL,
  replay_export_status TEXT NOT NULL DEFAULT 'NotStarted',
  created_at_wall_ns INTEGER NOT NULL,
  started_at_ros_ns INTEGER,
  stopped_at_ros_ns INTEGER,
  robot_type TEXT,
  robot_version TEXT,
  lower_commit TEXT,
  eef_type TEXT,
  control_profile TEXT NOT NULL,        -- act | act_vr | vr_only
  topics_version TEXT,
  resolved_topics_json TEXT NOT NULL,
  gate_status TEXT NOT NULL DEFAULT 'Pending',
  watchdog_status TEXT NOT NULL DEFAULT 'Disabled',
  quality_status TEXT NOT NULL DEFAULT 'Pending',
  result_type TEXT,
  result_event_ros_ns INTEGER,          -- ROS 时间（ns）
  result_event_mono_ns INTEGER,         -- 本机单调时间（ns）
  result_event_source TEXT,
  session_dir TEXT NOT NULL,
  record_pid INTEGER DEFAULT 0,
  record_command TEXT,
  stdout_path TEXT,
  stderr_path TEXT,
  watchdog_report_path TEXT,
  watchdog_log_path TEXT,
  quality_report_json TEXT,
  error_message TEXT
);

CREATE TABLE hil_bags (
  bag_id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL,
  bag_type TEXT NOT NULL,             -- original | preview
  path TEXT NOT NULL,
  state TEXT NOT NULL,
  size_bytes INTEGER DEFAULT 0,
  duration_sec REAL DEFAULT 0,
  quality_report_json TEXT,
  UNIQUE (episode_id, bag_type),
  FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
);

CREATE TABLE hil_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL,
  event_type TEXT NOT NULL,           -- success/failure/abort/estop/fault/state_transition
  ros_time_ns INTEGER,
  monotonic_time_ns INTEGER NOT NULL,
  wall_time_ns INTEGER NOT NULL,
  source_header_stamp_ns INTEGER,     -- 来源消息 header.stamp（若有）
  source TEXT NOT NULL,
  payload_json TEXT,
  FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
);
```

职责边界：

- SQLite：session 状态唯一可写事实源；
- `transitions.jsonl`（sidecar）：逐步审计和 replay 构建输入；
- `/hil/transition_audit`、`/hil/result_event`（bag 内）：崩溃安全的审计源，与 sidecar 互为校验；
- `session.json`：SQLite 导出的只读快照，供人工浏览，任何代码不得据此写状态；
- `schema.json`：文件格式说明。

## 6. 文件布局（staging → 发布）

```text
data/rl_runs/hilserl_episodes/hilserl_vr/
├── hil_recording.db                       # 唯一可写事实源
├── sessions/<episode_id>/
│   ├── session.json                       # DB 导出快照（只读、可再生）
│   ├── gate.json                          # 录前检查
│   ├── watchdog.report.json               # 最终 watchdog 报告
│   ├── watchdog.events.jsonl              # 事件日志
│   ├── record.stdout.log
│   ├── record.stderr.log
│   ├── bags/
│   │   ├── original.bag                   # H.265 原始审计源（含 /hil/* topic）
│   │   └── preview.c.bag
│   └── staging/                           # 录制中的 replay 派生物，训练不可见
│       ├── transitions.jsonl
│       └── frames/
├── accepted_replay/<episode_id>/          # 质量通过后从 staging 原子发布（rename）
│       ├── transitions.jsonl
│       ├── frames/
│       └── publish_manifest.json          # 质量报告 hash、发布时间、schema 版本
├── quarantine/<episode_id>/               # 质量失败 / record_error / 急停数据
└── transitions.jsonl                      # 全局审计索引
```

发布规则：

- `HILReplayWriter` 在 episode 运行期间只写 `sessions/<episode_id>/staging/`；
- 质量通过后由 `publish_replay.py` 用同分区 `os.rename`（或 rename 目录）原子移动到 `accepted_replay/<episode_id>/`，并写入 `publish_manifest.json`；
- 质量失败、`record_error`、急停 episode 移入 `quarantine/<episode_id>/`，保留用于安全分析，不靠 SQLite 状态"提醒"下游不要读；
- 下游训练加载器只允许扫描 `accepted_replay/`。

## 7. 时间基准

每条 transition、每个结果事件、每次状态迁移必须同时记录：

```text
ros_time_ns              # rospy.Time.now()，对齐主时钟
monotonic_time_ns        # time.monotonic_ns()，进程内测量时钟
wall_time_ns             # time.time_ns()，仅用于人读展示
source_header_stamp_ns   # 来源 ROS 消息 header.stamp（若该事件由消息触发）
```

对齐规则（固定，不允许各模块自行选择）：

- 图像、state、action、VR 事件之间的相互对齐一律使用 **ROS 时间**（优先 `source_header_stamp_ns`，无 header 时用采样点的 `ros_time_ns`）；
- 超时、watchdog 间隔、post-roll 时长等时长测量一律使用 **monotonic**；
- wall time 只出现在文件名、日志和 `session.json` 中，禁止参与对齐或时长计算。

`timebase.py` 提供 `now_stamps() -> TimeStamps`，一次采样同时返回三个时钟，保证同一事件的三个时间戳来自同一采样点。使用 `use_sim_time` 时 ROS 时间可能与 wall 有大偏差，属正常，禁止用 wall 校正 ROS 时间。

现有代码中 `timestamp` / `timestamp_s` 字段一律视为遗留 wall/相对时间，接入时不得当作 ROS 时间参与对齐。

## 8. 录制与 HIL runner 的接口

### 8.1 录制 session 接口

```python
class HILRecordingSession:
    def create(self, request: RecordRequest) -> SessionSnapshot: ...
    def register_producer(self, name: str, pid: int, kind: str) -> None:
        """把 ACT/SAC runner、teleop adapter 登记为本 session 合法 producer。"""
    def start(self, episode_id: str) -> SessionSnapshot: ...
    def record_event(self, event: ResultEvent) -> None: ...
    def update_transition(self, info: dict) -> None: ...
    def request_stop(self, episode_id: str, reason: str) -> None:
        """非阻塞。只提交停止请求；实际停止序列由 episode controller 执行。"""
    def wait_finalized(self, episode_id: str, timeout_s: float) -> SessionSnapshot:
        """阻塞等待 Stopping → Finalizing → Finalized/Failed 完成。"""
    def publish_replay(self, episode_id: str) -> ExportReport:
        """只接受 Finalized(Healthy | ManuallyAcceptedDegraded)；否则移 quarantine 并抛错。"""
    def cancel(self, episode_id: str) -> SessionSnapshot: ...
    def recover_interrupted(self) -> RecoveryReport: ...
```

### 8.2 runner 接入时机

```python
session = recorder.create(request)
recorder.register_producer("act_runner", os.getpid(), kind="policy")
session = recorder.start(episode_id)       # gate 通过后才开始 runner

def on_step(step_id, observation, action, next_observation, info):
    recorder.update_transition({
        "step_id": step_id,
        "stamps": timebase.now_stamps(),            # ros/mono/wall 三时钟
        "source_header_stamp_ns": info.get("obs_header_stamp_ns"),
        "intervention_mask": info.get("intervention_mask"),
        "intervention_segment_id": info.get("intervention_segment_id"),
        "intervention_segment_step": info.get("intervention_segment_step"),
        "teleop_replay_action": info.get("teleop_replay_action"),
    })

result = runner.run_episode(env, max_steps=steps, on_step=on_step)

# 停止与发布是异步两阶段，不允许 stop 后立刻同步导出
recorder.request_stop(episode_id, reason=result.termination_reason)
final = recorder.wait_finalized(episode_id, timeout_s=30.0)
if final.session_state in ("Finalized(Healthy)", "Finalized(ManuallyAcceptedDegraded)"):
    recorder.publish_replay(episode_id)
```

需要注意：录制器不能改变 `env.step()` 的 action。它只记录 policy action、raw VR action、mask、实测 state 和 SafetyGate 后 action。

### 8.3 结果事件优先级

```text
estop / SafetyGate fault
        > abort
        > failure
        > success
```

结果事件必须去重，并按 ROS 时间排序。B 单击的延迟确认、双击和长按逻辑继续由当前 `RosTeleopAdapter` 负责；`HILRecordingSession` 负责可靠落盘和 episode 关联。

### 8.4 HIL 审计 topic（进入 rosbag）

新增两个 ROS topic，由 `audit_publisher.py` 发布并被 rosbag 录制，保证 Python runner 崩溃时 bag 自身仍是完整审计源：

```text
/hil/transition_audit    # 每步一条
  episode_id, step_id, header.stamp(ROS),
  policy_action, executed_action, raw_vr_action,
  intervention_mask, intervention_segment_id, intervention_segment_step,
  reward, fault_code

/hil/result_event        # 每个结果事件一条
  episode_id, header.stamp(ROS),
  event_type(success|failure|abort|estop|fault),
  source, payload_json
```

sidecar（`transitions.jsonl`）与 bag 内审计流互为校验：质量检查阶段比对两者的 step 数与时间戳，偏差超阈值判 `Failed(self_check)`。

## 9. 录前 Gate 设计

### 9.1 互斥语义（本版关键修订）

HIL 正常模式是 **ACT 推理 + rosbag 录制 + VR 随时接管**，三者并行是常态。Gate 只阻止以下冲突，**不得把 ACT/SAC runner 当成冲突项**：

```text
阻止：
  - 第二个 rosbag recorder（同一 session_dir 或同一 topic 集）
  - 另一个处于 Recording/Stopping 的 episode/session
  - 非 HIL 体系的外部控制节点（不在 producer 登记表中却在发控制 topic）
  - 未恢复的历史活动 session（存在 .active 文件或 DB 中 Recording 残留）

放行：
  - 已通过 register_producer 登记的 ACT/SAC runner
  - 当前 session 的 teleop adapter / VR 节点
```

### 9.2 必检项目

```text
ROS master 可用
streaming 类核心 topic 已发布且最近频率达阈值
latched 类 topic（如 /tf_static）已收到至少一条消息（不做 Hz/freshness 检查）
相机/末端 topic 按 profile 解析后存在
ROS 时间可用且与 monotonic 采样一致推进
磁盘低于 start_block_percent
recorder/session 互斥检查（按 §9.1，不含 ACT）
record 输出目录可写
```

Gate 输出必须包含：

```json
{
  "status": "Pass|Block|Degraded",
  "checked_at_ros_ns": 123400000000,
  "checked_at_mono_ns": 456700000000,
  "missing_topics": [],
  "low_rate_topics": [],
  "stale_topics": [],
  "latched_missing": [],
  "disk_usage_percent": 42.1,
  "producers": [{"name": "act_runner", "pid": 12345}],
  "reasons": []
}
```

`Block` 禁止启动录制；`Degraded` 只有在明确配置允许且最终人工确认时才能发布 replay。

### 9.3 分级 topic profile

每个 topic 单独声明角色与检查模式，不共用一套规则：

```yaml
version: hil-v002
robot_type: Kuavo
eef_type: leju_claw
control_profile: act_vr          # act | act_vr | vr_only
topics:
  - name: /sensors_data_raw
    role: training               # training | audit | calibration
    mode: streaming              # streaming | latched
    required_for_start: true
    required_for_export: true
    min_hz: 25
    freshness_s: 1.0
  - name: /joint_cmd
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 25
    freshness_s: 1.0
  - name: /tf
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 10
    freshness_s: 2.0
  - name: /tf_static
    role: calibration
    mode: latched                # 只检查"收到过"，不检查 Hz/freshness
    required_for_start: true
    required_for_export: true
  - name: /kuavo_arm_traj        # 仅 VR/IK 相关 profile 强制
    role: training
    mode: streaming
    required_for_start: false
    required_for_export: true
    profiles: [act_vr, vr_only]
    min_hz: 25
    freshness_s: 1.0
  - name: /ik_fk_result/input_pos
    role: audit
    mode: streaming
    required_for_start: false
    required_for_export: false
    profiles: [act_vr, vr_only]
  - name: /ik_fk_result/eef_pose
    role: audit
    mode: streaming
    required_for_start: false
    required_for_export: false
    profiles: [act_vr, vr_only]
  - name: /leju_claw_state
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 10
    freshness_s: 2.0
  - name: /cam_h/color/h265_stream
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 20
    freshness_s: 1.0
  - name: /cam_l/color/h265_stream
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 20
    freshness_s: 1.0
  - name: /cam_r/color/h265_stream
    role: training
    mode: streaming
    required_for_start: true
    required_for_export: true
    min_hz: 20
    freshness_s: 1.0
  - name: /hil/transition_audit
    role: audit
    mode: streaming
    required_for_start: false    # 由本系统自己发布，start 时尚未有消息
    required_for_export: true
  - name: /hil/result_event
    role: audit
    mode: streaming
    required_for_start: false
    required_for_export: true
  - name: /leju_claw_command
    role: audit
    mode: streaming
    required_for_start: false
    required_for_export: false
  - name: /kuavo/arm_zeros
    role: calibration
    mode: latched
    required_for_start: false
    required_for_export: false
  - name: /kuavo/offset
    role: calibration
    mode: latched
    required_for_start: false
    required_for_export: false
```

规则：

- `mode: latched` 的 topic 只做"是否收到过至少一条"检查，禁止套用 Hz/freshness；
- 带 `profiles` 字段的 topic 只在对应 control_profile 下参与检查；纯 ACT episode 不强制 `/kuavo_arm_traj` 和 IK topics；
- 实际 topic 必须由现场 `rostopic list` 和当前机器人型号确认，不能仅凭模板启动。

## 10. Watchdog 与停止协议

### 10.1 运行中

监控循环频率与开销约束：

- 500 ms 主循环：**只读取缓存状态**（recorder PID 存活、上次 bag stat 结果、各 topic 最近消息时间的内存缓存），不做任何重型探测；
- 1 s：stat bag 文件大小并更新增长时间；
- 5 s：磁盘用量检查；
- topic 频率的重型 probe（如需要）不进入 500 ms 循环，由订阅回调持续更新"最近消息时间"内存表，watchdog 只读该表。

需要记录：

- recorder PID 是否存在；
- bag 文件大小和最后增长时间；
- 核心 streaming topic 的最近消息时间（latched topic 不参与 freshness）；
- 低频、丢失、stale、最大 gap；
- watchdog 自身异常。

**线程边界（硬性约束）**：watchdog 运行在监控线程中，发现问题时只能构造 `StopRequest(reason=...)` 提交给 episode controller 的队列，由主控制线程执行停止；watchdog 不得直接调用 `env.step()`、robot 控制接口或 recorder 停止函数。`bag stalled` 时提交 `StopRequest(record_error)`，不要继续让 runner 运行并产生无法对齐的 transition。

### 10.2 停止序列（保证终止时刻不丢失）

由 episode controller 单线程执行，顺序固定：

```text
1. 记录 terminal event（B 结果 / 急停 / 超限），写 SQLite + /hil/result_event
2. 控制进入安全保持/停止（runner 不再下发新 action）
3. rosbag 继续录制 post-roll（默认 0.5 s，可配 0.3–1.0 s），
   捕获 B 结果确认、急停后的最终关节状态和最后几帧图像
4. 确认最终 state 消息已收到（以 ROS 时间判断晚于 terminal event）
5. 发送 rosbag SIGINT，等待正常退出；超时才使用受控 SIGTERM/SIGKILL
6. session_state → Finalizing，进入异步质量检查：
   bag 可读性 → topic/message/频率检查 → sidecar 与 /hil/* 审计流比对 → 写 quality_report
7. 质量结果决定 Finalized(Healthy) / Failed(self_check)
```

重复 stop 必须幂等；正常 stop、watchdog stop、急停 stop 共用同一条后处理路径。调用方通过 `wait_finalized()` 获取最终状态，禁止在 `request_stop()` 返回后立刻同步读取质量结果。

## 11. Replay 发布规则

只有以下条件全部满足才允许 `publish_replay()`：

- `session_state in {Finalized(Healthy), Finalized(ManuallyAcceptedDegraded)}`；
- episode result 已确定；
- state/action/image 以 ROS 时间轴可匹配（§7 规则）；
- sidecar 与 bag 内 `/hil/transition_audit` 步数/时间戳比对通过；
- 没有 `record_error`、硬盘 hard-stop 或未清理 `.active`；
- transition 中存在 mask/segment 字段；
- 接管步 action 使用实测 state + mask 覆盖 + SafetyGate 后值；
- 接管切换 seam 按 `intervention_segment_step` 过滤；
- reward 与终止原因已写入每条 transition/episode summary。

急停（estop）episode 一律不进入 `accepted_replay/`，但**必须**移入 `quarantine/`（safety 数据集）保留，不得删除。

发布动作本身：

```text
staging/ → accepted_replay/<episode_id>/    # 同分区原子 rename
写 publish_manifest.json:
  replay_schema_version = hil-replay-v002
  replay_source_quality_report = <path + sha256>
  replay_source_session = <episode_id>
  published_at_wall_ns / published_at_ros_ns
SQLite: replay_export_status → Published（与 manifest 写入同一事务序）
```

这样后续 reward classifier、SAC、离线评估都能追溯到原始 session 和质量证据。

## 12. 恢复与故障注入验收

### 12.1 启动恢复流程

启动时 `recover_interrupted()` 必须：

1. 扫描 DB 中残留 `Recording/Stopping/Finalizing` 的 session；
2. 清理孤儿 rosbag 进程（按记录的 PID + 命令行双重确认）和 `.active` 文件；
3. 残留 bag 若可读则照常走 Finalizing 质量检查，不可读则 `Failed(record_error)`；
4. 残留 staging 数据一律移入 `quarantine/`，不留在 staging；
5. 全部处理完成之前，Gate 对新 session 保持 `Block`。

### 12.2 必测场景

| 场景 | 预期结果 |
|---|---|
| 正常 ACT + VR 接管 + B success | Finalized(Healthy)，replay 发布到 accepted_replay，result=success |
| ACT 推理运行中启动录制 | Gate 放行（ACT 是登记 producer，不算冲突） |
| B 双击 failure | result=failure，数据仍可 Healthy 并发布；不误判为 record error |
| B 长按 abort | result=abort，不产生 success reward |
| 左手双键急停 | 控制立即停止，post-roll 录到急停后最终关节状态，result=estop，数据入 quarantine（safety），不进 accepted_replay |
| 终止后立即断言最后帧 | post-roll 内能找到晚于 terminal event ROS 时间的 state/图像 |
| recorder 立即退出 | Failed(record_error)，staging 入 quarantine，不发布 |
| daemon/runner 被 kill | 下次启动按 §12.1 恢复；bag 内 /hil/* 审计流仍完整可读 |
| 核心 streaming topic 中断 | watchdog 提交 StopRequest，episode controller 停止；不阻塞在监控线程 |
| /tf_static 无新消息 | 不触发 stale 告警（latched 模式） |
| 纯 ACT profile 无 /kuavo_arm_traj | Gate 不报缺失（profile 过滤） |
| bag 不增长 | StopRequest(record_error)，保存 watchdog 证据 |
| 磁盘达到 hard-stop | 停止录制/推理，episode 入 quarantine |
| 重复 STOP / request_stop 并发 | 幂等，不重复后处理、不覆盖已有质量结果 |
| stop 后立刻 publish_replay | 被拒绝（未 Finalized），wait_finalized 后才允许 |
| sidecar 与 bag 审计流步数不一致 | Failed(self_check)，不发布 |
| 新 episode 紧接旧 episode | 新 session 不继承旧 PID、topic、producer 或结果字段 |

### 12.3 验收证据

每个用例至少保存：

```text
session.json（快照）
gate.json
watchdog.report.json
watchdog.events.jsonl
record.stdout.log / record.stderr.log
SQLite session row + hil_events 状态迁移记录
bag quality report（含 sidecar/bag 比对结果）
publish/quarantine 决定与 manifest
```

## 13. 实施阶段

### P0：文档和 schema

- 固定 `hil-replay-v002` 字段与 `publish_manifest.json` 格式；
- 创建 SQLite schema（含 PRAGMA、schema_version、唯一约束）和状态常量、白名单迁移表；
- 实现 `timebase.py` 与 `TimeStamps` 结构；
- 固定分级 topic profile（hil-v002）；
- 不启动 rosbag，不碰真机控制。

### P1：本地 session + gate

- 实现 create/register_producer/start/request_stop/wait_finalized/cancel；
- `HILReplayWriter` 输出根目录改为 session staging；
- 录前检查 ROS、streaming/latched 分级 topic、磁盘和目录；互斥检查按 §9.1（放行登记 producer）；
- 用假的 recorder process 做单元测试，覆盖双状态机全部合法/非法迁移。

### P2：真实 rosbag + watchdog + 审计 topic

- 接 `rosbag record` 子进程；保存 PID、命令、stdout/stderr；
- 实现 `/hil/transition_audit`、`/hil/result_event` 发布并纳入录制；
- watchdog：bag growth、topic freshness（仅 streaming）、限额和硬盘检查；StopRequest 队列机制；
- 500 ms 循环只读缓存，验证无重型探测；
- 先在 MuJoCo/ROS 仿真验证。

### P3：质量检查 + 原子发布

- Finalizing 异步线程：bag 可读性、topic/message/频率检查、sidecar 与 bag 审计流比对；
- 输出结构化质量报告；
- 实现 staging → accepted_replay 原子发布与 quarantine 流转；
- 质量不合格自动 quarantine；验证 raw VR 与 replay action 不混淆；
- 实现停止序列 post-roll 与"最终 state 晚于 terminal event"断言。

### P4：真机 HIL 验收

- 先短 episode、低风险动作；
- 验证 VR 接管、B 结果、急停、post-roll 和录制状态同步；
- 逐项执行 §12.2 故障注入；
- 通过后再开始 R0 reward 数据采集。

## 14. 迁移完成标准

- [ ] `HILRecordingSession` 能创建并持久化 episode，双状态机迁移全部走白名单事务；
- [ ] Gate 能阻止第二 recorder、并发 session、外部控制节点和未恢复 session，且放行登记的 ACT runner；
- [ ] 所有事件/transition 携带 ros/mono/wall 三时间戳，对齐一律用 ROS 时间；
- [ ] rosbag recorder 的 PID/命令/日志可追踪，bag 内含 `/hil/transition_audit` 与 `/hil/result_event`；
- [ ] watchdog 能发现 recorder 退出、bag 不增长、streaming topic stale 和磁盘风险，且只通过 StopRequest 停止；
- [ ] 停止序列含 post-roll，能录到 terminal event 之后的最终 state；
- [ ] `request_stop` / `wait_finalized` / `publish_replay` 三段式无竞态，重复调用幂等；
- [ ] session/DB/文件状态在 kill/restart 后可恢复，残留 staging 进 quarantine；
- [ ] B success/failure/abort、急停、SafetyGate fault 可区分；急停数据保留在 quarantine（safety）；
- [ ] raw VR action、replay action、policy action 三者可追溯，sidecar 与 bag 审计流可比对；
- [ ] 只有 `accepted_replay/` 中的 episode 才能进入 HIL replay/SAC；
- [ ] MuJoCo 和真机短 episode 验收全部通过；
- [ ] 迁移过程中未修改 `third_party/kuavo_brain/kuavobrain-v3.x`。

## 15. 第一批实现文件

建议实际编码顺序：

1. `kuavo_rl/hil_recording/models.py`：状态、事件、`TimeStamps`、报告 dataclass；
2. `kuavo_rl/hil_recording/timebase.py`：三时钟采样与对齐规则；
3. `kuavo_rl/hil_recording/database.py`：SQLite 初始化（PRAGMA/WAL/版本）、白名单状态迁移事务和恢复查询；
4. `kuavo_rl/hil_recording/session.py`：session 生命周期（create/register_producer/start/request_stop/wait_finalized）；
5. `kuavo_rl/hil_recording/gate.py`：ROS/分级 topic/disk gate（互斥不含 ACT）；
6. `kuavo_rl/hil_recording/rosbag_recorder.py`：子进程和 `.active` 处理；
7. `kuavo_rl/hil_recording/audit_publisher.py`：`/hil/*` 审计 topic；
8. `kuavo_rl/hil_recording/watchdog.py`：监控循环 + StopRequest 队列；
9. `kuavo_rl/hil_recording/quality.py`：bag 检查 + sidecar 比对；
10. `kuavo_rl/hil_recording/publish_replay.py`：staging → accepted_replay 原子发布 / quarantine；
11. `scripts/rl/eval_act_execute_first.py`：由 session 统一包住 episode（三段式停止）。

第一步不应该是修改 reward 或 SAC。先保证每条数据都能回答：这条数据从哪来、是否完整、是否安全停止、结果是什么、为什么允许进入 replay。
