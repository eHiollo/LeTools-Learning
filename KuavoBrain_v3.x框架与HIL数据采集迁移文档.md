# KuavoBrain v3.x 框架与 HIL 数据采集迁移文档

> 参考仓库：`third_party/kuavo_brain/kuavobrain-v3.x`  
> 参考基线：KuavoBrain v3.x / 文档声明版本 `3.0.0`  
> 目的：提炼公司内部遥操作数据采集框架，迁移到当前 HIL-SERL 项目；不引入云端上传、OTA 和内部平台依赖。

## 1. 结论

KuavoBrain v3.x 的核心价值不在某一个遥操作按键或控制算法，而在于把一次采集变成可审计、可恢复、可判定质量的数据会话：

```text
设备/任务元数据
        ↓
录前 Gate：ROS、必要 topic、时间、磁盘、互斥状态
        ↓
会话持久化：SQLite + session 状态 + record PID/命令/日志
        ↓
rosbag 录制：按 robot profile / eef 动态解析 topic
        ↓
录中 Watchdog：频率、丢失、过期、卡死、磁盘、时长/大小
        ↓
安全停止：正常 stop / 限额 stop / 磁盘 hard-stop / 崩溃恢复
        ↓
录后处理：拆包、bag 质量检查、状态落库
        ↓
HIL 适配：读取 bag + manifest/DB，生成 episode/replay
```

当前项目应迁移以下能力：

1. `topics.yaml` 的 topic group + robot profile 动态解析；
2. `RecordSessionManager` 的会话状态机、录制进程管理和幂等 stop；
3. 录前 gate、常驻 watchdog、录后 `BagQualityChecker`；
4. SQLite 的会话/文件/质量状态持久化；
5. 崩溃、断电、孤儿 `.active` 文件和不完整 bag 的恢复策略。

当前阶段不迁移：云平台 API、NAS/云上传、OTA、HTTP/WS 平台协议、内部推理服务和 Debian 发布体系。

## 2. 仓库总体结构

```text
kuavobrain-v3.x/
├── kuavobrain/                    # 主服务和核心库
│   ├── apps/
│   │   ├── kb_daemon/              # 主 daemon：编排、命令、后台服务
│   │   ├── kb_cli/                 # 本地诊断/录制/数据库 CLI
│   │   └── kb_ota_agent/           # OTA 独立代理
│   ├── libs/
│   │   ├── kb_core/                # Result、日志、配置、进程和文件工具
│   │   ├── kb_storage/             # StorageLayout、SQLite、数据库迁移
│   │   ├── kb_ros/                 # topic 配置、rosbag、相机/臂流
│   │   ├── kb_record/               # 录制 session 生命周期
│   │   ├── kb_record_monitor/       # gate、watchdog、bag 质量检查
│   │   ├── kb_daemon_service/       # 核心命令编排和状态互斥
│   │   ├── kb_robot_control/        # 头部/折叠臂等附加控制
│   │   ├── kb_upload/               # NAS/云上传
│   │   ├── kb_cloud/                # 平台 API
│   │   ├── kb_inference/             # 推理子进程与状态机
│   │   └── kb_gateway_*              # HTTP/WS 接口
│   ├── config/                      # device/topics/monitor 等配置
│   └── message/                    # kuavo_msgs 自定义消息/服务
├── time-sync/                       # Chrony/RTC/ROS 时间参考与 bag 检查
├── camera-special-features/         # Orbbec/RealSense 驱动和相机健康检查
├── multi-camera-h265-encoder/       # H.265 编码、水印、带宽优化
├── robot-perception-bringup/        # 感知链路启动编排
├── kuavo-audio-receiver/             # 音频采集
└── packaging/                       # Deb、离线安装、校验、回滚、OTA
```

这里的 `kb_daemon` 是主编排器，`kb_record` 是会话管理器，`kb_ros` 负责 ROS 录制，`kb_record_monitor` 负责“数据是否值得被接受”的判断，`kb_storage` 负责状态不丢失。

## 3. 依赖关系与职责

```text
kb_core
  ↓
kb_storage ← kb_cloud / kb_upload
  ↓
kb_ros
  ↓
kb_record_monitor
  ↓
kb_record
  ↓
kb_daemon_service
  ↓
kb_gateway_http / kb_gateway_ws / kb_cli
```

| 模块 | 当前仓库职责 | HIL 迁移结论 |
|---|---|---|
| `kb_core` | 统一错误返回、日志、配置、进程管理 | 保留最小 Python 等价层或直接复用现有 `kuavo_rl` 工具。 |
| `kb_storage` | 路径、SQLite、session/bag/upload 状态 | **必须迁移核心能力**。HIL 不能只依赖内存或 episode 结束后才写文件。 |
| `kb_ros` | topic 动态解析、rosbag 启停、camera/arm stream | **必须迁移 topic resolver 和 recorder**；相机驱动本身按当前项目保留。 |
| `kb_record` | 创建/开始/停止/取消/恢复会话 | **必须迁移**，作为 HIL recorder 的核心。 |
| `kb_record_monitor` | 录前 gate、频率 watchdog、bag 质量检查、磁盘/限额 | **必须迁移**，这是安全可靠性的关键。 |
| `kb_daemon_service` | 约 30 个命令、录制/推理/OTA 互斥 | 只保留本地 HIL orchestrator，不复制内部平台命令。 |
| `kb_cloud` / `kb_upload` | 平台、NAS、云端传输 | 本项目禁用；本地可保留 `ReadyToExport` 状态。 |
| `kb_inference` | 内部模型推理子进程 | 用当前 ACT/SAC runner 对接，不搬内部推理状态机。 |
| `kb_gateway_*` | HTTP/WS 远程控制与状态推送 | 初期可由本地 CLI/ROS 节点替代，后续需要 UI 再做薄适配。 |
| `time-sync` | 多设备时钟同步和诊断 | 真机 HIL 必须参考其校时/验证思想，不能只用 Python wall clock。 |

## 4. 标准录制生命周期

### 4.1 命令入口

仓库主要通过 daemon 命令驱动：

```text
CREATE_SESSION <task_id> <device_sn>
START_RECORD <task_id>
STOP_RECORD <task_id>
CANCEL_RECORD <task_id>
PREPARE_NEW_RECORDING <task_id>
```

`START_RECORD` 在 session 不存在时可以按请求参数自动创建，但必须有 `device_sn`。启动前还会检查推理是否活动、OTA 是否进行以及录制 gate 是否允许。

HIL 迁移后保留同样语义，但命令可简化为 Python API/CLI：

```text
create_episode → start_recording → teleop/ACT/HIL → stop_recording
                                      ↓
                         success/failure/abort/estop 结果事件
```

### 4.2 会话状态

```text
Preparing
   ↓ gate 通过
Recording
   ↓ 用户停止/限额停止/磁盘保护
Stopping
   ↓ rosbag 退出
Splitting（可选）
   ↓
Checking
   ↓
ReadyToUpload / ReadyToExport
```

异常分支：

```text
任意活动状态 --进程崩溃/断电/不可读--> Failed
任意状态 --用户取消--> Canceled
ReadyToExport --明确删除--> Deleted
```

所有状态使用稳定英文值落库；中文只在 UI/日志展示层翻译。HIL 新状态建议使用 `ReadyToReplay`，不要直接把“文件存在”当成“数据可训练”。

### 4.3 Stop 必须幂等

仓库的 `STOP_RECORD` 只处理当前处于 `Stopping` 的 bag；重复 stop 不会重复跑录后处理。这个语义要原样保留，避免 VR 急停、用户停止和 watchdog 同时触发时重复关闭、重复入库或覆盖质量结果。

## 5. Topic 解析与录制内容

`config/topics.yaml` 不是一张固定 topic 列表，而是：

```text
topic_groups
  + robot_profiles(robot_type)
  + supported_eef(eef_type)
  + camera/audio/body/latch 组合
  = resolved record topics
```

### 5.1 当前 Kuavo 采集重点

HIL 至少应覆盖：

| 类别 | 示例 topic | 用途 |
|---|---|---|
| 相机 | `/cam_h/*`、`/cam_l/*`、`/cam_r/*`、可选 `/cam_w/*` | 观测、回放、reward classifier |
| 机器人状态 | `/sensors_data_raw`、`/joint_cmd`、`/tf` | state、时间对齐、动作/状态审计 |
| 机械臂 | `/kuavo_arm_traj`、`/ik_fk_result/input_pos`、`/ik_fk_result/eef_pose` | VR/ACT action 和 IK 结果 |
| 末端 | `/leju_claw_*`、`/dexhand/state` 或 linker hand topics | gripper/hand 状态 |
| 标定 | `/kuavo/arm_zeros`、`/kuavo/offset`、`/tf_static` | 离线复现和坐标解释 |
| 人工事件 | HIL adapter 自定义事件 topic/sidecar | success/failure/abort/急停时间戳 |

接管 trigger/grip 是控制输入，不等于任务结果。B success/failure/abort 事件需要和 rosbag 使用同一时钟写入，或至少保存可校准的 ROS 时间戳。

### 5.2 原始数据与预览数据分离

仓库支持原始 bag 与预览 bag 分离：

- `preview_only_topics`：缩略图、`/sensors_data_raw` 等适合快速浏览的 topic；
- `original_only_topics`：H.265 原始流、深度流等用于高保真重建的 topic；
- 预览 bag 使用 `.c.bag` 后缀。

HIL 迁移建议保留这个双产物设计：训练/人工检查先读 preview，出现问题时回到 original；不因为 preview 可播放就认为原始数据完整。

## 6. 录前 Gate、录中 Watchdog、录后检查

### 6.1 录前 Gate

`RecordGateEvaluator` 根据 watchdog snapshot 和 `record_monitor.yaml` 判断能否开始。当前配置的关键约束包括：

- ROS master 和必要 publisher 存在；
- `/joint_cmd`、`/sensors_data_raw` 等核心 topic 已发布且达到最低频率；
- 相机/末端 topic 按当前 robot/eef profile 动态检查；
- 时间年份/时间同步状态有效；
- 磁盘未达到开始录制禁止阈值；
- 当前没有推理、OTA 或其他互斥活动；
- `self_check.skip=false` 时，录后 bag 检查仍然开启。

Gate 失败时应拒绝开始，并把缺失 topic、低频 topic、过期 topic 和磁盘原因写进 session，而不是开始录制后再猜数据是否有效。

### 6.2 录中 Watchdog

仓库采用多层监控：

```text
A. 运行时频率探针：短窗口观察 Hz
B. 录后 bag 质量检查：以实际 bag 消息为权威
C. ROS master publisher 探针：判断 topic 是否仍被发布
```

同时监控：

- recorder PID 是否仍存活；
- bag 文件是否持续增长；
- topic 是否丢失、低频、stale 或出现时间间隔过大；
- 磁盘使用率；
- 单 bag 时长和文件大小上限；
- watchdog 自身报告和事件日志是否可写。

注意：运行时频率探针不能替代 bag 质量检查；同进程 ROS 订阅去重等问题可能让探针看到的频率不等于实际写入 bag 的频率。

### 6.3 停止和磁盘保护

当前配置体现了三类保护：

```text
runtime_warn_percent = 90%       → 警告
hard_stop_percent = 95%          → 强制停止录制并停止推理
max_bag_size = 50 GB              → 停止当前 session
max_duration = 14400 s            → 停止当前 session
```

迁移到 HIL 时，磁盘 hard-stop 必须优先于“继续保存完整 episode”。如果无法保证落盘，episode 应标记为 `record_error`，不能标成普通 failure 后进入训练。

### 6.4 录后质量检查

`BagQualityChecker` 应输出结构化结果，而非只输出日志：

```json
{
  "status": "Healthy|Degraded|Failed|Skipped",
  "message": "...",
  "durationSec": 12,
  "totalMessages": 123456,
  "missingTopics": [],
  "lowRateTopics": [],
  "observedHz": {"/joint_cmd": 25.0}
}
```

HIL importer 只有在质量状态为 `Healthy`，或经过人工确认的 `Degraded` 时才允许进入训练 replay；`Failed`、录制异常和 incomplete `.active` 文件只进入审计/修复队列。

## 7. 持久化与恢复

### 7.1 当前仓库的持久化事实源

当前 v3 代码已将 session 级权威状态放到：

```text
<data_root>/rosbag.db
├── sessions       # 当前 session 权威状态
├── bag            # bag 文件及质量/上传状态
├── task_cache     # 任务和策略缓存
└── upload_queue   # 上传队列
```

`session.meta`/旧 `manifest.json` 是历史兼容概念，不能在迁移时误认为当前 v3 的唯一事实源。当前 HIL 可以保留一个面向数据集的 `episode.json` 导出，但它应由 SQLite/session 状态生成，不能反过来成为并发录制状态的唯一存储。

### 7.2 HIL 最小 session 字段

```text
episode_id / task_id
collector_id / recorder_id
robot_type / robot_version / lower_commit / eef_type
topics_version / resolved_topics
session_state / created_at / started_at / stopped_at
record_pid / record_command / stdout / stderr
session_dir / bags_dir / original_bag / preview_bag
gate_check_status + gate_check_message
watchdog_status + watchdog_report_path + watchdog_log_path
bag_check_status + bag_quality_report
result: success | failure | abort | estop | record_error | timeout
result_event_time_ros / result_event_source
intervention segments / masks / ACT policy version
```

### 7.3 崩溃恢复

daemon 启动时必须执行：

1. 扫描 `Recording`、`Stopping`、`Preparing` 的历史 session；
2. 规范化为 `Failed(record_error)`，而不是伪装成正常结束；
3. 处理或隔离残留 `.active` 文件；
4. 检查 recorder PID 是否仍存在并清理孤儿进程；
5. 对已有 bag 做可读性/质量检查；
6. 恢复数据库与文件系统状态的一致性。

这一步是 HIL 安全可靠性的必要条件：操作者按急停、机器断电或 ROS 崩溃后，下一条 episode 不能继承上一条的错误状态。

## 8. 迁移到当前 HIL-SERL 的目标架构

### 8.1 建议的本地模块

```text
kuavo_rl/hil_recording/
├── config.py                 # topic/session/monitor 配置
├── topics.py                 # robot + eef profile 解析
├── session.py                # episode 状态机与 SQLite
├── rosbag_recorder.py        # rosbag 启停、PID、stdout/stderr
├── gate.py                   # 录前 gate
├── watchdog.py               # 录中频率/增长/磁盘监控
├── quality.py                # 录后 bag 检查
├── recovery.py               # 启动恢复、孤儿进程/active 文件
├── result_events.py          # B reward 与急停事件时间戳
└── export_replay.py          # 质量合格后导出 HIL replay
```

第一阶段可以使用 Python + `rospy`/`rosbag` 实现；不需要把整个 C++ daemon 复制过来。关键是保留状态边界、持久化时机、失败分类和 gate 优先级。

### 8.2 与现有 VR/ACT runner 的连接点

```text
HIL episode controller
  ├── ACT policy action
  ├── RosTeleopAdapter：VR 接管、B result、左手双键急停
  ├── KuavoHILSerlEnv：SafetyGate 后 action / reward
  └── HilRecordingSession：同步录 bag + 事件 + replay 索引
```

录制器不改变控制决策，只观察并记录：

- ACT 原始 action；
- raw VR/IK audit action；
- intervention mask；
- 实测 state；
- SafetyGate 后 replay action；
- reward/result 事件；
- gate/watchdog/quality 状态。

这能保持此前已经修正的语义：raw VR 目标不能直接当训练 action，未接管手臂不能被陈旧 VR 数据覆盖。

## 9. 迁移顺序与验收

### M0：只读对照

- 固定本仓库参考 commit/版本；
- 记录 topics、状态常量、配置阈值和 DB 字段；
- 不修改 `third_party/kuavo_brain/kuavobrain-v3.x`。

### M1：本地 HIL recorder 骨架

- 实现 session SQLite 和状态迁移；
- 实现 topic profile resolver；
- 用当前 VR/ACT runner 触发 start/stop；
- 写入 `recordCommand`、PID、日志路径和 resolved topic 列表。

### M2：安全检查

- 录前 gate 阻止缺 topic/低频/磁盘不足/控制互斥；
- 录中 watchdog 检测 bag 不增长、recorder 退出和 topic stale；
- 录后质量检查生成结构化报告；
- 自动把不合格 episode 排除出 replay。

### M3：HIL 结果与 replay 对齐

- B 单击/双击/长按事件写入 ROS 时间和 session；
- 急停、SafetyGate、SDK 错误单独分类为安全/录制错误；
- 使用 segment/mask 过滤接管切换缝；
- 只导出质量合格且结果字段完整的 episode。

### M4：仿真和真机验收

```text
正常 ACT + VR 接管 → success
正常 ACT + VR 接管 → failure
VR 接管 → 左手双键急停
录制中 kill recorder/daemon
录制中断 ROS topic
磁盘接近 hard-stop
重复 STOP_RECORD
```

每个场景都要验证：控制安全、session 状态、bag 文件、质量报告、SQLite 状态、下一次启动恢复结果。

## 10. 当前不应直接照搬的部分

1. `manifest.json` 作为并发状态唯一来源：v3 当前权威是 SQLite session；HIL 只生成导出 manifest。
2. 云端 task/upload/OTA 状态：本地 HIL 不需要，但要保留 `ReadyToExport` 和失败重试边界。
3. 内部 HTTP/WS 命令协议：先用本地 API/CLI，避免引入内部鉴权和平台耦合。
4. 内部相机硬编码：使用现有 Kuavo 相机驱动，但迁移动态 topic profile 和健康检查。
5. 所有 topic 全量录制：HIL 应按训练需要分原始/预览/审计集合，并使用磁盘预算。

## 11. 文件级参考索引

| 参考文件 | 需要学习的内容 |
|---|---|
| `kuavobrain/README.md` | 模块分层、主服务启动顺序、配置和应用入口 |
| `kuavobrain/DATABASE.md` | SQLite 数据库、`sessions`/`bag` 字段和状态 |
| `kuavobrain/config/topics.yaml` | topic groups、robot profiles、bag split、频率阈值 |
| `kuavobrain/config/record_monitor.yaml` | Gate、watchdog、磁盘、录制上限和 bag quality 配置 |
| `kuavobrain/libs/kb_record/` | session 生命周期、停止后处理、恢复和状态持久化 |
| `kuavobrain/libs/kb_record_monitor/` | gate、watchdog、频率探针、bag 质量检查 |
| `kuavobrain/libs/kb_ros/` | rosbag 启停、topic 解析、相机/臂流 |
| `kuavobrain/libs/kb_storage/` | runtime 路径、SQLite 和 schema 扩展 |
| `kuavobrain/libs/kb_daemon_service/` | 命令互斥、start/stop 编排、磁盘 hard-stop |
| `time-sync/` | ROS 时间参考、录制时间检查和诊断 bag |

## 12. 当前结论

后续 HIL 数据录制应把 KuavoBrain v3.x 当作“采集可靠性参考实现”：迁移它的 session、gate、watchdog、quality、DB 和 recovery 语义，保留当前项目的 VR/ACT/SafetyGate/action-normalization 逻辑，去掉云端、OTA、平台网关和内部模型服务。

第一项具体实现应是 `kuavo_rl/hil_recording/session.py + gate.py + watchdog.py`，先接现有 `HILReplayWriter`，完成本地仿真 fault-injection 验收，再接真机。这样可以最大限度避免“控制成功但数据不可训练”或“数据看起来有但关键 topic 丢失”的问题。
