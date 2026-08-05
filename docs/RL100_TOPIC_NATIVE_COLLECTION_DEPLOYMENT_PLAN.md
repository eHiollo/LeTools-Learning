# RL-100 Topic-Native 采集与实机部署执行方案

本文是 RL-100 topic-native 采集与部署的执行规格，当前仓库代码已按此规格落地。目标是让训练数据中的 state/action 与真实 ROS topic
逐字段、逐顺序、逐单位对应，使策略输出可以经过安全检查后直接发布到原控制话题。

本方案替代当前 `state16/action16`、Qiangnao 1-DoF 压缩以及
`action[t] = state[t+1]` 的临时方案。实机 ROS 话题、控制模式和现场限位仍需按文档中的验收步骤确认。

实机采集的执行链路采用 Brain 风格原始 rosbag 模式：采集期间只异步记录相机、状态、command、
`/tf` 和 `/tf_static` 原始消息；episode 结束后再按参考相机时间戳对齐到 YAML 的 10 Hz，最后生成
点云和 topic-native NPZ/Zarr。文档后续关于 cache、state/action 契约和质量审计的规则仍然适用；
`online_pointcloud` 仅作为显式回退模式保留。

## 1. 已锁定的最终契约

### 1.1 Observation

```text
point_cloud: (1024, 3), float32, base_link

agent_pos/state: (32,), float32
  [0:20]  = /sensors_data_raw.joint_data.joint_q
  [20:32] = /dexhand/state.position
```

要求：

- `/sensors_data_raw.joint_data.joint_q` 当前实机必须为 20 维，保持原 topic 顺序，单位 rad。
- `/dexhand/state.position` 必须为 12 维，保持原 topic 顺序和原始数值尺度。
- 下位机源码中该 topic 的固定 name/order 为
  `[l_thumb,l_thumb_aux,l_index,l_middle,l_ring,l_pinky,r_thumb,r_thumb_aux,r_index,r_middle,r_ring,r_pinky]`，
  position 反馈为手部原始 `0..100` 位置量。采集 preflight 必须同时核对 12 个 name、维度和
  实机 min/max，不能只看 shape。
- 不再提取 `[4:18]` 写入训练 state，不再把 Qiangnao 压成左右两个标量。
- 不再用手部 command 覆盖 `/dexhand/state`。state 只表示原始状态 topic。
- 若实机 raw joint dim 不是 20，preflight 直接失败；不自动换 slice 或静默改变 shape。

### 1.2 Action

```text
action: (26,), float32
  [0:14]  = /kuavo_arm_traj.position
  [14:20] = /control_robot_hand_position.left_hand_position
  [20:26] = /control_robot_hand_position.right_hand_position
```

单位与数值范围：

- `action[0:14]`：degree，与 `/kuavo_arm_traj.position` 完全一致。
- `action[14:26]`：0..100，与 `robotHandPosition` 左右 `uint8[6]` 完全一致。
- 不归一化后写盘。RL-100 normalizer 在训练时分别拟合各维。
- 不再输出 16 维 action，不再复制一个标量扩成 6 个手指值。
- 不再使用 `next_state` 作为 action。

### 1.3 Next fields

zarr 继续保留 RL-100 所需字段：

```text
state[t]       = 原始状态快照，32维
action[t]      = t时刻有效命令缓存，26维
next_state[t]  = state[t+1]
next_action[t] = action[t+1]
```

episode 最后一帧：

```text
next_state[-1]  = state[-1]
next_action[-1] = action[-1]
```

## 2. 异步控制话题的最终处理规则

`/kuavo_arm_traj` 与 `/control_robot_hand_position` 是事件驱动话题，不要求每个采样周期都有新消息。
采集器必须使用两个独立、线程安全的 command cache，并执行零阶保持。

### 2.1 Episode 初始化

操作者发出“开始录制”时：

```text
arm_cache = 当前实测双臂14维 rad -> degree
hand_cache = [0, 99, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0]

arm_seen_after_record_start = false
hand_seen_after_record_start = false
recording_samples_enabled = false
record_generation += 1
record_start_received_at = monotonic_now()
```

`arm_cache` 的 rad->degree 只用于“手部命令先到、手臂命令尚未到”的临时 hold 初始化。
一旦收到 `/kuavo_arm_traj`，后续直接保存 topic 的 degree，不再转换。

订阅器可能在 RESET/等待确认期间已经收到旧命令。因此 `*_seen_after_record_start` 只能由
`received_at >= record_start_received_at` 且 generation 匹配的合法消息置 true；录制开始前收到的
历史消息不得触发 episode，也不得覆盖本 episode 的初始化 cache。

### 2.2 开始写样本的条件

收到以下任意一个合法消息后开始写 episode：

```text
/kuavo_arm_traj
或
/control_robot_hand_position
```

具体行为：

- 手臂通常先动：第一条 arm command 到达即开始写；手部使用默认 12 维值，直到首条 hand command。
- 如果手部先动：第一条 hand command 到达即开始写；手臂使用初始化时实测姿态的 degree hold，直到首条 arm command。
- 两个话题都没有出现时，不写 state-only 样本。
- episode 结束时仍未收到另一个话题不构成错误，因为缺失侧已有明确初始化值。
- 因为默认 hand action 在第一条 hand topic 前是人为定义的 hold，而不是本 episode 已收到的消息，
  开录 preflight 必须检查 `/dexhand/state.position` 与该默认值在配置容差内。若不一致，提示操作者
  先把手恢复到默认姿态；不得静默把默认值换成反馈值，也不得带着明显不一致继续采集。

### 2.3 运行期间

```text
arm callback:
  验证 position 为14维、有限值
  arm_cache = position原值
  arm_header_stamp = header.stamp
  arm_received_at = 单调时钟接收时间
  arm_changed = true

hand callback:
  验证 left/right各6维、0..100
  hand_cache = left6 + right6
  hand_header_stamp = header.stamp
  hand_received_at = 单调时钟接收时间
  hand_changed = true

每个10 Hz数据帧:
  state32 = latest raw_joint_q20 + latest dexhand_state12
  action26 = arm_cache14 + hand_cache12
  保存点云、state、action和对齐元数据
  清除本帧changed标志
```

采样必须定义一个线性化点，避免“未来命令”写到较早样本：

- 四个状态/命令 callback 与 sampler 最好由一个 `TopicNativeSampleCoordinator` 共用锁；若仍拆成
  state hub 和 command cache，至少让每个 snapshot 原子复制，并以 `sample_cutoff_received_at` 为准。
- 一个样本只能使用在该 cutoff 前已经完成 callback 的 cache；ROS `header.stamp` 只用于诊断对齐，
  不能单独作为因果判定依据（主机时钟偏差可能导致倒置）。
- snapshot 与清除 `*_changed` 必须在同一次加锁中完成，不能先读再单独清标志，否则会丢更新。

首次收到合法消息后，command cache 永不过期：

- 不使用 `0.2s` timeout 判断 command 无效。
- 话题停止更新表示控制器继续保持最后目标。
- 不因“本帧无新 action 消息”而丢帧或丢 episode。
- 格式错误、NaN、维度错误的消息不得覆盖最后合法 cache。

### 2.4 对齐元数据

每帧至少记录到 staging NPZ metadata/辅助数组：

```text
sample_stamp
sample_cutoff_received_at
joint_state_stamp
dexhand_state_stamp
arm_command_stamp
hand_command_stamp
joint_state_received_at
dexhand_state_received_at
arm_command_received_at
hand_command_received_at
arm_command_changed
hand_command_changed
arm_command_seen
hand_command_seen
arm_command_source       # topic | measured_hold
hand_command_source      # topic | default_hold
```

这些是长度为 `T` 的逐帧数组，不能塞进单个 episode `meta_json`。必须扩展
`save_episode_npz()` / `load_episode_npz()`，在 NPZ 中保存真实数组；`meta_json` 只保存 episode 级
字符串和配置。建议同时把数值时间戳、age、changed/seen 写到 zarr 的额外 `meta/*` 数组，RL-100
读取器可忽略它们，但最终 zarr 自身仍可审计。若第一版不写 zarr 辅助数组，build 后的 inspect
必须显式关联原 staging 文件，不能声称仅凭 zarr 已完成时间对齐检查。

## 3. 采集代码改造

### 3.1 `kuavo_rl/ros_teleop.py`

删除/停用 Qiangnao 标量压缩逻辑：

- `qiangnao_scalar_index`
- `_latest_hand` 的 2 维表示
- `last_gripper_command() -> tuple[float, float]`
- action16 拼装逻辑

新增 topic-native command cache，建议单独封装：

```python
@dataclass(frozen=True)
class TopicCommandSnapshot:
    arm14_deg: np.ndarray
    hand12_raw: np.ndarray
    arm_header_stamp_s: float
    hand_header_stamp_s: float
    arm_received_at_s: float
    hand_received_at_s: float
    arm_changed: bool
    hand_changed: bool
    arm_seen: bool
    hand_seen: bool
    arm_source: str
    hand_source: str

class TopicCommandCache:
    def reset(self, measured_arm14_rad, *, generation, record_start_received_at): ...
    def update_arm(self, joint_state_msg): ...
    def update_hand(self, robot_hand_position_msg): ...
    def has_any_topic_command(self) -> bool: ...
    def snapshot_and_clear_changed(self, sample_cutoff_received_at) -> TopicCommandSnapshot: ...
```

所有 cache 读写使用同一把 lock，snapshot 必须原子复制数组和时间戳。

### 3.2 `kuavo_rl/rl100_zarr/live_collect.py`

重写 `_record_one_episode()` 的样本来源：

- 删除 `action = next_state`。
- 删除 `state[7]`、`state[15]` 手部 command 覆盖。
- 不再通过现有 action16 `env.step` info 取训练 action。
- 从 timestamp-aware 原始状态读取器取得 raw joint20 和 dexhand12。
- 从 `TopicCommandCache.snapshot_and_clear_changed()` 取得 action26。
- 在收到任意首条 topic command 前可以维持控制/标签轮询，但不 append 样本。
- 第一条 topic command 到达后，从当帧开始 append。
- `env.step()` 若仍用于 VR 控制和标签，只承担控制/事件作用，不再定义训练 action。

建议将采样拆成纯函数，便于无 ROS 测试：

```python
def compose_topic_native_state(raw_joint_q, dexhand_position) -> np.ndarray: ...
def compose_topic_native_action(arm14_deg, hand12_raw) -> np.ndarray: ...
```

### 3.3 原始状态读取

当前 Kuavo gym observation 会切 arm 并重排夹爪，不适合作为 state32 数据源。实现者应增加专用
RL-100 topic-native state hub，直接订阅：

```text
/sensors_data_raw  -> joint_data.joint_q完整20维
/dexhand/state     -> position完整12维
```

建议文件：

```text
kuavo_rl/rl100_zarr/ros_state.py
```

接口：

```python
@dataclass(frozen=True)
class TopicStateSample:
    state32: np.ndarray
    joint_stamp_s: float
    hand_stamp_s: float
    joint_received_at_s: float
    hand_received_at_s: float
    joint_hand_skew_s: float
    joint_age_s: float
    hand_age_s: float
    joint_sensor_time_s: float

class TopicStateHub:
    def start(self): ...
    def preflight(self, timeout_s): ...
    def snapshot(self, joint_max_age_s, hand_max_age_s, max_skew_s) -> TopicStateSample: ...
```

state topic 是观测流，仍需 freshness/skew 检查；command cache 则不因静默而过期，两者语义必须区分。
`sensorsData` 同时含 `header.stamp` 和 `sensor_time`：以有效的 `header.stamp` 作为 ROS 状态时间，
另存 `sensor_time` 和本机单调接收时间，不得把三者混成一个字段。

`joint_max_age_s` 和 `hand_max_age_s` 必须分开配置。实现前先在实机连续统计至少 60 秒两个状态
topic 的频率、间隔 p50/p95/p99 和 name/range，再据此确定阈值；不得因为 `/dexhand/state` 暂时慢
就改用 hand command 伪造 state。超过阈值的帧跳过并计数，连续超限则终止本 episode。

## 4. Schema 与数据质量门

### 4.1 修改常量

涉及：

```text
kuavo_rl/contracts.py
kuavo_rl/rl100_zarr/schema.py
kuavo_rl/rl100_zarr/config.py
configs/rl/rl100_zarr_collect_upper_cams.yaml
```

新常量建议：

```python
RL100_TOPIC_STATE_DIM = 32
RL100_TOPIC_ACTION_DIM = 26
RAW_JOINT_DIM = 20
DEXHAND_STATE_DIM = 12
ARM_COMMAND_DIM = 14
HAND_COMMAND_DIM = 12
```

不要直接修改项目中其他 ACT/HIL 仍依赖的通用 `STATE_DIM=16`、`ACTION_DIM=16`；新增 RL-100
topic-native 专用契约，避免破坏其他训练链。

### 4.2 质量检查

每个 episode 保存前验证：

- state shape `(T,32)`，action shape `(T,26)`。
- 全部值有限。
- raw joint dim 始终为20，dexhand dim 始终为12。
- arm action 为14维，hand action 为12维。
- hand action 全部位于 `[0,100]`。
- 首个样本之前确实收到过 arm/hand 任一 topic command。
- action 不得由 state/next_state 整体复制得到。
- command `received_at` 必须单调不倒退；允许重复 cache stamp，表示 hold-last。有效 header stamp
  若明显倒退则质量门失败；无效/零 header 必须单独计数，不能伪装成本机接收时间。
- 任一逐帧 command `received_at` 不得晚于对应 `sample_cutoff_received_at`。
- 报告 arm/hand changed frame 比例和每维 min/max/range。
- 默认 hand hold 值允许长期不变；只有任务要求抓取成功时，再要求至少一个 hand 维度发生有效变化。

inspect 输出必须明确展示：

```text
state_joint20 min/max
state_hand12 min/max/range
action_arm14_deg min/max/range
action_hand12 min/max/range
arm_command_changed_ratio
hand_command_changed_ratio
initial command source
state age/skew p50/p95/p99/max
command hold duration p50/p95/p99/max
causality violation count (必须为0)
```

旧 `state16/action16` zarr 与新 schema 不兼容，禁止混合 build 或继续训练同一 checkpoint。

### 4.3 Reward 的混合单位问题

当前 `assign_episode_rewards()` 直接计算
`norm(action[t]-action[t-1])`。新 action26 同时包含 degree 和手部 `0..100`，直接使用默认
`smooth_penalty: 0.01` 会让奖励尺度被手指跳变或单位选择主导。

第一版新契约必须在采集/构建 YAML 中设置：

```yaml
smooth_penalty: 0.0
```

若以后确实需要平滑惩罚，只能先分别按配置的 arm14 degree scale 与 hand12 raw scale 做无量纲化，
再用分组权重计算，并用独立测试固定奖励范围。不得沿用现有原始 action L2 norm。

### 4.4 Writer 与 attrs

`writer.py` 当前硬编码 `kuavo_contract: v62-16d-r1`，必须移除该默认值或让调用方强制传入，
不能先写旧值再被 attrs 偶然覆盖。新 zarr 至少写入：

```text
contract = rl100_topic_native_v1
state_dim = 32
action_dim = 26
state_order / action_order
state_units / action_units
dexhand_joint_names
arm_joint_names
hand_default
smooth_penalty
collection_config_sha256
```

build 时缺少这些 attrs 或 contract 不一致应直接失败。

## 5. RL-100 训练配置

新 task YAML 的 shape_meta：

```yaml
shape_meta:
  obs:
    point_cloud:
      shape: [1024, 3]
      type: point_cloud
    agent_pos:
      shape: [32]
      type: low_dim
  action:
    shape: [26]
```

要求：

- 数据集读取器不能假设 state/action 同维。
- normalizer 分别拟合 `agent_pos32` 和 `action26`。
- horizon、`n_obs_steps`、`n_action_steps` 仍从训练 YAML 配置，不在部署端写死。
- 新模型输出目录、task 名和 zarr 名使用新版本号，例如 `grasp_8_4_topic_v1`，防止误载旧模型。

训练前至少抽取多个 episode，人工核对 topic action 曲线与 VR 操作时间一致。

## 6. 实机部署改造

当前部署代码锁定 state16/action16，并通过 Kuavo gym/SDK 把两个手部标量扩展到12维。该路径必须
针对 RL-100 topic-native 模型重写；其他策略可继续保留原路径。

### 6.1 Observation

部署端复用采集端同一个 `TopicStateHub` 和点云配置：

```text
agent_pos32 = raw_joint_q20 + dexhand_state12
point_cloud = 与采集完全相同的三相机融合结果
```

checkpoint preflight 改为验证：

```text
point_cloud [1024,3]
agent_pos [32]
action [26]
```

### 6.2 Policy output 到 ROS

新增专用 publisher，不经过 action16/Qiangnao 1-DoF bridge：

```python
class RL100TopicCommandPublisher:
    def publish(self, action26):
        arm14_deg = action26[:14]
        left6 = action26[14:20]
        right6 = action26[20:26]

        # safety通过后
        publish JointState(
            name=[arm_joint_0, ..., arm_joint_13],
            position=arm14_deg,
        ) to /kuavo_arm_traj
        publish robotHandPosition(left6, right6) to /control_robot_hand_position
```

发布前处理：

- 模型原始 action 保留 float32 用于日志。
- arm14 保持 degree 发布。
- `JointState.name` 必须严格填 14 项 `arm_joint_0..arm_joint_13`。下位机控制器会检查 name 的
  长度；仅填写 position 的消息会被拒绝或在不同控制模式下不产生有效控制。
- `velocity`、`effort` 第一版留空；不要用未经训练的值填充。
- hand12 先 clip 到 `[0,100]`，再按明确规则 `round` 并转换为 `uint8`。
- 两个消息使用同一控制 tick 的 ROS timestamp。
- `execute_steps`、控制频率和 action horizon 全部来自部署 YAML。当前 topic-native 部署按训练的
  10 Hz 消费 action，每次推理保留一个小 chunk（当前 checkpoint 为4步），并在队列低水位时由后台
  线程补充下一块；模型推理不再阻塞每个控制 tick。结果返回时按实测推理延迟丢弃已经过去的 action，
  队列耗尽只允许有限次实测 hold，随后进入 FAULT。下游 Kuavo 控制器负责其原生高频轨迹跟踪，
  RL100 不再额外制造一套 ROS 100 Hz 插值器。

### 6.3 Safety

安全检查不能直接把 degree 当 rad：

```text
模型输出 arm14_deg
    -> 转为 rad，仅用于位置限制、相对实测状态跳变和每步速度检查
    -> 安全门输出 limited_arm14_rad
    -> 转回 limited_arm14_deg 后发布
```

严禁“rad 中做了限幅，但最后又发布原始未限幅 degree”。日志同时保存 raw action、limited action
和每维 clip mask。

需要两套明确的限幅：

- arm：现场批准的14维 rad上下限、相对 measured arm14 rad 的首步跳变、相邻命令跳变。
- hand：12维 `[0,100]` 上下限、相邻命令最大变化量。

任何 shape、NaN、state stale、point cloud stale、action buffer 长时间耗尽、连续限幅或 publisher
异常都进入 FAULT，并停止产生新命令；不得回退到 action16 bridge。单次慢推理只会消耗缓冲区，
不会因为超过单个控制周期而立即误报故障。

FAULT 处理必须显式配置，不能发布全零 action：若 arm state 仍新鲜且控制链正常，只允许发布一次
“当前实测 arm hold + 最后安全 hand hold”，随后停更并等待人工复位；若状态已 stale、通信异常或
急停触发，则不再合成 hold，直接停止策略 publisher 并走现场急停/下位机安全流程。

live preflight 还必须确认：

- 两个 publisher 都有预期 subscriber，topic type/md5 匹配。
- 当前下位机控制模式确实接受 `/kuavo_arm_traj`；源码在不同 RL/MPC 模式走不同 callback，不能
  只以“publish 成功”判断执行成功。
- 在正式策略前，用“当前实测姿态 hold”做低风险单 tick，比较发布 echo 与随后实测关节反馈，
  确认 name/order/degree 链路正确。
- 同一时刻只能有一个手臂和手部命令源；VR 节点、旧 bridge 与策略 publisher 不得抢占 topic。

### 6.4 部署文件

重点修改/新增：

```text
kuavo_rl/rl100_policy.py
  由硬编码16/16改为严格检查32/26

kuavo_rl/rl100_real_runner.py
  接收state32/action26；安全门改为arm14 degree + hand12 raw

kuavo_rl/rl100_deploy_config.py
  新增topic、hand12限制和raw state契约

scripts/rl/run_rl100_real.py
  使用TopicStateHub和RL100TopicCommandPublisher

configs/rl/rl100_real_deploy.yaml
  明确state32/action26、两个发布topic和现场安全参数
```

原 `ROSBackend -> KuavoGymBridge -> exec_action16()` 不再用于该 RL-100 模型，但不要删除，避免
影响其他 HIL/ACT 功能。

## 7. 配置建议

采集 YAML 新增：

```yaml
contract: rl100_topic_native_v1
state_dim: 32
action_dim: 26
raw_joint_dim: 20
dexhand_state_dim: 12
arm_command_dim: 14
hand_command_dim: 12
arm_command_topic: /kuavo_arm_traj
hand_command_topic: /control_robot_hand_position
hand_default: [0, 99, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0]
start_on_any_command: true
command_hold_last: true
command_timeout_s: null
joint_state_max_age_s: <根据60秒实测填写>
dexhand_state_max_age_s: <根据60秒实测填写>
state_max_skew_s: <根据60秒实测填写>
smooth_penalty: 0.0
```

部署 YAML 新增：

```yaml
contract: rl100_topic_native_v1
observation:
  raw_joint_topic: /sensors_data_raw
  dexhand_state_topic: /dexhand/state
  expected_raw_joint_dim: 20
  expected_dexhand_dim: 12
publish:
  arm_topic: /kuavo_arm_traj
  hand_topic: /control_robot_hand_position
  arm_unit: degree
  hand_range: [0, 100]
  hand_rounding: nearest
  arm_joint_names: [arm_joint_0, arm_joint_1, arm_joint_2, arm_joint_3,
                    arm_joint_4, arm_joint_5, arm_joint_6, arm_joint_7,
                    arm_joint_8, arm_joint_9, arm_joint_10, arm_joint_11,
                    arm_joint_12, arm_joint_13]
inference:
  control_hz: 10
  execute_steps: 4
  action_buffer_size: 8
  action_low_watermark: 2
  timeout_s: 0.50
safety:
  fault_hold_once_if_state_fresh: true
```

训练 zarr attrs、checkpoint manifest、部署 run manifest 都必须写入同一个
`contract: rl100_topic_native_v1`。不匹配时直接拒绝。

## 8. 测试计划

### 8.1 纯单元测试

- state20+12 正确组成32维，顺序不变。
- arm14+left6+right6 正确组成26维，顺序不变。
- episode reset 后 arm cache 等于 measured arm14 的 degree，hand cache 等于固定默认值。
- arm 首先到达：立即启用写样本，hand 保持默认值。
- hand 首先到达：立即启用写样本，arm 保持 measured hold。
- 两个 topic 同时/交错到达：cache 独立更新。
- 消息静默多秒：cache 不过期。
- 非法新消息不覆盖最后合法值。
- state-only 前缀不写入。
- 录制开始前的旧 topic 消息不会触发新 episode，也不会污染新 generation 的 cache。
- callback 与采样并发时，command received_at 永远不晚于 sample cutoff，changed 标志不丢失。
- action 不等于 next_state 构造结果。
- hand round/clip 后发布左右各6维。
- shadow 模式两个 publisher 调用次数均为0。
- live 每 tick 两个 publisher 各调用一次，且来自同一 action26。
- arm JointState 的 name/position 均为14维且顺序固定，velocity/effort为空。
- mixed-unit action 在 `smooth_penalty: 0.0` 下不会产生额外 step penalty。

### 8.2 离线数据验收

- zarr state `(T,32)`、action `(T,26)`。
- 随机抽帧与原始 rosbag/topic 消息逐维比较。
- arm action 数值与 `/kuavo_arm_traj.position` 一致。
- hand action 数值与 left/right arrays 一致。
- 首段手臂运动期间 hand action 是指定默认值。
- 首次手部控制后 hand cache 正确切换并保持。
- 至少画一条 episode 的 state/action/timestamp 曲线人工检查。

### 8.3 实机部署验收

1. checkpoint inspect 确认32/26和contract attr。
2. ROS preflight 确认raw joint20、dexhand12、三路点云。
3. 确认控制模式、唯一发布源、subscriber、JointState name/order 和 degree 链路。
4. shadow 500 tick，检查action26范围、延迟和publish=0。
5. live仅手臂：强制hand保持默认值，1 tick开始。
6. live仅手部：arm保持当前姿态，分别测试左右6维。
7. 联合发布10 tick，确认topic echo逐维等于安全处理后的模型输出。
8. 再逐步增加episode长度。

## 9. 推荐实现顺序

1. 新增专用32/26契约和纯 compose/validate 函数。
2. 实现 `TopicCommandCache` 与无ROS单测。
3. 实现 `TopicStateHub` 与时间戳测试。
4. 扩展 staging 逐帧数组格式、zarr 审计字段和 writer attrs。
5. 修改采集循环，删除next_state action和手部标量覆盖。
6. 修改schema/reward/inspect，生成少量新staging与zarr。
7. 人工核对数据曲线后再修改训练task YAML。
8. 修改checkpoint loader和离线回放到32/26。
9. 实现专用topic publisher与安全门，填全 JointState name。
10. 完成shadow和分阶段实机验收。

每一步都应保持已有 action16/HIL/ACT 测试通过；RL-100 topic-native 通过新增模块隔离，禁止为了
新数据契约全局替换通用16维常量。

## 10. Definition of Done

- 数据中不存在 `action=next_state` 或Qiangnao标量压缩。
- state逐维对应两个状态topic，action逐维对应两个控制topic。
- 任一控制topic首次到达即可开始写样本，另一侧使用已确认的初始化值。
- command静默时执行hold-last，不制造缺失action。
- 新zarr、训练YAML、checkpoint和部署manifest都声明同一个contract版本。
- 策略输出26维，经安全门后可逐字段发布到两个原控制topic。
- shadow、离线回放、单topic和双topic实机验收全部通过。
