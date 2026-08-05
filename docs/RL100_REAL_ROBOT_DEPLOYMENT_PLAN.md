# RL-100 BC 实机部署代码方案（Kuavo 5W v62 + Qiangnao）

本文是给后续代码实现者的执行规格。当前阶段只部署离线训练完成的 RL-100 BC
策略，不做在线 RL、自动复位、奖励模型或边采边训。

## 1. 已确定的边界

- 训练数据必须使用新采集链生成的
  `data/rl100/grasp_8_4_v2/grasp_8_4_v2.zarr`。旧文件
  `third_party/RL-100/data/grasp_8_4.zarr` 的 action 等于 state，且夹爪不动，禁止用于训练或验收。
- 模型输入是融合深度点云和机器人状态，不使用 RGB：
  - `point_cloud`: `(n_obs_steps, 1024, 3)`, `float32`，坐标系为 `base_link`。
  - `agent_pos`: `(n_obs_steps, 16)`, `float32`。
- 状态和动作统一为 16 维绝对关节目标：
  `[左臂 7 rad, 左夹爪 0..1, 右臂 7 rad, 右夹爪 0..1]`。
- 实机 `/sensors_data_raw.joint_q` 当前为 20 维，双臂切片必须是 `[4:18]`。
  实现仍应根据 `ARM_SLICE_BY_RAW_DIM` 动态选择，不能把 20 或 28 写死。
- Qiangnao 状态来自 `/dexhand/state.position` 的 12 维 `0..100`；一自由度表示取
  左手下标 0、右手下标 6，再除以 100。
- Qiangnao 命令走当前 Kuavo SDK/环境已有映射。策略的左右标量会展开成左右各 6 维，
  其中当前约定第二个值固定为 100，最终裁剪并转换到 `0..100`。
- 相机外参暂不标定。训练和部署必须使用与采集时完全相同的零外参/TF、相机安装位置、
  topic、工作空间裁剪和点数。以后若改为标定外参，旧数据和旧模型不应直接继续使用。
- 第一版只支持 RL-100 workspace `.ckpt`。不要把 RL-100 checkpoint 接入现有 LeRobot
  推理加载器，也不要优先支持策略目录等其他不明确格式。
- 默认必须是 shadow 模式，只推理和记录，绝不发布。live 模式需要单独显式确认。

## 2. 总体数据流

```text
三路 depth + CameraInfo + TF
            |
            v
DepthPointCloudHub -> base_link 融合/裁剪/FPS 1024 点 ----+
                                                       |
/sensors_data_raw + /dexhand/state -> 16-D state ------+--> 时间对齐历史
                                                               |
                                                               v
                                                        RL100Policy
                                                               |
                                                     action chunk [N, 16]
                                                               |
                                                    仅取第 1 个动作
                                                               |
                                      实测状态跳变门 + SafetyGate + watchdog
                                                               |
                                      shadow: 只记录 / live: ROSBackend.publish
                                                               |
                              KuavoBaseRosEnv -> 双臂 SDK(rad) + Qiangnao(0..100)
```

部署节点不要自己直接向 `/kuavo_arm_traj` 发布弧度。采集时该 topic 的消息是角度，
实机执行应复用 `ROSBackend -> KuavoGymBridge -> KuavoBaseRosEnv.exec_action()`，最终由
`control_arm_joint_positions` 按弧度执行。

## 3. 建议新增和修改的文件

### 3.1 新增 `kuavo_rl/rl100_policy.py`

职责：只负责 checkpoint 检查、模型构建和张量推理，不依赖 ROS。

建议接口：

```python
@dataclass(frozen=True)
class RL100CheckpointInfo:
    checkpoint_path: Path
    checkpoint_sha256: str
    state_dict_key: str       # ema_model 或 model
    n_obs_steps: int
    n_action_steps: int
    horizon: int
    point_count: int
    point_dim: int
    state_dim: int
    action_dim: int
    scheduler_type: str
    use_cm: bool

class RL100Policy:
    @classmethod
    def from_checkpoint(cls, checkpoint_path, device, model_source="auto"): ...
    def warmup(self) -> dict: ...
    def predict(self, point_cloud_history, state_history) -> np.ndarray: ...
```

加载流程必须如下：

1. 用 `torch.load(..., pickle_module=dill, map_location="cpu")` 读取 `.ckpt`。
2. 验证顶层至少存在 `cfg` 和 `state_dicts`。
3. 使用 `hydra.utils.instantiate(payload["cfg"].policy)` 直接构建策略；不要构建完整
   `TrainDP3Workspace`，否则会额外创建 critic、optimizer 和训练状态。
4. `model_source=auto` 时，如果训练配置开启 EMA 且 `ema_model` 存在，优先加载
   `ema_model`，否则加载 `model`。显式要求 EMA 但 checkpoint 没有时应报错，不能静默切换。
5. 默认 strict 加载。仅当所有 key 都有 `module.` 前缀时允许统一去除此前缀，并在日志记录。
6. 调用 `.to(device).eval()`；归一化器已包含在策略 state dict 中，不要从数据集重新拟合。
7. 推理使用 `torch.inference_mode()`。
8. `predict_action()` 的 `use_cm` 参数默认值是 `False`，实现时必须从 checkpoint 的
   `cfg.policy.use_cm`/策略属性解析后显式传入，不能遗漏。`deterministic` 和
   `distill2mean` 也由部署 YAML 明确给出并做兼容性检查。

checkpoint 加载后立即验证：

- `shape_meta.obs.point_cloud.shape == [1024, 3]`
- `shape_meta.obs.agent_pos.shape == [16]`
- `shape_meta.action.shape == [16]`
- `n_obs_steps >= 1`、`n_action_steps >= 1`
- 模型输出 `action` 最后一维是 16，所有值有限。

`predict()` 接收无 batch 维的 NumPy 历史，内部变成：

```text
point_cloud: [1, n_obs_steps, 1024, 3]
agent_pos:   [1, n_obs_steps, 16]
```

返回 `[n_action_steps, 16]` 的未归一化绝对动作。不要在部署层再次做训练数据归一化。

### 3.2 新增 `kuavo_rl/rl100_real_runner.py`

职责：实现无 ROS 细节的部署状态机、历史缓存、调度、超时和安全决策。ROS 对象通过依赖
注入，便于单元测试。

建议主要对象：

```python
class DeployState(Enum):
    INIT = ...
    PREFLIGHT = ...
    READY = ...
    SHADOW = ...
    ARMED = ...
    RUNNING = ...
    HOLD = ...
    FAULT = ...
    STOPPED = ...

@dataclass(frozen=True)
class TimedRL100Observation:
    state16: np.ndarray
    point_cloud: np.ndarray
    state_stamp_s: float
    hand_stamp_s: float
    point_cloud_stamp_s: float
    received_at_s: float

class ObservationHistory: ...
class RL100RealRunner: ...
```

历史缓存长度只能取 checkpoint 的 `n_obs_steps`。启动时所有输入通过 preflight 后，允许用
第一帧重复填充历史，以免必须等待多个周期；日志要记录 `history_padded=true`。运行中不得用
旧帧填充丢失的新帧。

每个控制 tick 的顺序固定为：

1. 读取带原始 ROS 时间戳的 state、hand 和点云样本。
2. 检查维度、有限值、新鲜度和各来源时间差。
3. 更新历史并执行一次同步推理。
4. 检查推理耗时和整条 observation-to-command 延迟。
5. 只取 `predicted_chunk[0]`，即每次重新规划、只执行第一个动作。第一版禁止一次下发整个 chunk。
6. 先做“预测绝对动作相对当前实测 state16”的首步跳变限制，再进入现有 `SafetyGate` 做
   位置范围和相邻命令限幅。
7. shadow 模式只落日志；live 模式才构造 `PublishedCommand` 并调用 `ROSBackend.publish()`。

禁止异步推理后继续发布过期动作。如果单次推理超过控制周期，应降低部署频率或使用训练配置
兼容的 CM 推理；不能靠缓存旧动作掩盖超时。

### 3.3 修改 `kuavo_rl/rl100_zarr/ros_depth.py`

保留当前深度解码、TF、融合、工作空间裁剪和 FPS 算法，新增返回元数据的方法，例如：

```python
def get_point_cloud_sample(self) -> PointCloudSample:
    # points, fused_stamp_s, oldest_age_s, camera_stamps, max_camera_skew_s
```

要求：

- 每路 depth 必须保留消息 header stamp 和本机接收时间。
- `require_all_cameras=true` 时任何一路超时都不能生成 live 命令。
- 检查三路 depth 的最大时间差。
- TF 查询失败、点数不足或裁剪后为空都必须返回结构化故障，不能复用上一帧并伪装成新帧。
- 原有 `get_point_cloud()` 可以保留为兼容封装，采集行为不能被破坏。

### 3.4 修改 Kuavo 状态读取链

当前 `ROSBackend` 会用 `time.time()` 伪造当前 observation 时间，并在缺少字段时把 age/skew
设为 0，这不满足实机部署安全要求。应在 `ObsBuffer`/Kuavo env 中保留：

- `/sensors_data_raw` 的 header stamp 与接收时间；
- `/dexhand/state` 的 header stamp 与接收时间；
- 实际 `raw_joint_dim`；
- state-hand skew 和最大 age。

这些字段应一路透传到 `BackendObservation`。部署 runner 在字段缺失时应 preflight 失败，不能
把缺失解释为零延迟。普通采集路径若依赖旧行为，可以新增专用 timestamp-aware 方法以保持兼容。

### 3.5 修改 `kuavo_rl/config.py` 和 `kuavo_rl/safety.py`

- 修复 `build_env_config_from_dict()`，把 YAML 中的 `max_cross_topic_skew_s`、
  `require_deadman_for_teleop` 等已有字段完整传入。
- `SafetyGate` 增加可测试的 measured-state jump 检查，或在 runner 前增加独立
  `MeasuredStateGate`。两者都要区分手臂 rad 和夹爪归一化尺度。
- fault 后不能持续把 `_hold()` 当正常预测无限发布。推荐只发送一次“当前实测姿态 hold”，随后
  停止发布并进入 `FAULT`；若 SDK 已有更可靠的停止接口，则优先调用该接口。
- SDK 抛异常立即进入 `FAULT`。

### 3.6 新增 CLI 和启动脚本

- `scripts/rl/run_rl100_real.py`：Python CLI，只做参数解析和对象组装。
- `scripts/rl/run_rl100_real.sh`：source ROS/工作空间、设置 RL-100 `PYTHONPATH`，然后调用 CLI。
- `configs/rl/rl100_real_deploy.yaml`：唯一部署配置。

CLI 建议使用子命令，避免一个 `--live` 混合所有职责：

```text
inspect-checkpoint  读取 cfg、模型 key、hash，并做随机输入 warmup
offline-replay      读取 zarr，不连接 ROS、不发布
ros-preflight       检查 topic、TF、维度、时间戳、点云和控制端连接，不推理、不发布
shadow              实时观测和推理，只记录
live                经过确认后有限步数发布
```

## 4. 部署 YAML 规格

建议配置结构如下。数组中的关节上下限只是占位符，代码实现前必须替换为现场确认值；禁止沿用
当前 `[-3.14, 3.14]` 作为正式实机限制。

```yaml
checkpoint:
  path: outputs/rl100/grasp_8_4_v2/checkpoints/latest.ckpt
  device: cuda:0
  model_source: auto       # auto | ema_model | model

inference:
  deterministic: false
  distill2mean: false
  use_cm: auto             # auto 表示严格跟随 checkpoint cfg.policy.use_cm
  execute_steps: 4         # 当前 checkpoint 的 action horizon；每 tick 消费一个 action
  action_buffer_size: 8    # 小缓冲，避免推理抖动导致动作断流
  action_low_watermark: 2  # 低于该值时后台补充下一块
  control_hz: 10           # 与采集 fps 一致；action 时间语义不变
  timeout_s: 0.50          # 启动/空缓冲等待上限，不是每 tick 的同步推理硬门槛
  warmup_runs: 3

observation:
  pointcloud_config: configs/rl/rl100_zarr_collect_upper_cams.yaml
  history_startup: repeat_first
  state_max_age_s: 0.15
  hand_max_age_s: 0.15
  depth_max_age_s: 0.15
  max_state_hand_skew_s: 0.10
  max_camera_skew_s: 0.10
  max_state_cloud_skew_s: 0.10

robot:
  deploy_config: configs/deploy/total/deploy_real_rl100_upper.yaml
  expected_raw_joint_dims: [20]
  expected_eef_type: qiangnao
  publish_unit: rad_to_sdk

safety:
  arm_joint_low_rad: [REPLACE_14_VALUES]
  arm_joint_high_rad: [REPLACE_14_VALUES]
  max_arm_step_rad: 0.02
  max_gripper_step: 0.05
  max_arm_state_jump_rad: 0.05
  max_gripper_state_jump: 0.10
  max_consecutive_clips: 3
  max_consecutive_source_failures: 1
  max_episode_steps: 200
  max_episode_duration_s: 20.0

startup:
  expected_pose16: [REPLACE_16_VALUES]
  max_arm_error_rad: 0.10
  max_gripper_error: 0.15
  require_physical_estop_ready: true
  require_operator_start: true

mode:
  shadow: true
  live_confirmation_token: ""

logging:
  output_dir: logs/rl100_real
  jsonl: true
  save_predicted_chunks: true
  save_pointcloud: false
```

配置约束：

- 网络结构、shape、`n_obs_steps`、`n_action_steps`、horizon、scheduler 和 normalizer 只以
  checkpoint 内嵌训练 YAML 为准，部署 YAML 不重复定义。
- `pointcloud_config` 应复用采集 YAML，或者以后提取成采集和部署共同引用的独立配置块；不要复制
  三份相机/裁剪参数后分别维护。
- `control_hz` 默认等于采集 `fps=10`。动作仍按10 Hz消费，不能因为推理较慢而把动作时间语义
  降到5 Hz；模型在后台按低水位补充 action chunk，结果按推理延迟丢弃已过期的开头动作。
- live 时 token 必须由操作者当次输入，不能长期写死在版本库。CLI 同时要求 `--confirm-live`。

## 5. 状态机和故障行为

```text
INIT -> PREFLIGHT -> READY -> SHADOW
                       |         |
                       |         +-- 操作者停止 -> STOPPED
                       v
                     ARMED -> RUNNING -> HOLD -> STOPPED
                                  |
                                  +------ 任一硬故障 -> FAULT -> STOPPED
```

- `PREFLIGHT`：checkpoint、CUDA、ROS master、topic 类型、频率、joint map、Qiangnao、三路深度、
  CameraInfo、TF、工作空间点数全部通过。
- `READY`：至少收到一组新鲜且对齐的完整观测，并完成模型 warmup。
- `SHADOW`：至少先连续运行规定步数，确认绝无 publish 调用。
- `ARMED`：仅 live 子命令可进入；检查启动姿态、物理急停和操作者确认。
- `RUNNING`：每 tick 消费一个已缓存 action；action buffer 低水位时后台补充下一块。
- `HOLD`：人工 pause 时保留当前姿态但停止继续推理；恢复前重新检查观测历史。
- `FAULT`：NaN、维度变化、旧消息、TF/相机缺失、时间偏差、action buffer 长时间耗尽、连续
  限幅、SDK 异常或急停触发。记录首个根因，停止产生新命令；不得自动恢复到 RUNNING。

ROS shutdown、Ctrl-C 和异常退出都要进入同一清理路径。信号处理器不执行复杂 ROS 操作，只设置
stop event，由主循环完成安全退出。

## 6. 必须记录的审计信息

启动时写一份 run manifest：

- checkpoint 绝对路径、SHA-256、内嵌配置快照、选用的 `model/ema_model`；
- git commit、部署 YAML 快照、采集点云 YAML 快照；
- ROS master、机器人型号、raw joint dim 和实际切片；
- 相机 topic/frame、是否零外参、控制频率、设备和 warmup 延迟。

每 tick 至少写 JSONL：

- 单调时钟、各 ROS stamp、age/skew；
- state16、原始首步动作、安全处理后动作、完整预测 chunk（可配置）；
- 推理耗时、循环耗时、是否 publish；
- 是否 clip、clip 维度和原因；
- 状态机状态、fault code、相机有效点数。

默认不保存每帧完整点云，避免实时磁盘压力；故障前后的环形缓存可选择性落盘。

## 7. 测试文件和验收标准

### 7.1 单元测试

新增：

- `kuavo_rl/tests/test_rl100_policy.py`
  - 正确选择 EMA/model；显式 EMA 缺失时报错。
  - 从内嵌 cfg 构建，normalizer state 被加载。
  - 16/16/1024 shape 不匹配时拒绝。
  - `module.` 前缀兼容仅作用于全量 key。
  - `use_cm` 被显式传入。
- `kuavo_rl/tests/test_rl100_real_runner.py`
  - 历史重复首帧和滑窗顺序。
  - 始终只执行 chunk 第一个动作。
  - shadow 下 publish 次数恒为 0。
  - stale、skew、NaN、超时、连续 clip 和 SDK 异常进入 FAULT。
  - fault 后不再发布。
- 扩展 `kuavo_rl/tests/test_safety_gate.py`
  - 14 个手臂维度和 2 个夹爪维度分别限幅。
  - 预测相对实测状态跳变被拦截。
- 扩展 `kuavo_rl/tests/test_rl100_zarr.py`
  - 点云 sample 返回真实时间戳、相机 skew 和 age。
- 扩展 Qiangnao/bridge 测试
  - `[L7,gL,R7,gR]` 的 16 维顺序不变。
  - `gL/gR` 正确映射到左右手 6 维命令且范围为 `0..100`。

所有单元测试都不得需要 ROS master 或真实 GPU；模型测试使用最小伪 checkpoint/假 policy。

### 7.2 离线回放

`offline-replay` 从新 zarr 逐帧构造与实机一致的历史：

- 运行完整数据时无 NaN、shape 错误和归一化异常；
- 输出上下限、动作增量分位数、预测与示范的 MAE；
- 单独报告手臂和夹爪，并检查夹爪开合方向与转折时刻；
- diffusion 多模态策略不能只用单次样本 MSE 判定好坏，至少固定随机种子重复多次并报告分布；
- 将预测动作经过同一 SafetyGate，统计 clip 比例。正式 shadow 前应先人工检查高 clip 片段。

### 7.3 实机分阶段验收

按以下顺序执行，任何一级失败都不能跳到下一级：

1. 新 zarr `inspect` 通过：action 不等于 state、夹爪有运动、维度和有限值正确。
2. `inspect-checkpoint` 通过，随机输入 warmup 结果有限，checkpoint shape 为 16/16/1024x3。
3. `offline-replay` 完整跑通并人工看关键片段。
4. `verify_joint_map.py --live` 通过；实机仍确认 raw dim 20、slice `[4:18]`。
5. `ros-preflight` 连续运行至少 60 秒，三路点云、TF、state、hand 无 stale/skew 故障。
6. `shadow` 至少 500 tick：publish=0、fault=0；推理 p99 小于控制周期，动作分布合理。
7. live 首次限制为 1 tick，夹爪命令屏蔽，人在急停旁观察双臂方向。
8. 双臂低步长运行 10 tick；确认 rad 单位、左右顺序和关节方向。
9. 机械臂保持，单独测试左右 Qiangnao 的 0、0.5、1.0。
10. 双臂和夹爪联合，逐步将 episode 上限从 10、50 提升到正式值。

shadow 的延迟验收以 10 Hz 时 p99 `<100 ms` 为目标。若达不到，先检查 checkpoint 是否训练为
CM；否则把控制频率降为 5 Hz 后重新完成 60 秒 preflight 和 500 tick shadow，禁止带超时上实机。

## 8. 最终命令形态

代码完成后 README 应给出以下命令，具体 checkpoint 路径由训练产物替换：

```bash
# 上位机相机
export ROS_MASTER_URI=http://kuavo_master:11311
export ROS_IP=192.168.26.12
source /opt/ros/noetic/setup.bash
source ~/kuavo_ros_application/devel/setup.bash
cd ~/wjy/robot-il/LeTools-Learning
roslaunch configs/launch/hil_upper_cams.launch

# 部署终端：只检查 checkpoint
bash scripts/rl/run_rl100_real.sh inspect-checkpoint \
  --config configs/rl/rl100_real_deploy.yaml

# ROS 全链路检查，不推理、不发布
bash scripts/rl/run_rl100_real.sh ros-preflight \
  --config configs/rl/rl100_real_deploy.yaml --duration-s 60

# 默认 shadow，只推理不发布
bash scripts/rl/run_rl100_real.sh shadow \
  --config configs/rl/rl100_real_deploy.yaml --max-steps 500

# 真机发布：实现时必须同时要求交互确认/一次性 token
bash scripts/rl/run_rl100_real.sh live \
  --config configs/rl/rl100_real_deploy.yaml --confirm-live --max-steps 1
```

不要通过 CLI 覆盖模型结构和训练参数。checkpoint 路径、运行模式和有限步数属于部署参数；
模型结构、scheduler 和 observation/action shape 属于 checkpoint 参数。

## 9. 明确禁止的实现方式

- 禁止使用旧 `grasp_8_4.zarr` 做回放或训练验收。
- 禁止复用 `kuavo_deploy/src/scripts/script.py` 的 LeRobot checkpoint loader 加载 RL-100。
- 禁止在代码里写死 `n_obs_steps=2`、`n_action_steps=4`、horizon 或 denoise steps。
- 禁止把 28 维 `[12:26]` 套到当前 20 维实机状态。
- 禁止在部署时改变相机 TF、工作空间裁剪、点数或状态/动作顺序。
- 禁止把 Qiangnao 夹爪当 Leju claw 或二指夹爪。
- 禁止绕过 `PublishedCommand`、安全门和已有 Kuavo SDK 控制链直接发布。
- 禁止异步重复发布上一动作来掩盖模型超时或观测断流。
- 禁止 fault 后自动恢复发布。
- 禁止以宽泛的 `[-pi, pi]` 作为最终实机关节安全范围。

## 10. 实现前仍需现场补齐的值

以下内容无法从训练 checkpoint 自动安全推断，代码可以先保留必填占位，但未填写时 live preflight
必须失败：

1. 14 个手臂关节的现场批准位置上下限。
2. 任务标准起始姿态 `expected_pose16` 及允许误差。
3. 操作者物理急停/开始/暂停信号的最终接口。
4. 最终 `.ckpt` 路径，以及该 checkpoint 实际内嵌 cfg 的检查结果。

## 11. Definition of Done

- 新增代码通过全部无 ROS 单元测试。
- 新采集 zarr 和最终 checkpoint 均有可保存的 inspect 报告。
- train/deploy 的点云和 16 维 state/action 契约逐项一致。
- 时间戳来自真实 ROS 消息，不存在用当前时间伪造 freshness 的路径。
- shadow 模式有自动测试证明不会 publish。
- 所有故障路径都会停止产生新命令，并留下结构化 fault 日志。
- 按第 7.3 节完成逐级实机验收，才能把配置中的默认模式从 shadow 改为 live；建议即使验收后仍保留 shadow 为默认。
