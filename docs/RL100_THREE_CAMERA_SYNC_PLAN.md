# RL-100 三相机时间同步改造方案

本文是交给后续代码实现者的执行规格，目标是解决 Kuavo 实机上三路深度相机
（头部 Orbbec Gemini 335L、左右腕 RealSense D405）同步性差的问题，并保证采集和部署使用
同一套点云时序语义。

当前阶段只定义方案，不在本文工作中修改采集实现。

## 1. 已确认的现状

当前 RL-100 点云链路有以下行为：

- `DepthPointCloudHub` 每路只保存一张最新深度图。
- 深度新鲜度使用本机接收时间判断。
- 三相机偏差使用 ROS `header.stamp` 的 `max-min` 判断。
- 三张“最新图”没有经过近似同步，可能来自不同采样时刻。
- TF 查询使用 `rospy.Time(0)`，即最新 TF，而不是图像采样时刻的 TF。
- 当前采集配置订阅三路 `compressedDepth`。
- 为暂时保证采集程序可运行，当前 camera skew 门限已经放宽到 2 秒；该设置只是临时旁路，
  不能作为正式数据质量标准。

实机测量结果：

- 三路原始相机目标频率均约 30 Hz。
- 头部 `/camera/depth/image_raw` 实测约 30 Hz。
- 头部 compressedDepth 的有效接收频率明显更低，且 header age 明显高于两路腕部相机。
- 三相机 header skew 的 p50 约 120 ms、p95 约 208 ms、最大约 413 ms；实际采集过程中
  曾出现 0.5 至 1.1 秒。
- 消息 `received_age` 很小但 header age 很大，说明异常不完全等同于网络接收延迟，可能包含
  compressedDepth 排队、旧帧晚到、不同驱动时间域以及“最新帧直接拼接”等因素。

`third_party/kuavo_brain` 给出的参考行为：

- 生产相机链优先使用 raw depth 话题。
- 相机启动文件禁用默认 compressed/compressedDepth 插件，后续编码器从 raw 输入并保留原 header。
- Orbbec 驱动支持 `device/global/system` 时间域及 host time sync。
- 启动链具有系统时间门禁，但没有发现用于当前 Orbbec + 两路 RealSense 生产链的三相机
  `ApproximateTime` 同步器。
- 仓库中的 `ApproximateTime` 是多 Orbbec 示例，不是当前混合相机生产路径。

## 2. 最终技术决策

三台相机属于混合厂商设备，不能依赖同一套硬件触发。最终采用：

```text
三路 raw depth
    -> 统一且可验证的 ROS header 时间域
    -> 每相机有界环形缓存
    -> 基于 header 时间的因果近似同步
    -> 按每张图自己的时间查询 TF
    -> 三路点云转换、裁剪与融合
    -> 写入采集/部署统一的 PointCloudSample 与审计数据
```

以下决定是实现约束：

1. 正式模式使用 raw depth，不再以 compressedDepth 作为首选输入。
2. 正式同步使用 `header.stamp`；本机接收时间只用于因果边界、新鲜度和诊断。
3. header 时间域未通过 preflight 时，不得静默改用接收时间并宣称完成同步。
4. 每个采样点同步失败时跳过该采样点，不立即丢弃整个 episode。
5. 连续失败超过配置上限才中止 episode。
6. 采集与部署必须共用同一个 `DepthPointCloudHub` 同步实现和同一套 YAML 参数。
7. 不修改 RL-100 的 state32/action26 topic-native 契约。
8. 不引入 Orbbec 多机硬件触发方案，因为腕部相机是 RealSense。

## 3. 话题和相机驱动配置

### 3.1 正式 raw depth 话题

根据当前实际 ROS 命名，在 YAML 中改为对应的 raw 话题：

```yaml
cameras:
  - name: head_cam_h
    depth_topic: /camera/depth/image_raw
    depth_msg_type: image
    camera_info_topic: /camera/depth/camera_info
    frame_id: camera_color_optical_frame

  - name: wrist_cam_l
    depth_topic: /left_wrist_camera/depth/image_rect_raw
    depth_msg_type: image
    camera_info_topic: /left_wrist_camera/depth/camera_info
    frame_id: left_wrist_camera_depth_optical_frame

  - name: wrist_cam_r
    depth_topic: /right_wrist_camera/depth/image_rect_raw
    depth_msg_type: image
    camera_info_topic: /right_wrist_camera/depth/camera_info
    frame_id: right_wrist_camera_depth_optical_frame
```

实现者必须先用 `rostopic type` 验证以上 raw topic 的消息类型确实是
`sensor_msgs/Image`，并确认深度编码为 `16UC1/mono16` 或现有解码器支持的格式。

不要在没有实机验证的情况下直接修改 `config.py` 中所有项目的默认 topic。优先修改 RL-100
实机采集和部署 YAML；默认值是否迁移应在兼容测试通过后单独决定。

### 3.2 Orbbec 时间域

目标参数：

```yaml
time_domain: global
enable_frame_sync: true
```

`enable_sync_host_time` 必须通过实机诊断决定，不能仅根据驱动默认值猜测。使用 Orbbec 的
timestamp debug 比较：

```text
selected_us
hw_us
sys_us
global_us
ROS now
```

验收标准是 selected timestamp 与 ROS 系统时间处于同一 epoch、不会倒退、固定偏移和漂移可接受。
如果实际启动链不是 `third_party/kuavo_brain` 中的 launch，应以运行中的 ROS private param 和日志为准。

### 3.3 RealSense 时间戳

实现者必须确认两路 RealSense 发布的 header：

- 与 ROS 系统时间处于同一 epoch；
- 连续递增；
- 左右相机之间没有持续增长的漂移；
- 重启驱动后不会突然切换时间域。

系统 chrony/time gate 只能保证主机时钟稳定，不等于相机帧已经同步。相机 header 验证必须作为
独立 preflight 项。

## 4. 配置模型改造

修改：

```text
kuavo_rl/rl100_zarr/config.py
configs/rl/rl100_zarr_collect_upper_cams.yaml
configs/rl/rl100_real_deploy.yaml
kuavo_rl/rl100_deploy_config.py（若部署配置单独定义点云限制）
```

建议在 `RL100CollectConfig` 或独立点云同步配置中增加：

```yaml
camera_sync:
  mode: buffered_header
  reference_camera: head_cam_h
  buffer_size: 32
  max_header_skew_s: 0.10
  warn_header_skew_s: 0.05
  max_received_age_s: 0.20
  max_receive_skew_s: 0.20
  require_monotonic_header: true
  require_same_time_epoch: true
  tf_at_image_stamp: true
  tf_timeout_s: 0.05
  max_consecutive_sync_failures: 10
```

命名可以适配现有风格，但语义不得合并：

- `max_header_skew_s`：选中三帧的采样时间差。
- `max_received_age_s`：当前时刻距离消息本机接收时刻的最大值。
- `max_receive_skew_s`：三帧本机接收时刻的差，用于发现排队或突发传输。
- `tf_timeout_s`：按图像时间查询 TF 的等待上限。

兼容期可保留旧字段 `camera_max_skew_s`，但加载时必须明确映射并打印一次弃用提示。不能让新旧字段
同时存在时悄悄选择其中一个；应报配置冲突。

所有时间阈值必须为正，`warn_header_skew_s <= max_header_skew_s`，缓存至少能覆盖
`max_received_age_s + max_header_skew_s` 对应的帧数。

## 5. `DepthPointCloudHub` 数据结构

主要修改文件：

```text
kuavo_rl/rl100_zarr/ros_depth.py
```

### 5.1 帧结构

用不可变帧对象替代单一 `_CamSample` 最新值：

```python
@dataclass(frozen=True)
class TimedDepthFrame:
    camera_name: str
    depth: np.ndarray
    intrinsics: tuple[float, float, float, float]
    frame_id: str
    header_stamp_s: float
    received_wall_s: float
    received_monotonic_s: float
    sequence: int | None
```

必须同时记录墙钟和单调时钟：

- `header_stamp_s`：跨传感器采样时间对齐。
- `received_monotonic_s`：本进程内因果截止和 freshness。
- `received_wall_s`：日志和人工排查。

禁止把 `time.time()` 值与 `time.monotonic()` cutoff 直接比较。当前 command cache 使用
`time.monotonic()`，深度同步也应使用单调接收时间执行 cutoff。

### 5.2 环形缓存

每个 enabled camera 保存独立 `deque(maxlen=buffer_size)`。回调流程：

1. 记录 `received_monotonic_s` 和 `received_wall_s`。
2. 解析并校验 header stamp、frame id、图像维度、编码和有限性。
3. 检查该相机 header 是否倒退；倒退帧拒绝入队并计数。
4. 将有效帧原子追加到该相机 deque。
5. 不在 ROS callback 内做 TF 查询、点云反投影、FPS 或三路融合。

为限制内存，优先缓存原始 `uint16` 深度并在选中后转换为米。如果现有 `CvBridge` 路径已经返回
float32，则必须评估 `3 * buffer_size * H * W * 4` 的内存占用，并在 preflight 中报告估算值。

CameraInfo 不必每帧复制；可按相机保存最新有效内参及其 frame id，并在分辨率变化时使旧缓存失效。

## 6. 三相机近似同步算法

新增一个尽可能纯 Python/NumPy、无 ROS 依赖的选择函数，便于单元测试：

```python
def select_synchronized_frames(
    histories: Mapping[str, Sequence[TimedDepthFrame]],
    *,
    cutoff_monotonic_s: float,
    reference_camera: str,
    max_received_age_s: float,
    max_header_skew_s: float,
) -> SynchronizedDepthBundle:
    ...
```

固定算法如下：

1. 只考虑 `received_monotonic_s <= cutoff_monotonic_s` 的帧，防止把 sampler cutoff 后到达的未来帧
   写入较早样本。
2. 在参考相机缓存中，选择 cutoff 前接收的最新且未过期帧。第一版参考相机固定为头部相机，因为
   当前头部链路最慢。
3. 对每个非参考相机，选择 `abs(frame.header_stamp-reference.header_stamp)` 最小的候选。
4. 相同差值时，优先 header 较早的帧，再优先更早接收的帧，保证选择结果确定且避免未来视觉信息。
5. 计算选中三帧的 header skew、receive skew 和各自 received age。
6. 任一帧过期、header 无效、缺少相机、header skew 超限时返回结构化失败，不返回旧的上一次 bundle。
7. 成功后返回不可变 bundle，并记录每路实际选择的 stamp、age 和 buffer index/sequence。

第一版不要实现自动 fixed-offset 校正。若后续证明某驱动存在稳定固定偏移，应先修驱动时间域；只有
无法修复且经过独立标定时，才考虑显式的 `per_camera_stamp_offset_s`，并把偏移写入数据集 attrs。

`received_at` 同步只能作为诊断模式，不得成为正式训练数据的默认模式，因为它包含解码、ROS 排队和
线程调度延迟。

## 7. TF 与点云融合

当前 `_lookup_T_base_cam()` 使用 `rospy.Time(0)`，必须增加按图像时间查询：

```python
lookup_transform(
    base_frame,
    frame_id,
    rospy.Time.from_sec(frame.header_stamp_s),
    rospy.Duration(tf_timeout_s),
)
```

三张图分别使用自己的 header stamp 查询 TF，不能三路共用参考帧时间，也不能全部使用最新 TF。
原因是左右腕相机随手臂运动，旧图配最新 TF 会直接造成点云空间错位，即使三路图像 header 已同步。

严格模式下：

- 指定时间的 TF 不可用则当前采样失败。
- 不允许静默 fallback 到 `Time(0)`。
- preflight 应分别报告三路 `tf_at_stamp` 是否成功。

成功选择 bundle 后才执行：

```text
深度转米 -> 各相机反投影 -> 各自时刻 TF 到 base_link
-> 工作空间裁剪 -> 三路融合 -> FPS 到 1024 点
```

`PointCloudSample.fused_stamp_s` 应定义为参考相机 stamp，而不是当前实现的 `min(stamps)`。同时保留
`min/max/每相机 stamp`，避免丢失信息。

## 8. 输出结构和审计字段

扩展 `PointCloudSample`：

```python
@dataclass(frozen=True)
class PointCloudSample:
    points: np.ndarray
    reference_camera: str
    reference_stamp_s: float
    generated_wall_s: float
    generated_monotonic_s: float
    oldest_received_age_s: float
    max_header_skew_s: float
    max_receive_skew_s: float
    camera_header_stamps: dict[str, float]
    camera_received_wall: dict[str, float]
    camera_received_monotonic: dict[str, float]
    camera_received_ages: dict[str, float]
    valid_points: int
```

若保留旧字段，应通过只读 property 提供兼容别名，禁止同时维护两份可能不一致的数据。

在 `live_collect.py` 的逐帧 audit 中增加：

```text
point_cloud_reference_camera
point_cloud_reference_stamp
point_cloud_header_skew
point_cloud_receive_skew
head_depth_stamp
left_depth_stamp
right_depth_stamp
head_depth_received_at
left_depth_received_at
right_depth_received_at
head_depth_age
left_depth_age
right_depth_age
```

现有 staging NPZ、Zarr `meta/*` 已支持逐帧 audit 数组，实施者应复用该机制，不要把逐帧数据塞入
单个 episode `meta_json`。

episode 汇总报告增加：

- header skew 的 p50/p95/p99/max；
- receive skew 的 p50/p95/p99/max；
- 每路 received age 的 p50/p95/p99/max；
- 每路接收帧数、有效帧数、header 倒退数和解码失败数；
- 同步尝试数、成功数、超限数、缺帧数、TF-at-stamp 失败数；
- 同步采样成功率。

## 9. 采集循环改造

修改：

```text
kuavo_rl/rl100_zarr/live_collect.py
```

每个 10 Hz 采样周期按以下顺序执行：

1. 记录唯一的 `sample_cutoff_monotonic_s`。
2. 从 command cache 读取 cutoff 前的 action26。
3. 从 state hub 读取 cutoff 前可用的 state32；若 state hub 暂时仍只有 latest snapshot，至少保证其
   callback 完成时间不晚于 cutoff，后续可单独升级为历史选择。
4. 调用点云 hub，以同一个 cutoff 选择同步三相机 bundle。
5. 点云选择/TF/融合成功后才 append state、action、point cloud 和 audit。
6. 单点失败时运行一次 env step 以保持 B/Y 标签响应，但不写样本。
7. 失败计数应区分 `camera_sync`、`camera_stale`、`camera_tf`、`state_stale`，不要全部归为
   `source_error`。

不要让同步器阻塞等待未来帧跨过 sampler cutoff。因果要求高于“凑齐三帧”；本周期没有合格 bundle
就跳过，下一周期重试。

当前 `max_consecutive_source_failures=3` 对三相机同步过严。建议把相机同步连续失败上限独立设置为
10（10 Hz 下约 1 秒），状态源失败仍保留较严格门限。最终值由实机 pilot 决定。

## 10. 部署链一致性

部署 CLI `scripts/rl/run_rl100_real.py` 已复用 `DepthPointCloudHub`，因此不要另写一套部署同步器。

部署要求：

- `ros-preflight` 必须执行 buffered sync，而不是只检查三路各有一张图。
- preflight profile 至少 30 秒并输出同步分位数。
- shadow/live 推理输入使用与采集完全相同的 reference camera、raw topics、buffer、阈值、TF-at-stamp
  和点云融合规则。
- checkpoint/dataset attrs 应记录同步配置摘要，部署启动时对关键字段做一致性检查。
- live 模式中同步失败不得重复发布上一条策略动作；按现有安全状态机进入 skip/hold，连续失败再 fault。

## 11. Preflight 分层

`preflight --check-ros` 和部署 `ros-preflight` 应分为以下检查：

### 11.1 Topic 基础检查

- 三路 raw depth 和 CameraInfo 均存在。
- 消息类型、编码、维度和 frame id 正确。
- 三路频率满足最低要求，例如 25 Hz。
- 最近接收时间满足 freshness。

### 11.2 Header 合法性检查

- header 非零、有限、处于合理 epoch。
- header 不倒退。
- `wall_now-header_stamp` 不出现持续数百毫秒以上异常。
- 三路 header 偏差没有持续增长趋势。

### 11.3 Buffered sync profile

连续运行 30 至 60 秒，真实执行选择算法并统计：

- 同步成功率；
- header skew 分位数；
- receive skew 分位数；
- 每路 age；
- TF-at-stamp 成功率。

只有这一层通过，preflight 的总 `ok` 才能为 true。临时调试模式允许输出 warning 后继续，但 JSON
必须明确 `strict_sync_ok=false`，不能把放宽门限后的结果标为正式通过。

## 12. 单元测试

主要增加到：

```text
kuavo_rl/tests/test_rl100_zarr.py
```

如果测试过大，可新增：

```text
kuavo_rl/tests/test_rl100_camera_sync.py
```

至少覆盖：

1. 三路完全同时间戳，选择成功。
2. 不同频率下选择最接近参考帧的候选。
3. 相同距离时选择较早 header，结果确定。
4. cutoff 后到达的帧不得被选择。
5. header skew 超限返回结构化失败。
6. received age 超限返回 stale，而不是 skew。
7. 任一路缓存为空返回 missing camera。
8. header 倒退帧被拒绝并计数。
9. 无效/零 header 在 strict 模式失败。
10. 不得混用 wall clock 和 monotonic clock。
11. 缓存达到 maxlen 后只淘汰最旧帧。
12. selected bundle 中每路 frame id、stamp、received time 保持正确。
13. TF 查询收到每路自己的 header stamp，而不是 Time(0)。
14. TF-at-stamp 失败不 fallback 到最新 TF。
15. 一次同步失败不会复用上次成功点云。
16. `PointCloudSample` 新旧兼容属性一致。
17. YAML 新字段解析、默认值、非法组合和新旧字段冲突。
18. audit 数组长度始终与 episode steps 一致，并能写入/读回 NPZ 和 Zarr。
19. 部署 preflight 使用相同 buffered sync 路径。

测试中不要依赖真实 ROS。把帧选择和阈值判断提取为纯函数；TF buffer 使用 fake/mock。

## 13. 实机分阶段验收

### 阶段 A：只读诊断

- 不改相机驱动参数。
- 同时统计 raw 与 compressedDepth 的 rate、header age、receive age 和 skew。
- 确认 raw 链路是否消除头部旧 header 问题。

通过条件：三路 raw 稳定接近 30 Hz，且 raw 明显优于 compressedDepth。

### 阶段 B：raw + buffered sync shadow

- 采集器使用 raw topic 和新同步器。
- 不正式保存训练 episode，或只保存标记为 pilot 的数据。
- 门限先设 `warn=50ms`、`hard=100ms`。

建议通过条件：

```text
header skew p95 < 50 ms
header skew max < 100 ms
received age p95 < 100 ms
同步成功率 >= 99%
TF-at-stamp 成功率 >= 99.9%
三路无持续 header 倒退或漂移
```

如果 raw 后仍有稳定固定偏移，不要先把 hard limit 放宽；先回到相机时间域诊断。

### 阶段 C：小规模正式 pilot

- 录制 2 至 5 个 episode。
- inspect staging NPZ 和最终 Zarr audit。
- 可视化融合点云，重点检查手臂快速运动时腕部点云是否拖影、重影或空间跳变。
- 验证失败采样被跳过而 episode 标签仍正常。

### 阶段 D：批量采集

只有阶段 C 通过后才能批量采集训练数据。数据集 attrs 至少记录：

```text
camera_sync_mode
camera_reference
camera_topics
camera_msg_types
camera_buffer_size
camera_warn_skew_s
camera_hard_skew_s
tf_at_image_stamp
sync_profile_summary
```

## 14. 回滚与兼容策略

- 保留 compressedDepth 解码支持，供旧配置和诊断使用，但正式 YAML 切到 raw。
- 保留 `get_point_cloud()` 兼容包装；内部调用新同步路径。
- 如需临时回滚，允许显式 `camera_sync.mode: latest_legacy`，但启动日志和数据集 attrs 必须标记
  `UNSYNCHRONIZED_LEGACY`。
- 禁止在新模式失败后自动回退 legacy，因为这会让同一个数据集混入两种时序语义。
- 当前用户配置中的 2 秒阈值属于未提交工作，不得在实现过程中覆盖或误提交其他无关修改。
- `third_party/kuavo-ros-control/` 是用户同步的独立源码目录，不得纳入本任务提交。

## 15. 实现顺序和提交要求

推荐按以下顺序实现，每一步都先运行相关测试：

1. 增加配置结构和验证，但暂不切换默认运行模式。
2. 实现 `TimedDepthFrame`、每相机 deque 和纯同步选择函数。
3. 增加基于每帧时间的 TF 查询和扩展 `PointCloudSample`。
4. 接入 `live_collect.py` 的统一 cutoff、分类失败和逐帧 audit。
5. 接入采集/部署 preflight 的 profile 统计。
6. 修改实机采集和部署 YAML 为 raw + buffered header sync。
7. 补齐 NPZ/Zarr、部署链及兼容测试。
8. 运行完整 RL100 测试和静态检查。
9. 对照本文逐条检查，再进行实机阶段 A/B。

实现完成前至少运行：

```bash
python -m pytest -q kuavo_rl/tests/test_rl100_camera_sync.py
python -m pytest -q kuavo_rl/tests/test_rl100_zarr.py
python -m pytest -q kuavo_rl/tests/test_rl100_topic_native.py
python -m pytest -q kuavo_rl/tests/test_rl100_real_runner.py
```

若没有新增独立测试文件，则删去第一条并确保相关用例已放入 `test_rl100_zarr.py`。

提交前执行：

```bash
git diff --check
git status --short
```

提交必须只包含本方案授权的同步实现、配置、测试和文档，不得带入用户已有的无关改动。

## 16. 完成定义

只有同时满足以下条件，才能声明三相机同步改造完成：

- 正式采集和部署 YAML 使用三路 raw depth。
- 三路 frame header 通过时间域、单调性和 epoch preflight。
- 点云 hub 使用有界缓存和因果近似同步，不再直接拼三路 latest。
- 腕部动态 TF 按各自图像时间查询，不使用最新 TF 冒充历史 TF。
- 单点失败不会复用旧点云，也不会立即丢弃整条 episode。
- 全部逐帧同步审计能够从 staging NPZ 和 Zarr 读回。
- 采集与部署使用同一同步路径和同一配置语义。
- 单元测试通过，实机阶段 B/C 达到本文验收指标。
- 2 秒临时 skew 门限退出正式配置，正式 hard limit 恢复到 100 ms 或实测证明更严格的值。
