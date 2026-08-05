# RL-100 Topic-Native 实施记录

本文件记录本次代码落地的关键检查，不替代
[`RL100_TOPIC_NATIVE_COLLECTION_DEPLOYMENT_PLAN.md`](RL100_TOPIC_NATIVE_COLLECTION_DEPLOYMENT_PLAN.md)。

## 2026-08-05

1. 盘点 Kuavo 下位机源码，确认：`/kuavo_arm_traj` 使用 14 维 degree 且要求
   `JointState.name = arm_joint_0..arm_joint_13`；Qiangnao command 为左右各 6 个
   `uint8`，dexhand feedback 为固定 12 个 name、位置范围 0..100。
2. 新增隔离的 `rl100_topic_native_v1` 契约：state 32 维、action 26 维；没有改动项目其他
   HIL/ACT 使用的通用 16 维契约。
3. 实现无 ROS 依赖的双 command cache：任意一路首条合法消息即可写样本；另一侧使用确认过的
   measured/default hold；命令永久 hold-last，并以 monotonic cutoff 做因果对齐。
4. 实现 `TopicStateHub`：直接读取完整 raw joint20 与 dexhand12，校验名称、范围、freshness、
   skew，并支持状态频率/间隔/范围 profile。
5. 改造 NPZ staging、zarr writer 和 inspect：逐帧 audit 数组真实落盘，zarr attrs 声明
   contract、单位、顺序、维度、关节名、默认手部值、smooth penalty 和配置哈希。
6. 实现直接 ROS publisher 和 topic-native safety gate：arm 在 rad 中做限幅后转回 degree，
   hand clip/round 为 uint8；shadow 不发布，fault 最多发送一次 measured-arm + last-safe-hand hold。

## 阶段提交

- `608d332 feat: 落地RL100主题原生采集契约与部署基础链路`

- 本次收尾提交包含 README、profile/验收修正、离线 Zarr 契约硬校验和最终测试记录；不会加入
  现有未跟踪的 `third_party/kuavo-ros-control/`。

## 已完成的本地验证

- topic-native、zarr、policy、runner、ROS teleop 定向测试：38 passed。
- synthetic smoke：成功生成并 inspect `state(*,32)`、`action(*,26)` zarr，contract/units/attrs 正确。
- staging audit round-trip：`audit__*` 从 NPZ 进入 zarr `meta/*`，因果违规计数为 0。
- `git diff --check`、修改 Python 文件 `py_compile` 通过。

根目录全量 pytest 在当前环境收集阶段无法完成：环境缺少 gymnasium/torch，另有仓库既有 Python
版本兼容错误；这不影响上述无 ROS RL-100 定向测试。

## 尚需实机确认

- 运行 `ros-preflight --duration-s 60`，记录 joint/dexhand 频率和 p50/p95/p99 后再决定 freshness 阈值。
- 确认两个 command publisher 各有唯一正确 subscriber、下位机控制模式接受 command topic，并做
  当前姿态 hold 的 echo/反馈单 tick。
- 先 shadow，再按方案做 hand-only、arm-only、联合 10 tick，最后才允许 live。
