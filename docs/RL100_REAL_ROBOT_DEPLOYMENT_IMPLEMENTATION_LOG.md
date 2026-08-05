# RL-100 实机部署实现记录

本记录只描述代码实现和验证过程，不替代现场验收。

- 已实现 checkpoint 直载：从 RL-100 workspace `.ckpt` 的 `cfg.policy` 构建模型，加载
  `ema_model/model` 和内置 normalizer，拒绝 LeRobot 格式。
- 已实现同步 runner：维护 checkpoint 指定长度的 observation history，每次只发布 action chunk
  首步；shadow 模式不会调用 backend publish。
- 已新增点云及状态时间戳路径：深度融合返回 source stamp/age/skew，Kuavo obs buffer 透传关节和
  Qiangnao 原始 ROS stamp；时间字段缺失时 runner 会失败关闭。
- 每次 shadow/live run 会写 JSONL tick 审计和 manifest（checkpoint SHA、内嵌 cfg、部署及点云 YAML
  快照、Git commit、ROS 网络变量）。
- 已增加部署 YAML。它默认 shadow，现场关节范围及起始姿态仍是 live 的强制前置条件。
- 已完成 `py_compile` 和部署相关定向单测：31 项通过（policy、runner、安全门、点云、动作契约）。
- 已尝试运行完整 `kuavo_rl/tests`。当前机器是 Python 3.8，且缺少 `torch`、`gymnasium`；
  另外旧 HIL recording 测试使用 Python 3.10 的 `X | None` 注解，因环境版本在收集阶段失败。
  这不是本次部署模块的失败；需在目标 ROS/PyTorch 环境重新执行全套测试和实机分阶段验收。
