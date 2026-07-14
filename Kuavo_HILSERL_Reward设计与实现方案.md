# Kuavo HIL-SERL Reward 设计与实验方案

> 更新：2026-07-14。本文以“人工标注是 ground truth”为原则，定义 Kuavo 真机/仿真 HIL-SERL 的 reward 实验、上线门禁和回退条件。

## 1. 结论

推荐路线是：先用已接入的人工稀疏 reward 启动数据闭环，再用同一批人工标签校准任务专用 reward classifier；Robometer、SARM 和 TOPReward 均先作为离线候选评分器，不阻塞训练，也不直接进入控制环。

```text
人工 result 标签 + VR 接管数据
        ↓
固定、按 episode 切分的评测集
        ↓
人工稀疏 reward 训练 / 数据采集
        ↓
任务专用 success classifier（离线 → 影子）
        ↓
满足门禁后才允许在线 success reward
        ↓
Robometer / SARM / TOPReward 离线赛马、回填或重加权
```

核心约束：任何模型 reward 想给正奖励，必须先在固定的人工标注评测集上通过门禁。模型不替代人工 failure、abort、急停或 SafetyGate 的判定。

## 2. 当前能力与边界

| 项目 | 当前状态 | 用途 / 限制 |
|---|---|---|
| 人工 reward | 已接入仿真联调 | 右手 `B` 单击=success、双击=failure、长按=abort；安全故障/急停独立处理。真机仍需现场复核。 |
| VR 接管 replay 存储 | 已实现 | 每个 transition 保存观测、图像、策略 action、原始 VR/IK action、接管 mask、段编号与 SafetyGate 后 action。 |
| HIL replay importer | 未实现 | 目前数据已落盘，尚未自动导入 SAC online replay。不能宣称 SAC 已使用这些 HIL 样本训练。 |
| RewardClassifier | 上游能力可用，Kuavo 未接入 | 需要 Kuavo 专用数据、训练、评测和在线适配。 |
| Robometer-4B | 已有离线校准路径 | 仅 episode 级异步评分；未通过门禁前禁用在线 reward。 |
| SARM / TOPReward | 上游可选能力，Kuavo 未接入 | 先离线生成/比较 progress，不是当前 SAC 的即插即用在线 reward。 |

### 2.1 现行人工事件语义

| 事件 | reward | episode 结果 |
|---|---:|---|
| 普通控制步 | 0 | 继续 |
| `B` 单击：人工 success | +1 | `terminated` |
| `B` 双击：人工 failure | 0 | `truncated` |
| `B` 长按：人工 abort | 0 | `truncated` |
| SafetyGate、急停、SDK 故障 | -1 | `terminated` |
| 超时 | 0 | `truncated` |

`trigger`、`grip`、摇杆以及左手双键急停不承担任务成功标签。人工 result 事件必须记录原始时间戳、事件类型和操作者，便于回溯。

### 2.2 接管 action 与 reward 数据不能混淆

VR 的原始 IK 目标是审计数据，不是直接用于训练的 action label。当前 replay 对接管步使用：

```text
接管前实测关节状态
  + 只覆盖 intervention_mask 对应的手臂维度
  + SafetyGate 裁剪
  = replay action
```

未被接管的一侧保持接管前实测状态，避免把陈旧的 VR 目标写成另一只手的训练标签。后续 importer 应依据 `intervention_segment_id` 和 `intervention_segment_step` 跳过接管切换缝合处的少量步，并保留 mask 供采样策略使用。

## 3. 设计原则

1. 人工标签优先。人工 success/failure/abort 和安全事件是唯一 ground truth；模型只能先做候选信号。
2. 对比以 episode 为单位。相邻帧高度相关，训练集和评测集必须按 episode、场景和采集批次切分，不能随机拆帧。
3. 正奖励宁缺毋滥。误报成功会让 SAC 学会骗分；失败/中止仍由人工和安全链路决定。
4. 控制与评分解耦。不能让慢速视觉/VLM 推理阻塞 10 Hz 控制；先离线或影子记录，测量实际延迟和 GPU 争用后再决定在线频率。
5. 任务完成态与过程进度拆开。success classifier 只判断“当前是否完成”；未经校准的 progress 不得直接作 dense reward。

## 4. R0–R4 实验路线

### R0：带标签采集与固定考卷

目标是建立 `data/reward_calibration/` 下可复现、不可随意更换的评测集。初始采集目标可采用：

- 成功至少 30 个 episode；其中必须包含策略自主完成的样本，不能全是示教；
- 失败至少 30 个 episode，覆盖空抓、抓偏、掉落、卡死、抓到但未放到目标；
- 至少 10 个困难场景：光照变化、遮挡、物体/目标偏移、相机轻微扰动；
- 每个 episode 保存三路相机、16-D state、人工结果、事件时间戳、接管/安全元信息。

上述数量是启动规模，不是证明低假阳性率的充分统计量。建议固定一组按 episode 切分的 holdout 集，并随着数据增长扩大失败 holdout。仅 30 个独立失败 episode 无法可靠证明“假阳性率不超过 2%”。

对 classifier 的帧标签也应谨慎：成功 episode 的末端稳定窗口才是正样本；失败 episode 的终态、近似成功但未完成的画面应作为负样本/难负样本。不要把一个成功 episode 的全部过程帧都标成成功。

### R1：人工稀疏 reward 先开跑

不等待模型 reward。ACT/现有策略运行时，使用 B 按键记录 success/failure/abort，并保留 VR 接管。这样既可以提供稀疏终局 reward，也持续积累 R0 评测集和 classifier 训练集。

当前已具备事件和 replay 落盘能力；R1 进入真正 SAC 训练前，仍需完成 HIL replay importer、时间对齐校验和真机按键复核。

### R2：训练 Kuavo 专用 reward classifier

第一版只训练二分类 `is_success_state`，使用与部署一致的相机组合和图像预处理。上游的 ResNet10 reward classifier 可作为实现起点，但不是预训练后即可直接使用的 Kuavo reward。

离线门禁建议：

| 指标 | 上线目标 | 说明 |
|---|---:|---|
| ROC-AUC | ≥ 0.95 | 在固定、episode-disjoint holdout 上计算。 |
| 假阳性率 | 目标 ≤ 2% | 以独立失败 episode 统计；样本量不足时同时报告置信区间，不能只报百分比。 |
| 成功判定稳定性 | 连续多帧确认 | 建议连续 3 帧或等效时间窗均超过阈值，且结果不可与人工终态矛盾。 |
| 推理延迟 | p95 ≤ 10 ms 才可考虑每控制步同步 | 必须在实际 Kuavo GPU、相机输入和策略并行负载下测量。未达标则只低频/异步影子运行。 |

通过离线门禁后，至少影子运行 20 个 episode：模型分数只记录，不改变 reward 或终止。影子验收重点是终局 success decision、假阳性和人工结果的一致性，而非逐帧“准确率”。建议覆盖不少于 10 个成功和 10 个失败/中止 episode；数据不足则继续采集。

转正后，classifier 只能补充 success reward。人工 failure/abort、SafetyGate 和急停始终具有更高优先级；在线成功仍使用连续帧门控。

### R3：Robometer / SARM / TOPReward 离线赛马

三者使用同一份 R0 固定评测集，输出均写入旁路文件，不影响控制：

- Robometer：保持既有门禁，Spearman ≥ 0.7、success AUC ≥ 0.85；还必须检查失败轨迹的 progress 不会持续爬升。
- SARM：可提供阶段/进度估计，但单阶段线性时间进度可能奖励“拖时间”或不适用于可回退任务。先验证任务阶段定义，再讨论使用。
- TOPReward：适合离线视频轨迹评分；模型较大、推理慢，不作为 10 Hz 控制路径候选。

离线赢家的首要用途是样本筛选、错误标注发现和历史轨迹排序。若要用于 replay 重加权，必须先验证加权后人工成功率与安全指标没有变差。

注意：上游 RA-BC 是面向特定 VLA/行为克隆训练流程的后处理能力，不等于 SAC 可直接接收的 reward shaping。Robometer 的 progress 也不能因名字相近就直接当作 RA-BC 权重；需要单独实现并验证 importer/采样器逻辑。

### R4：受保护的在线启用与回退

模型 reward 转正后，持续记录模型分数、人工事件、SafetyGate、策略回报和人工成功率。以下任一触发即关闭模型正奖励，回退到人工稀疏 reward：

- 人工抽查中，模型终局判定不一致率超过 5%；
- SAC 模型回报上升，但人工成功率没有同步提升；
- 新光照、物体位置或相机异常下出现高置信度成功；
- 成功判定发生在任务物体未就位、触碰危险区或 SafetyGate 已介入时；
- 视觉推理使控制周期或 GPU 资源超过已验证预算。

## 5. 为什么不是直接使用 Robometer 的 dense progress

Dense 信号的信息量更大，但 RL 更需要“方向正确”而不是“每步都有分”。未经 Kuavo 人工标签校准的 progress 可能对失败轨迹也给出上升分数，SAC 会优化这个漏洞而非完成搬运任务。

因此当前优先级是：

```text
现在：人工 B 事件 + 数据采集
随后：Kuavo success classifier 离线验证、影子运行
并行：Robometer/SARM/TOPReward 离线比较
最后：仅让通过门禁的信号进入 replay 加权或在线 reward
```

## 6. 数据存储与最小字段

现有 HIL replay 的推荐根目录为：

```text
data/rl_runs/hilserl_episodes/hilserl_vr/
├── transitions.jsonl                 # 审计索引
└── replay/
    ├── schema.json
    └── episodes/<episode_id>/
        ├── transitions.jsonl
        └── frames/
```

每个训练/评测 episode 至少关联：相机帧与时间戳、16-D state、policy action、raw VR/IK audit action、SafetyGate 后 replay action、`intervention_mask`、接管段信息、人工结果、故障和终止原因。R0 评测集还应保存场景 ID、物体初始位姿区间、光照/遮挡标签、操作者和软件版本。

## 7. 验收清单

- [x] 仿真中人工 success/failure/abort 事件已接入（右手 B 手势）。
- [x] VR 接管的原始数据与可训练 replay action 已区分落盘。
- [ ] 真机确认 B 单击、双击、长按与急停互不冲突。
- [ ] HIL replay importer 将带 mask/segment 的样本安全导入 SAC replay。
- [ ] 建成 episode-disjoint 的 R0 固定评测集，并记录其版本。
- [ ] 人工稀疏 reward 的时间对齐、终止语义和 SafetyGate 语义在真机验收。
- [ ] RewardClassifier 完成离线门禁与影子运行。
- [ ] Robometer 若用于任何训练信号，先通过校准门禁并在失败轨迹检查 progress。
- [ ] 模型 reward 的回退开关、日志和人工抽查流程可用。

## 8. 下一步

1. 用当前 ACT + VR 接管跑 R0 采集，先固定一版人工标注 holdout。
2. 实现 HIL replay importer：按接管 mask/segment 过滤并导入 SAC replay，保留人工终局 reward。
3. 在 R0/R1 数据上训练并影子评测 ResNet10 success classifier。
4. 并行离线运行 Robometer；SARM/TOPReward 在 GPU 与数据格式就绪后再纳入对比。
