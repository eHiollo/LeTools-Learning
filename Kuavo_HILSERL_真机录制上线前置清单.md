# Kuavo HIL-SERL 真机录制上线前置清单

> 依据：`Kuavo_HILSERL_KuavoBrain数据采集接入方案.md`（v2）P4 + §12  
> 状态：代码 P0–P3 已落地（dry-run / 单测通过）；**真机尚未放行**  
> 用途：后面一起做真机联调时按本清单逐项勾选，全部通过后再开始 R0 reward 采集。

---

## 0. 当前结论（先读）

| 项 | 状态 |
|---|---|
| `kuavo_rl/hil_recording/` session / gate / watchdog / staging 发布 | ✅ 本地单测通过 |
| dry-run 假 rosbag 路径 | ✅ 可用 |
| 真实 `rosbag record` + ROS topic gate | ⬜ 未在现场验证 |
| `/hil/transition_audit`、`/hil/result_event` 进真 bag | ⬜ 未在现场验证 |
| VR / B / 急停 + 录制状态同步 | ⬜ 未在真机验证 |
| 下游只读 `accepted_replay/` | ⬜ 训练加载器尚未强制 |
| R0 reward 真机采集 | 🚫 未放行 |

**原则**：仿真 / 影子未过 → 禁止 policy 下发；短 episode 故障注入未过 → 禁止开始正式采集。

---

## 1. 现场环境与硬件（到场先做）

### 1.1 人与安全

- [ ] 至少两人：安全操作者 + 监控记录者
- [ ] 急停物理按钮可用，左手双键急停路径已口头复述
- [ ] 工作空间清空；无损任务物体；底盘/腿部不下发
- [ ] 约定最大 session 时长、最大步数、最低磁盘剩余

### 1.2 机器与 ROS

- [ ] 机器人型号 / eef（如 `leju_claw`）与 `configs/rl/hil_topics_v002.yaml` 一致
- [ ] `roscore` / 机器人 bringup 正常
- [ ] `use_sim_time` 状态明确（真机一般 `false`）
- [ ] 三路相机 H.265 有流；夹爪 / 关节 state 有流
- [ ] Quest3 VR + IK 链路已按 `Kuavo_HILSERL_VR人工接管联调方案.md` 通过（若跑 `act_vr`）
- [ ] 磁盘：录制盘用量 < `start_block`（默认 90%）；预留至少一场 bag 空间

### 1.3 软件环境

- [ ] 使用 `letools-rl`（或现场约定环境），可 `import kuavo_rl.hil_recording`
- [ ] `rosbag` 命令在 PATH 中：`which rosbag`
- [ ] 仓库未改 `third_party/kuavo_brain`（只读边界）
- [ ] 启动前执行恢复：  
  `python -m kuavo_rl.hil_recording.cli --root data/rl_runs/hilserl_episodes/hilserl_vr recover`

---

## 2. Topic / 契约现场冻结（不下发策略动作）

模板见 `configs/rl/hil_topics_v002.yaml`。**禁止只靠模板开录**，现场必须 `rostopic list` 对照后改配置或 profile。

### 2.1 必勾

- [ ] `scripts/rl/verify_joint_map.py`：28-D → `[12:26]`、夹爪量级与数据集一致
- [ ] `scripts/rl/preflight.py`：数据集 / 观测契约无阻断项
- [ ] 对每个 `required_for_start: true` 的 streaming topic 记录实测 Hz
- [ ] `/tf_static` 确认为 latched（收到过即可，不做 Hz 检查）
- [ ] 纯 ACT profile：确认不强制 `/kuavo_arm_traj`、IK topics
- [ ] `act_vr` profile：确认 VR/IK 相关 topic 在接管时存在
- [ ] 控制 topic 发布者列表：登记 ACT/teleop 为 producer；无未登记外部控制节点

### 2.2 建议记录表（贴到实验笔记）

| topic | mode | 实测 Hz / 是否 latched | 是否改阈值 | 备注 |
|---|---|---|---|---|
| /sensors_data_raw | streaming | | | |
| /joint_cmd | streaming | | | |
| /tf | streaming | | | |
| /tf_static | latched | | | |
| /leju_claw_state | streaming | | | |
| /cam_*/color/h265_stream | streaming（与 v3 统一；观测/导出亦走此路，非 JPEG） | | | |
| /kuavo_arm_traj | streaming (vr) | | | |
| /hil/transition_audit | audit | 本系统发 | | start 时可不存在 |
| /hil/result_event | audit | 本系统发 | | |

改完阈值后更新 `hil_topics_v002.yaml` 或现场副本，并记下 `topics_version`。

---

## 3. 上真机前还要补的软件项

代码骨架在，下列是**真机打开 live rosbag 前**建议补齐/确认的点：

### 3.1 必须确认

- [ ] Gate 真机路径：`skip_gate_ros=False` 时 ROS master / topic 存在 / latched 检查真实可用
- [ ] streaming 频率：注入缓存 rate provider（禁止每 500ms 重型 `rostopic hz`）
- [ ] `AuditPublisher` 在已有 ROS node 下不重复 `init_node` 冲突
- [ ] 真 bag 内能用 `rosbag info` / `rostopic echo -b` 看到 `/hil/transition_audit`、`/hil/result_event`
- [ ] sidecar 步数与 bag 内 audit（或 mirror）比对阈值在现场可接受
- [ ] `HILReplayWriter(staging_dir=...)` 与 session 目录一致；训练侧只扫 `accepted_replay/`
- [ ] obs `header.stamp` → `source_header_stamp_ns` 已从 env/backend 传到 `update_transition`（对齐用 ROS 时间）
- [ ] post-roll（默认 0.5s）后，terminal event 之后仍有 state/图像进 bag

### 3.2 建议补强（可同一天做）

- [ ] 外部控制节点探测：未登记却发布 `/joint_cmd` 时 Gate Block
- [ ] 训练加载器硬编码只读 `accepted_replay/`（读到 staging/quarantine 直接报错）
- [ ] MuJoCo/ROS 仿真先跑一遍 live 路径（`--hil-recording-live-rosbag`）再上真机
- [ ] 影子模式：`--shadow` + 录制，确认不发布 policy action 仍能完整落盘

---

## 4. 推荐上线顺序（一起做时按天推进）

### Day A：仿真 / 本地 live rosbag（仍可不碰真机）

```bash
# 单测回归
python -m pytest kuavo_rl/tests/test_hil_recording.py -v

# MockBackend 闭环（不依赖 Kuavo-Sim 相机；推荐先跑）
python scripts/rl/verify_hil_recording_sim.py \
  --root data/rl_runs/hilserl_sim_verify/hilserl_vr \
  --steps 8

# Kuavo-Sim + hil recording（先 dry-run；需相机流 + ACT infer:8765）
# 环境参考 scripts/rl/run_act_kuavo_sim_eval.sh
python scripts/rl/eval_act_execute_first.py \
  --kuavo-env --hil-recording \
  --policy remote --infer-port 8765 \
  --steps 20

# 有 ROS 时再开真 rosbag（仿真机）
python scripts/rl/eval_act_execute_first.py \
  --kuavo-env --hil-recording --hil-recording-live-rosbag \
  --steps 20
```

通过标准：

- [x] 单测 `test_hil_recording` 通过（2026-07-16）
- [x] MockBackend 闭环：`Finalized(Healthy)` + `accepted_replay` + `hil-replay-v002`（见 `data/rl_runs/hilserl_sim_verify/sim_verify_report.json`）
- [x] Kuavo-Sim 原生相机 + hil-recording dry-run（2026-07-17）：  
  ROS 启动 `ROBOT_VERSION=62` + `load_kuavo_mujoco_sim_wheel.launch publish_camera:=true`；  
  deploy=`configs/deploy/total/deploy_sim_mujoco_native_cams.yaml`；  
  结果 `Finalized(Healthy)` + `Published` → `data/rl_runs/hilserl_sim_verify/manifest_kuavo_dryrun.json`
- [x] live rosbag（2026-07-17）：`HIL_LIVE_ROSBAG=1` + `hil_topics_sim_v002.yaml`；  
  `Finalized(Healthy)` + `Published`；bag ≈163MB  
  `sessions/ep_1784255857/bags/original.bag`；manifest=`manifest_kuavo_live.json`

### Day B：影子真机（不下发或仅安全保持）

- [ ] joint map / preflight 通过
- [ ] topic 表冻结
- [ ] `--shadow --hil-recording`（或现场等价影子配置）短跑
- [ ] 急停演练：控制停、录制走完 post-roll、数据进 quarantine
- [ ] kill runner 后 `cli recover`，无残留 Recording、无孤儿 rosbag

### Day C：受控真机短 episode（低速、少步数）

```bash
python scripts/rl/eval_act_execute_first.py \
  --kuavo-env --ros-teleop \
  --hil-recording --hil-recording-live-rosbag \
  --steps 30 \
  --config configs/rl/kuavo_hilserl_real_mvp.yaml
```

约束：低速、固定工作空间、安全员在场、每场限定步数。

---

## 5. 真机故障注入矩阵（§12.2 精简勾选版）

每项保存：`session.json`、`gate.json`、`watchdog.report.json`、`quality_report.json`、SQLite 行、bag、publish/quarantine 决定。

| # | 场景 | 预期 | 过 |
|---|---|---|---|
| 1 | ACT + VR + B success | Healthy + `accepted_replay`，result=success | ☐ |
| 2 | ACT 运行中开录 | Gate 放行（producer 已登记） | ☐ |
| 3 | B 双击 failure | result=failure，数据可发布，非 record_error | ☐ |
| 4 | B 长按 abort | result=abort，无 success reward | ☐ |
| 5 | 左手双键急停 | 立即停控；post-roll 有最终 state；quarantine；不进 accepted | ☐ |
| 6 | 终止后看最后帧 | bag 内存在晚于 terminal ROS 时间的 state/图像 | ☐ |
| 7 | 杀掉 rosbag | Failed(record_error)，staging→quarantine | ☐ |
| 8 | kill runner 再启动 | `recover` 清理；可续录新 episode | ☐ |
| 9 | 拔掉/停核心 streaming topic | watchdog → StopRequest → controller 停 | ☐ |
| 10 | `/tf_static` 无新消息 | 不误报 stale | ☐ |
| 11 | 纯 ACT 无 arm_traj | Gate 不因缺 VR topic Block | ☐ |
| 12 | bag 卡住不增长 | record_error + watchdog 证据 | ☐ |
| 13 | 磁盘逼近 hard-stop | 停录；quarantine | ☐ |
| 14 | 重复 STOP | 幂等，不双重 finalize | ☐ |
| 15 | stop 后立刻 publish | 拒绝；须 `wait_finalized` | ☐ |
| 16 | sidecar / audit 步数不一致 | Failed(self_check)，不发布 | ☐ |
| 17 | 连续两个 episode | 不继承旧 PID / result / producer | ☐ |

**放行**：上表 1–8、14–15、17 必须过；其余尽量过，未过项写明风险后再决定是否采集。

---

## 6. 数据落盘检查（每场结束后）

根目录默认：`data/rl_runs/hilserl_episodes/hilserl_vr/`

```text
hil_recording.db
sessions/<episode_id>/     # bag、gate、watchdog、staging 残留应为空或已搬迁
accepted_replay/<id>/      # 仅质量合格可训练数据
quarantine/<id>/           # 急停 / 失败 / record_error
```

- [ ] 可训练数据只在 `accepted_replay/`
- [ ] 急停只在 `quarantine/`，未删除
- [ ] `publish_manifest.json` 含 `hil-replay-v002` 与 quality hash
- [ ] `session.json` 仅为快照，未当写入口
- [ ] raw VR ≠ 训练 action；mask / segment / SafetyGate 后 action 可追溯

CLI：

```bash
python -m kuavo_rl.hil_recording.cli --root data/rl_runs/hilserl_episodes/hilserl_vr list-active
python -m kuavo_rl.hil_recording.cli --root data/rl_runs/hilserl_episodes/hilserl_vr show <episode_id>
```

---

## 7. 明确不做 / 未放行

- 不在真机上首次验证 gRPC、依赖安装、reward 标定
- 不上传公司云 / NAS（本方案本地 only）
- 不修改 `third_party/kuavo_brain`
- 未过本清单前不做 R0 正式 reward 采集、不做 SAC 真机闭环
- 阿里云 ECS 不直连控制真机

---

## 8. 一起开工时的最小议程（建议 2–3 小时）

1. 环境 + recover + topic 表（30 min）  
2. 影子 / 短 episode 正常 success 一场（30 min）  
3. 急停 + kill recorder + recover 三场（45 min）  
4. B failure/abort + 连续两 episode（30 min）  
5. 勾选矩阵、归档证据、决定是否开 R0（15 min）

归档目录建议：`data/rl_runs/hilserl_acceptance/<date>/`（复制各场 `sessions/` 证据 + 本清单勾选结果）。

---

## 9. 相关文件

| 文件 | 用途 |
|---|---|
| `Kuavo_HILSERL_KuavoBrain数据采集接入方案.md` | 架构与状态机规范 |
| `kuavo_rl/hil_recording/` | 实现代码 |
| `configs/rl/hil_topics_v002.yaml` | topic profile |
| `configs/rl/kuavo_hilserl_real_mvp.yaml` | 真机 MVP 配置 |
| `scripts/rl/eval_act_execute_first.py` | `--hil-recording` 入口 |
| `Kuavo_HILSERL_VR人工接管联调方案.md` | VR / B / 急停 |
| `Kuavo_LeRobot_HILSERL_实机闭环RL工程部署手册.md` | 总阶段与安全门 |
