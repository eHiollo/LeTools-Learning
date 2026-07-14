# Kuavo HIL-SERL VR 人工接管联调方案

> 当前状态：Quest3/IK、分臂 grip 接管、松开恢复 ACT 与双左键急停已在 MuJoCo 联调通过。真机仍须按本文完成现场验收。
>
> 适用平台：Kuavo 5W v62 + Quest3 + 现有 ROS C++ 增量 IK 链路。

## 1. 设计原则

复用 Kuavo 现有 VR 遥操作链，不在 LeTools 中重新实现 UDP、骨骼解析或 IK：

```text
Quest3
  → monitor_quest3
  → /leju_quest_bone_poses
  → wheel_ik_ros_uni_cpp_node
  → /kuavo_arm_traj
  → humanoidController_wheel_wbc
```

LeTools 只负责：

- 订阅 VR 手柄状态和 IK 输出；
- 将 14-D 手臂轨迹转换为 canonical 16-D action；
- 在人工接管时停止发布策略动作；
- 接收 success/failure/abort 事件；
- 记录 intervention、reward 和延迟。

## 2. LeTools 侧实现

相关文件：

- `kuavo_rl/ros_teleop.py`
- `kuavo_rl/teleop.py`
- `kuavo_rl/env.py`
- `kuavo_rl/adapter.py`
- `scripts/rl/eval_act_execute_first.py`
- `configs/rl/kuavo_hilserl_real_mvp.yaml`

默认接入：

| 项目 | 默认值 |
|---|---|
| 手柄话题 | `/quest_joystick_data` |
| 手臂轨迹 | `/kuavo_arm_traj` |
| 手臂轨迹输入单位 | degree |
| LeTools action 单位 | radian |
| action 顺序 | `L7,left_claw,R7,right_claw` |
| deadman | 左/右 grip 分别大于 `0.8`，按侧激活对应手臂 |
| VR 数据超时 | `0.20s` |
| 双左键 | 急停 |
| Robometer | 默认关闭 |

夹爪目前保持 reference action。必须确认 qiangnao/夹爪消息映射后，才能接入夹爪动作。

## 3. VR 联调与验收顺序

### 3.1 已完成的仿真联调

MuJoCo 联调已确认：

- `/quest_joystick_data` 与 `/kuavo_arm_traj` 持续发布；
- `/kuavo_arm_traj` 为 14-D degree，已转换为 canonical 16-D radian；
- 左/右 grip 分臂接管、松开恢复 ACT、左手双键急停均已验证；
- B 已现场确认对应 `right_second_button_pressed`，用于人工 reward 时序手势。

真机首次运行仍须执行：

```bash
rostopic hz /quest_joystick_data
rostopic hz /kuavo_arm_traj
rostopic echo -n 1 /quest_joystick_data
rostopic echo -n 1 /kuavo_arm_traj
```

并重新确认 topic namespace、字段和单位。

### 3.2 Shadow 验收（真机首次必做）

先启动 ACT 推理服务：

```bash
bash scripts/rl/run_act_infer_server.sh
```

然后运行 shadow：

```bash
PYTHONPATH=. python scripts/rl/eval_act_execute_first.py \
  --kuavo-env \
  --policy remote \
  --infer-host 127.0.0.1 \
  --infer-port 8765 \
  --deploy-config configs/deploy/total/deploy_sim_smoke_cams_total.yaml \
  --config configs/rl/kuavo_hilserl_real_mvp.yaml \
  --steps 50 \
  --shadow \
  --ros-teleop
```

shadow 模式下不得有策略动作发布。重点检查 manifest/log 中：

```text
ros_teleop: true
teleop_source: quest3_ik
teleop_age_s: <small value>
is_intervention: true/false
```

### 3.3 控制权验收

```text
未按左右 grip：不接管
按左 grip：仅左臂 VR 接管；按右 grip：仅右臂 VR 接管
松开对应 grip：取消对应手臂接管
左手两个按键同时按下：急停（沿用 Kuavo Quest3 原 FSM）
```

人工接管期间，LeTools 不发布第二路 `/kuavo_arm_traj`。

### 3.4 配置 reward 按键

先执行：

```bash
rostopic echo /quest_joystick_data
```

B 已现场确认对应 `right_second_button_pressed`。当前实际配置为：

```yaml
teleop:
  reward_button: right_second_button_pressed
  reward_double_press_s: 0.35
  reward_long_press_s: 1.20
```

人工 reward 按键约定（避开 X/A/Y、扳机、grip、摇杆与左手双键急停）：

```text
B（right_second_button_pressed）单击：success
B 双击（间隔 ≤ 0.35s）：failure
B 长按（≥ 1.2s）：abort
```

单击会延迟 0.35 秒确认，避免在双击时误写 success。现场仅需确认 B 对应
`right_second_button_pressed`；若固件字段不同，只改 `reward_button`。

reward 语义：

- success：`+1`，episode 结束；
- failure：`0`，episode 截断；
- abort：`0`，episode 截断；
- SafetyGate/急停故障：`-1`。

### 3.5 记录 HIL 介入数据

`eval_act_execute_first.py` 在 Kuavo 环境运行时默认写入：

```text
data/rl_runs/hilserl_episodes/hilserl_vr/transitions.jsonl
```

轻量审计 JSONL 每行包含：实际执行 action、ACT proposal、是否 VR 接管、VR canonical action、接管来源/延迟、手工 reward 事件、reward、故障码和时间戳。

同时写入可训练 replay：

```text
data/rl_runs/hilserl_episodes/hilserl_vr/replay/
  schema.json
  episodes/<episode_id>/transitions.jsonl
  episodes/<episode_id>/frames/*_{obs,next}_{state,camera}.npy|jpg
```

每条 replay transition 完整保存 `obs, action, reward, next_obs, terminated, truncated, is_intervention`；其中 `action` 是以实测关节状态为基准、仅覆盖 grip 接管侧、再经过 SafetyGate 限幅后的规范化标签。原始 VR IK target 只保存为 `extras.teleop_raw_action`，不直接用于训练；`extras.intervention_mask` 标明实际被人控制的关节，`intervention_segment_id/step` 用于训练时跳过接管拼接段。相机帧优先 JPEG，OpenCV 不可用时自动回退到 `.npy`。可用 `--record-dir`、`--record-experiment` 覆盖输出位置。只在确认 success/failure/abort 按键字段后，才将该数据用于 HIL-SERL replay。

## 4. 真机前必须补齐的检查

- `verify_joint_map.py --live` 确认 raw state 维度和手臂切片；
- 确认 `/kuavo_arm_traj` 14-D 顺序；
- 确认 degree/radian 没有双重转换；
- 确认 qiangnao/夹爪消息与 16-D action 的夹爪索引；
- 确认 VR IK 和策略不会同时产生有效控制；
- 确认急停、断联、陈旧观测、ROS shutdown；
- shadow 通过后才允许低速短时 ACT；
- Robometer 3.4 门禁通过前保持在线关闭。

## 5. 当前状态与剩余工作

VR 接管与 HIL replay 存储已在 MuJoCo 跑通；当前不再存在 VR 设备阻塞。尚未完成的是：

- 真机 shadow、joint-map/单位/夹爪映射验收；
- 用正式采集的 replay 导入 HIL-SERL online replay buffer，并以 `intervention_mask` 跳过接管拼接段；
- 用人工 reward 采集有效 success/failure 数据后训练、评估 SAC；
- Robometer 人工标注集与 3.4 离线校准门禁。
