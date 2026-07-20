# Kuavo HIL-SERL 数采实施问题清单

> 推进 C1→C6 时遇到的阻塞、偏差与待确认项。睡醒后优先看本节。

## 状态（2026-07-17）

| 阶段 | 状态 | 说明 |
|---|---|---|
| C0 | 完成 | 模型/DB v002/PendingReview/CLI 骨架 |
| C1 | 代码完成（单测） | Quest 事件源 + 校准 override + 独占 Gate；未接真机摇杆联调 |
| C2 | 部分 | dry-run OK；**`collect --vr-sim` 已接 VR 示教环**；ACT live 仍未接 |
| C3 | 部分 | label/review/inspect/--report；双人复核策略已有开关 |
| C4 | 部分 | `batch --dry-run` 循环 + rerecord 重试；无 live Quest/reset hook |
| C5–C6 | 未做 | live rosbag / 真机 |

## 已记录问题

### P1 — 右摇杆冲突 → 默认改为「按住 Y + 摇杆」

- 确认：裸右摇杆与腰部/底盘冲突。
- **当前默认** `episode_control: quest_y_stick`：
  - 按住 `Y` + `→/←/↓` = 开始·提前结束 / 重录 / 结束采集。
  - 未按 `Y` 时不消费摇杆；不要求 `collection_mode_ack`。
- 备用：`quest_y_chord`（纯按键）；遗留：`quest_right_stick`（需独占）。
- 残留：按住 `Y` 推杆瞬间旧节点仍可能读轴 → 操作上推完即松 `Y`。

### P2 — Quest 事件源与 `RosTeleopAdapter` 双订阅

- 仍双订阅 `/quest_joystick_data`；Y 组合事件源不发 action。
- 默认不再抢右摇杆；腰部/底盘可继续用摇杆。
- 注意：勿把 `Y+A` 误当成 `X+A`（遥操作激活）。

### P3 — Live ACT collect 未接线（VR 示教已接）

- **已接**：`collect --vr-sim --confirm-live` → `kuavo_rl/hil_collect_live.py`
  - `vr_only` + `shadow_mode`（不抢发 ACT；Quest IK 驱动）
  - RosTeleop + HIL session → `pending_review`
  - `Y + 摇杆` episode 控制
- **未接**：`act` / `act_vr` 的 live ACT runner。
- **2026-07-17 试跑**：脚本已进 Kuavo-Sim，但 `obs_buffer` 一直 `0/15`（`/camera/*` 与 joint 无新帧），服务 `/humanoid_get_arm_ctrl_mode` 超时 → 环境侧需先恢复仿真再示教。

### P4 — 交互式摇杆校准未做 ROS 采样 UI

- `calibrate-stick` 支持 `--manual` 与 `--tip-right/--tip-down` 样本推断。
- 未做「现场推摇杆自动采样」交互（需 ROS + 提示词循环）。
- 临时：操作者读一次 raw 轴，把数值传给 CLI。

### P5 — `act_vr` profile 在 dry-run 中降级为 `act`

- topic resolve 对 `act_vr` 可能要求更多 topic；dry-run 为避 Gate，create 时用 `act`。
- 真采时应按配置 `act_vr` + 完整 profile。

### P6 — 方案文档 §13 仍写「仅 C0」

- 实施已进入 C1/C2/C3；文档允许改动列表未更新。
- 非阻塞；睡醒后可补一节 C1+ 允许文件列表。

### P7 — 下一步建议

1. 恢复仿真健康：`rostopic hz /camera/color/image_raw`、`/sensors_data_raw` 有数据后再跑 VR 示教。
2. VR 示教：`bash scripts/rl/run_collect_vr_sim.sh 120`（Quest 已开、握 trigger/grip 动臂）。
3. C5：`LIVE_ROSBAG=1` 对账 bag `/hil/*`。
4. ACT live（`act_vr`）可后置。

## 本地验证命令

```bash
# 单测（当前 25 passed）
python -m pytest \
  kuavo_rl/tests/test_hil_collection.py \
  kuavo_rl/tests/test_hil_recording.py \
  kuavo_rl/tests/test_quest_episode_control.py -v

# dry-run 采集 → pending_review
python scripts/rl/collect_hil_dataset.py \
  --config configs/rl/hil_collection_sim_v001.yaml \
  collect --dry-run --max-steps 8 --operator fulin

# 仿真 VR 示教（默认：live bag + B 结束 + 自动 Brain→LeRobot v3）
bash scripts/rl/run_collect_vr_sim.sh       # 最多 50 条
bash scripts/rl/run_collect_vr_sim.sh 10    # 最多 10 条
# 流程：Reset → Y+→ 开始 → 握 grip 示教 → B结束
#   成功/失败 → accepted_replay + 自动 CvtRosbag2Lerobot → lerobot_v3/
# LIVE_ROSBAG=0 可关真 bag（则不会出 LeRobot）
# 2026-07-17: hil_topics_sim_v002 已按参考 bag 补录 /kuavo_arm_traj（Brain 必需）
#   参考: /home/fulin/visuals_joint_traj_accelerate/A10-A15-I-L-05-TQ_01_01-5W_59-leju_claw-...-v003.bag

## 2026-07-17 数据检查（15:07 左右两条约 300 step）
- `…070721…` / `…070805…`：**不能用** — quality 因缺 `intervention_segment_step` 失败进 quarantine；二次 quarantine 又把 staging（帧+transitions）清空，只剩 audit/json。
- 根因已修：采集写入补字段；quarantine 仅在 staging 有真实 payload 时才 move，避免空 staging 覆盖。
- 需重采；旧条不可恢复图像。

# dry-run batch
python scripts/rl/collect_hil_dataset.py \
  --config configs/rl/hil_collection_sim_v001.yaml \
  batch --dry-run --episodes 3 --max-steps 4 --operator fulin

# 离线标注
python scripts/rl/collect_hil_dataset.py inspect --pending-review
python scripts/rl/collect_hil_dataset.py label <eid> --label success --reason ok --labeler a
python scripts/rl/collect_hil_dataset.py review <eid> --approve --reviewer a --publish-accepted
python scripts/rl/collect_hil_dataset.py inspect --report

# 摇杆校准（写入 configs/rl/local/，已 gitignore）
python scripts/rl/collect_hil_dataset.py calibrate-stick --manual --operator fulin
python scripts/rl/collect_hil_dataset.py preflight --for-live-collect
```
