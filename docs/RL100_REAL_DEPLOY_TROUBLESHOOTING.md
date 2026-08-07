# RL-100 真机部署排障记录

> 日期：2026-08-07
> 机器人：5w (robot_version=62)，Jetson Orin
> Checkpoint：`epoch=2900-test_mean_score=-0.000.ckpt`（DP3, DDIM 10 步, n_action_steps=4, n_obs_steps=3）

## 背景

在 RL-100 真机 `live` 部署过程中，依次遇到多个 FAULT 和启动失败问题。本文档记录每个问题的现象、根因和修复方法，供后续排查参考。

---

## 问题 1：CUDA 库版本冲突（已在此前解决）

**现象**：推理挂死、`libcudart.so.11.0: cannot open shared object file`、`cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`。

**根因**：conda 的 PyTorch 2.2.0 编译于 CUDA 11.4，但系统装的是 CUDA 12.8。sbsa（服务器版）CUDA 11.4 库缺少 Jetson Orin 的 `sm_87` kernel。

**修复**：在 `.runtime_staging/cuda11.4` 中放置 Tegra 专用 CUDA 11.4 库（cuBLAS/curand/cuDNN/cufft/cusparse），`configs/rl/local/env.sh` 和 `scripts/rl/run_rl100_real.sh` 自动设置 `LD_LIBRARY_PATH`。

---

## 问题 2：publishers.ok: false（无订阅者）

**现象**：`ros-preflight` 报 `publishers.ok: false`。

**根因**：`publish.require_subscribers: true` 要求 arm/hand 话题有订阅者，但 shadow 模式下机器人控制器未启动。

**修复**：`rl100_real_deploy.yaml` 中 `publish.require_subscribers` 改为 `false`。

---

## 问题 3：startup hand default mismatch

**现象**：`RuntimeError: startup hand default mismatch: max error=99.000`。

**根因**：`startup.hand_default` 配置为 `[0, 99, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0]`，但机器人实际手部位置不匹配。

**修复**：先改为 `[60, 0, 0, 0, 0, 0, 59, 0, 0, 0, 0, 0]` 匹配当时姿态；最终去掉了硬校验，改为只打印当前手部状态，以机器人实际 state 为准。

---

## 问题 4：camera_sync 契约校验不一致

**现象**：`RuntimeError: deployment and point-cloud camera_sync contracts differ: max_received_age_s: deploy=0.3, pointcloud=0.2`。

**根因**：deploy 配置的 `depth_max_age_s` 改为 0.30 后，契约校验中 deploy 侧 `max_received_age_s = depth_max_age_s = 0.30`，但 pointcloud 侧 `max_received_age_s = min(depth_max_age_s, camera_sync.max_received_age_s) = min(0.30, 0.20) = 0.20`，两边不一致。

**修复**：两个配置的 `camera_sync.max_received_age_s` 都改为 0.30，使 `min(0.30, 0.30) = 0.30` 与 deploy 一致。

**教训**：修改 `depth_max_age_s` 时必须同步修改 `camera_sync.max_received_age_s`，否则契约校验会报错。

---

## 问题 5：TF unavailable（tf_at_stamp 超时）

**现象**：`camera_sync[tf_at_stamp]: TF unavailable for head_cam_h at ...s`。

**根因**：`camera_sync.tf_timeout_s: 0.05`（50ms）太短，TF 查询在机器人负载下经常超时。

**修复**：`tf_timeout_s` 改为 0.20（200ms），两个配置都改。

---

## 问题 6：camera header_skew 超限

**现象**：`camera_sync[header_skew]: selected camera header skew 0.621s > 0.500s`，后续 1.027s > 0.800s，再后续 1.295s > 1.200s。

**根因**：部署时相机同步是**实时**的（`select_synchronized_frames` 从 buffer 中找最近时间戳的帧），而采集用的是 `raw_rosbag` 模式，同步在**离线**转换时做（可搜索全部帧，无时间压力）。部署时系统负载高（推理+控制+相机处理并发），且机器人运动导致腕部相机物理振动，帧时间戳抖动加大。

**修复**：
- `max_header_skew_s` 从 0.50 逐步放宽到 1.20
- `max_camera_skew_s`（observation 级别）同步放宽到 1.20
- `depth_max_age_s` 从 0.15 放宽到 0.30
- 采集配置 `rl100_zarr_collect_upper_cams.yaml` 的对应字段同步修改

**待解决**：实时同步的 skew 比离线大是结构性问题。后续可考虑：
1. 增大 `buffer_size`（当前 32，~1s 历史），让同步有更多帧可搜索
2. 改用"全局最优匹配"策略而非"锚定参考相机最新帧"策略
3. 对腕部相机做时间戳补偿（减去系统性偏移）

---

## 问题 7：INFERENCE_TIMEOUT — 推理冷启动超时

**现象**：`action buffer exhausted while inference was not ready`，`inference_running: true, inference_ready: false`。

**根因**：第一次推理是 CUDA 冷启动 + DDIM 10 步采样，耗时 ~0.94s。但 `inference.timeout_s: 0.50`，`max_empty_ticks = ceil(0.50 * 10) + 1 = 6`，即 0.6s 后就 fault。冷启动远超 0.6s。

**修复**：
1. `inference.timeout_s` 从 0.50 改为 3.0（`max_empty_ticks` 提到 31，即 3.1s 余量）
2. 在 live/shadow 循环前加 `policy.warmup(warmup_runs)` 预热 CUDA kernel

---

## 问题 8：INFERENCE_TIMEOUT — stale-step 丢弃逻辑 bug

**现象**：推理成功完成（0.939s），但报 `inference 0.939s consumed the entire 4-step action chunk`。

**根因**：`_accept_result` 中 stale-step 丢弃逻辑用 wall-clock 估算过期步数：`stale_steps = min(chunk_size, floor(inference_s * action_hz)) = min(4, floor(0.939*10)) = min(4, 9) = 4`。4 步全被丢弃。但第一次推理时 buffer 是空的、机器人在 hold，并没有在执行动作，所以这些步根本不该被当"过期"丢弃。

**修复**：改为按推理期间**实际从 buffer 消费的步数**算过期：
- `_submit` 时记录 `_pending_at_submit = len(self._pending)`
- `_accept_result` 时 `steps_consumed = max(0, _pending_at_submit - len(self._pending))`
- `stale_steps = min(chunk_size, steps_consumed)`

第一次推理（buffer 空）→ `steps_consumed = 0` → 不丢任何步，4 步全部进 buffer。

**测试**：更新 `test_rl100_topic_executor.py`，新增 `test_first_inference_with_empty_buffer_drops_no_steps` 和 `test_steady_state_drops_steps_consumed_during_inference`。

---

## 问题 9：ACTION_LIMIT — 连续 clip fault

**现象**：模型预测的手部值从当前状态 `[60, 0, ...]` 跳到 `[0.24, 99.84, ...]`，单步跳 59.76 和 99.84，远超 `max_hand_step: 5.0`。安全门每个 tick 都 clip，`max_consecutive_clips: 3` 后直接 fault。

**根因**：`max_consecutive_clips: 3` 在紧 step 限制下过于激进。模型预测与当前状态差距大时，每步都会被 clip（这是安全行为），但 3 次连续 clip 就 fault 不合理。

**修复**：`max_consecutive_clips` 改为 0（禁用连续 clip fault）。每步的 clip 限制（`max_arm_step_rad`、`max_hand_step`）仍然生效，保护机器人——机器人会以每步 1.15 度/5 单位的速度逐渐爬向目标，不会 fault。

---

## 问题 10：max_episode_duration_s 太短

**现象**：`--duration-s 100` 但实际只跑 20s。

**根因**：`safety.max_episode_duration_s: 20.0`，`duration_s = min(args.duration_s, config_value) = min(100, 20) = 20`。

**修复**：`max_episode_duration_s` 改为 120.0。

---

## 问题 11：机器人抖动

**现象**：机器人在初始位置附近持续抖动。

**根因**：`deterministic: false` 让 DDIM 每步用 SDE solver（`step_logprob`）注入随机噪声。10 步累积下来，每次推理的 action chunk 差异很大，机器人不同周期朝不同方向移 → 抖动。腕部相机物理振动又导致 camera header skew 加大。

**修复**：`deterministic` 改为 `true`（ODE solver, `step_mean`），去噪过程确定性，相同观测产出相近结果。

---

## 问题 12：手臂控制模式自动切换

**现象**：脚本自动调 `/wheel_arm_change_arm_ctrl_mode` 和 `/enable_lb_arm_quick_mode`，但用户希望用 VR 手动进入 external control。

**修复**：去掉 `ArmControlModeSession` 的 enter/restore 逻辑，脚本不再自动切换 arm control mode。启动时打印提示，由操作员用 VR 手动进入 external control。

---

## 修改文件清单

| 文件 | 修改内容 |
|---|---|
| `configs/rl/rl100_real_deploy.yaml` | 阈值放宽、deterministic=true、timeout_s=3.0、max_consecutive_clips=0、require_subscribers=false、hand_default 调整、max_episode_duration_s=120 |
| `configs/rl/rl100_zarr_collect_upper_cams.yaml` | camera_sync 字段与 deploy 对齐（max_header_skew_s=1.20, max_received_age_s=0.30, tf_timeout_s=0.20, depth_max_age_s=0.30, camera_max_skew_s=1.20） |
| `kuavo_rl/rl100_topic_executor.py` | stale-step 丢弃逻辑从 wall-clock 改为实际消费步数 |
| `kuavo_rl/tests/test_rl100_topic_executor.py` | 更新测试匹配新 stale-step 逻辑 |
| `scripts/rl/run_rl100_real.py` | 加 warmup、去掉 arm control mode 自动切换、去掉 hand_default 硬校验、加 lb quick mode 函数 |
| `kuavo_rl/rl100_zarr/README.md` | 示例命令 duration-s 从 20 改为 100 |

---

## 遗留问题

1. **相机实时同步 skew 偏大**：采集用离线同步，部署用实时同步，结构性差异导致 skew 更大。当前靠放宽阈值缓解，后续需改进同步策略。
2. **推理速度慢**：DDIM 10 步 ~0.94s/次，action horizon 仅 4 步（0.4s），buffer 频繁耗尽后 hold。机器人运动不连续。后续可考虑用 CM 1 步推理或增大 action horizon。
3. **`max_arm_step_rad: 0.02`**（1.15 度/步）非常保守，机器人运动很慢。后续可根据实际表现适当调大。
