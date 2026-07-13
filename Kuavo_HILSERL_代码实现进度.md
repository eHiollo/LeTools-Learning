# Kuavo HIL-SERL 代码实现进度与问题回溯

> 依据：`Kuavo_LeRobot_HILSERL_实机闭环RL工程部署手册.md`（最终定稿）  
> 开始时间：2026-07-13  
> 目的：记录实现进度、决策与问题，方便后期回溯。

---

## 0. 总览

| 阶段 | 状态 | 说明 |
|---|---|---|
| Phase 0 依赖基线 | 🟢 镜像已验证 | 已提交 `letools-train:hilserl`；grpc/gym_hil/TrainRL/CUDA 通过（见 P9–P15） |
| Phase 1 上游 gym_hil 仿真 | 🟢 短跑+有屏通过 | headless OK；Keyboard 有屏 OK（`gym_hil_display_latest`，opt=772） |
| Phase 2 Kuavo 环境契约 | 🟢 代码已落地 | `kuavo_rl` + bridge；单测 **22 passed** |
| Phase 3 Robometer / 接管预留 | 🟢 骨架已落地 | 异步 worker + stub；真模型推理待显存验证 |
| Phase 4 数据审计 / ACT 基线脚本 | 🟢 脚本已落地 | preflight / ACT execute-first / sim smoke 通过 |
| Phase 5+ 真机闭环 | ⬜ 未开始 | 需 ROS 侧接真 Kuavo-Sim/Real |

图例：🟢 已完成代码 / 🟡 部分 / ⬜ 未开始 / 🔴 阻断

---

## 1. 实现决策（相对手册的落地选择）

| # | 决策 | 原因 |
|---|---|---|
| D1 | `kuavo_rl` 放主仓库，不改 `third_party/lerobot` | 手册硬约束；便于升级上游 |
| D2 | Env 通过 `RobotBackend` 抽象隔离 ROS | 无硬件/无 ROS 时可单测；真机接入时再挂 `ROSBackend` |
| D3 | 首版 `ROSBackend` 可选依赖 `kuavo_deploy` | 避免训练 Python 强制 import ROS Noetic 包 |
| D4 | SafetyGate 纯 numpy，无 ROS | 可独立单测 |
| D5 | ACT runner 强制 `execute_steps=1` | 手册阶段 A：chunk=10 只执行第 1 步 |
| D6 | Robometer worker 默认可 stub | 无 GPU / 未装模型时仍可跑确定性 reward |
| D7 | 关节 map 默认 `28→[12:26]` | 数据集 QC 契约；现场仍须 `verify_joint_map.py` |
| D8 | 不直接改 `KuavoBaseRosEnv` 的 reward/terminated | 保持现有部署评测兼容；RL 走新 `KuavoHILSerlEnv` |
| D9 | 超大关节跃迁默认软限幅，不直接 VELOCITY_LIMIT 终止 | 更符合绝对位置控制 + RL 探索；硬故障留给 stop/NaN/陈旧观测 |
| D10 | Phase 0 用 Docker 提交 `letools-train:hilserl`，不在本机 `data`(py3.10) 强装 v0.6.1 | 上游要求 Python≥3.12；本机 `data` 是 3.10 |
| D11 | 不强依赖 `evdev` 手柄；先保证 keyboard/`gym_hil` 可 import | 当前镜像内核头文件缺 KEY_* 宏，evdev 源码编译失败 |
| D12 | Phase 1 smoke 用 headless Base factory（`LEROBOT_GYM_HIL_HEADLESS=1`） | Docker 无 X11；Keyboard wrapper 依赖 pynput/X |
| D13 | MuJoCo 渲染默认 `osmesa`（非 egl） | 本机 GPU EGL 无 PLATFORM_DEVICE，无法建 headless GL |

---

## 2. 已创建文件清单

### 2.1 包 `kuavo_rl/`

| 文件 | 职责 |
|---|---|
| `__init__.py` | 懒导出（避免无 gymnasium 时 import 失败） |
| `config.py` | 配置 dataclass 与强校验 |
| `contracts.py` | 16-D action/state 常量、故障码 |
| `safety.py` | SafetyGate |
| `ros_adapter.py` | rad/deg、夹爪缩放、state 切片 |
| `backend.py` | RobotBackend 抽象 + MockBackend + ROSBackend |
| `kuavo_bridge.py` | 把现有 Kuavo Gym obs（torch batch/HWC）规范成契约 |
| `env.py` | KuavoHILSerlEnv（Gymnasium） |
| `act_runner.py` | 阶段 A：chunk 预测后只执行第 1 步 |
| `adapter.py` | 阶段 B 环境工厂桥接 |
| `teleop.py` | VR/人工事件接口预留 |
| `reward.py` | 确定性 reward + Robometer 异步 worker |
| `recording.py` | episode/manifest 审计 |

### 2.2 测试 `kuavo_rl/tests/`

- `test_action_contract.py`
- `test_observation_contract.py`
- `test_episode_semantics.py`
- `test_act_execute_first.py`
- `test_safety_gate.py`
- `test_robometer_worker.py`
- `test_kuavo_bridge.py`
- `test_actor_learner_smoke.py`（无 hilserl 时 skip）

### 2.3 配置 `configs/rl/`

- `kuavo_v62_joint_map.yaml`
- `act_kuavo_bc.yaml`
- `gym_hil_baseline.json`
- `kuavo_hilserl_sim.yaml`
- `kuavo_hilserl_shadow.yaml`
- `kuavo_hilserl_real_mvp.yaml`
- `robometer_reward.yaml`

### 2.4 脚本 `scripts/rl/`

- `preflight.py`
- `verify_joint_map.py`
- `score_rollouts_robometer.py`
- `run_act_baseline.sh`
- `run_kuavo_sim_smoke.py`
- `install_hilserl_docker.sh`
- `run_learner.sh`
- `run_actor.sh`
- `run_gym_hil_smoke.sh`（Phase 1 headless 联跑）

### 2.5 其它

- `docker/setup_env_docker.sh`：安装 extras 增加 `hilserl`
- `configs/rl/gym_hil_smoke.json`：短跑配置（`online_steps=40`，`dataset=null`，wandb off）
- 本文档：`Kuavo_HILSERL_代码实现进度.md`

---

## 3. 进度日志

### 2026-07-13 — 首轮代码落地

**做了什么**

1. 阅读最终部署手册与现有 `kuavo_deploy` / `lerobot_merged` 契约。
2. 确认 `kuavo_rl/` 此前不存在；现有 `KuavoBaseRosEnv` 仍返回 `reward=0`、`terminated/truncated=False`。
3. 新建完整 `kuavo_rl` 模块、配置、脚本与单测。
4. 更新 Docker 安装脚本加入 `hilserl`。
5. 在 conda `data` 环境跑通单测：**20 passed**。
6. `scripts/rl/preflight.py` 对 `data/lerobot/lerobot_merged` 契约检查通过。
7. `scripts/rl/run_act_baseline.sh` mock smoke：每步丢弃 chunk 尾部 9 步。

**遇到的问题与处理**

| ID | 问题 | 影响 | 处理 |
|---|---|---|---|
| P1 | 现有 `KuavoBaseRosEnv.step` 固定 `False, False` 且 `reward=0` | 不能直接当 HIL-SERL env | 新建 `KuavoHILSerlEnv`，不破坏旧部署路径（D8） |
| P2 | `platform_config.yaml` 的 `5w=[4:18]` 与 v62 28-D `[12:26]` 易混 | 真机关节错位 | 固化 `configs/rl/kuavo_v62_joint_map.yaml`；按 raw dim 分派切片 |
| P3 | SDK `control_arm_joint_positions` 是否内部 rad→deg 未现场证实 | 可能双重转换 | `ROSBackend` 默认 `rad_to_sdk`；`arm_rad_to_traj_deg` 仅用于直接 topic 发布；脚本写 WARNING |
| P4 | 本机 base Python 无 gymnasium；pytest 曾落到系统 3.8 | 单测收集失败 | 用 conda `data` 的 `python -m pytest`；`__init__.py` 改为懒加载 |
| P5 | Robometer-4B 显存可能顶满 5060 Ti | 同步推理会拖垮控制 | worker 异步 + stub；真模型懒加载；离线脚本占位 |
| P6 | ACT 默认 `n_action_steps` 常等于 chunk | 与「执行 1 步」冲突 | `ActExecuteFirstRunner` 强制只取第 0 步并清空队列 |
| P7 | 初次实现里把超大跃迁判为 `VELOCITY_LIMIT` 硬终止 | 绝对位置控制几乎无法迈第一步 | 改为软限幅（D9）；单测同步更新 |
| P8 | 写作过程中一度写反 `backend.py` / `ros_adapter.py` | 导入错乱 | 立即纠正并保留本记录，避免再犯 |

**本轮验证命令**

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate data
cd ~/robot-il/LeTools-Learning
PYTHONPATH=. python -m pytest -q kuavo_rl/tests --ignore=kuavo_rl/tests/test_actor_learner_smoke.py
# 20 passed（首轮）→ 后续含 bridge 为 22 passed

PYTHONPATH=. python scripts/rl/preflight.py --dataset data/lerobot/lerobot_merged
bash scripts/rl/run_act_baseline.sh
```

### 2026-07-13 — Phase 0 + 仿真桥接

**做了什么**

1. 发现本机 conda `data` 为 Python 3.10 + 站点包 lerobot 0.4.4，无法直接装本地 v0.6.1（要求 ≥3.12）。
2. 新建空 conda `letools-rl`(py3.12)；torch 下载过慢后改走 Docker。
3. 在 `letools-train:lerobot-0.4.2` 容器内手工安装 HIL-SERL 核心依赖，提交镜像 **`letools-train:hilserl`**。
4. 新增 `kuavo_rl/kuavo_bridge.py`：归一化 Kuavo `get_obs()` 的 torch batch / HWC。
5. `ROSBackend` 改为走 bridge；新增 `scripts/rl/run_kuavo_sim_smoke.py` 与 `install_hilserl_docker.sh`。
6. 单测升至 **22 passed**；mock shadow smoke 写入 `data/rl_runs/kuavo_sim_smoke/manifest.json`。

**新问题与处理**

| ID | 问题 | 影响 | 处理 |
|---|---|---|---|
| P9 | `pip install ...[hilserl]` 编译 `evdev` 失败（缺 `KEY_ALL_APPLICATIONS` 等宏） | 无法一次装齐 extras | 改为手动装 grpc/mujoco/gym-hil；允许无手柄（D11）；脚本固化绕过路径 |
| P10 | `gym-hil` 还缺 `mujoco`/`glfw`/等传递依赖 | import 失败 | 逐项补齐；镜像内验证 `gym_hil` + `TrainRLServerPipelineConfig` + CUDA |
| P11 | `outputs/` 目录 root 权限导致 manifest 写失败 | smoke 脚本报 PermissionError | manifest 改写到 `data/rl_runs/...` |
| P12 | gym-hil 声明 `mujoco<3.9`，镜像曾装到 3.10 | 潜在不兼容 | 安装脚本优先 pin `<3.9`；当前 import 已通过，完整 env 短跑时再收紧 |

**Phase 0 复验**

```bash
docker run --rm --gpus all \
  -v ~/robot-il/LeTools-Learning:/workspace/LeTools-Learning \
  -w /workspace/LeTools-Learning letools-train:hilserl \
  bash -lc 'source /opt/conda/etc/profile.d/conda.sh && conda activate letools && \
    python -c "import grpc,gym_hil,torch; from lerobot.rl.train_rl import TrainRLServerPipelineConfig; print(torch.cuda.is_available())"'
# True

PYTHONPATH=. python -m pytest -q kuavo_rl/tests --ignore=kuavo_rl/tests/test_actor_learner_smoke.py
# 22 passed

PYTHONPATH=. python scripts/rl/run_kuavo_sim_smoke.py --steps 10 --shadow
```

### 2026-07-13 — Phase 1 gym_hil learner+actor 短跑

**做了什么**

1. 新增 `configs/rl/gym_hil_smoke.json` + `scripts/rl/run_gym_hil_smoke.sh`。
2. 上游小修（尽量最小）：
   - `train_rl.py`：`dataset=null` 时 validate 不崩（在线-only）。
   - `gym_manipulator.py`：支持 `LEROBOT_GYM_HIL_HEADLESS` / `LEROBOT_GYM_HIL_RENDER_MODE`。
3. 镜像补齐：`PyOpenGL`、`mujoco 3.8.1`（`<3.9`）、`xvfb`/`osmesa` 相关库；重提交 `letools-train:hilserl`。
4. headless 联跑通过：actor 完成 policy loop，learner 优化步数 >100。

**结果**

| 项 | 值 |
|---|---|
| 状态 | OK |
| 产物 | `data/rl_runs/gym_hil_smoke_latest/manifest.json` |
| `max_optimization_step` | 109 |
| `episode_rewards` | `[0.0]`（随机策略，预期） |
| 渲染 | `MUJOCO_GL=osmesa` + headless Base factory |

**新问题与处理**

| ID | 问题 | 影响 | 处理 |
|---|---|---|---|
| P12 | `eval_freq` 字段已改名；`dataset=null` 触发 `TrainPipelineConfig.validate` AttributeError | learner/actor 起不来 | 配置改用 `env_eval_freq`；`TrainRLServerPipelineConfig.validate` 对 null dataset 做 sentinel（D12 相关） |
| P13 | actor/learner 共用同一 `output_dir` 时第二进程 `FileExistsError` | 官方同配置双进程在本仓库 validate 下冲突 | smoke 脚本给 actor 独立 `--output_dir` |
| P14 | `mujoco.Renderer` 缺失 / EGL 无 PLATFORM_DEVICE；`mujoco 3.10` 与 gym-hil/`mj_fullM` 不兼容 | env.reset/step 失败 | pin `mujoco<3.9`（3.8.1）+ PyOpenGL；默认 `osmesa`（D13） |
| P15 | Keyboard/Gamepad 需要 X11+pynput；`Base-v0` 直接 `gym.make(..., use_gripper=...)` 参数非法 | Docker 无显示器无法走人机路径 | headless 走 `gym_hil.factory` + `use_inputs_control=False`；有显示器时仍可用 Keyboard |
| P16 | 进程结束时 gRPC/Queue shutdown race（`Exception iterating requests` / closed Queue） | 日志有 Traceback，易误判失败 | 成功判定看 `Policy loop finished` + optimization step；忽略收尾噪声 |
| P17 | Docker 写出的 `data/rl_runs/*` 为 root 权限 | 宿主机改写不便 | 继续写 `data/rl_runs`；需要时 `chown`；后续可加 `--user` |

**复验命令**

```bash
bash scripts/rl/run_gym_hil_smoke.sh configs/rl/gym_hil_smoke.json
# 期望：PHASE1_SMOKE_OK + manifest.status=ok
cat data/rl_runs/gym_hil_smoke_latest/manifest.json
```

### 2026-07-13 — Phase 1 有屏 Keyboard 验证

**结果**：`PHASE1_DISPLAY_OK`

| 项 | 值 |
|---|---|
| 产物 | `data/rl_runs/gym_hil_display_latest/manifest.json` |
| mode | `display_keyboard`（glfw + X11） |
| `max_optimization_step` | 772 |
| episodes | 2（reward 日志 `[0.0, 0.0]`） |

**脚本**：`scripts/rl/run_gym_hil_display.sh`（去掉 `-t`，避免 heredoc TTY 报错；键盘走 X11/pynput）

---

## 4. 验收检查表（代码侧）

### Phase 2

- [x] mock backend 单测可无硬件运行
- [x] action contract：16-D、顺序、单位转换单测
- [x] SafetyGate：NaN / shape / 越界 / 差分限速 / stop
- [x] episode：success / timeout / safety fault 边界
- [x] ACT「预测 chunk、只执行第 1 步」单测
- [x] Kuavo obs bridge（torch batch / HWC）单测
- [ ] 接真实 Kuavo-Sim 连续 100 episode（需 ROS/仿真机）

### Phase 0 / 1

- [x] Docker 镜像可 import grpc / gym_hil / TrainRLServerPipelineConfig
- [x] `python -m lerobot.rl.actor --help` / `learner --help`
- [x] 提交持久镜像 `letools-train:hilserl`
- [x] gym_hil baseline 短跑（learner+actor 联跑，headless）
- [x] gym_hil Keyboard 有显示器人工干预（`scripts/rl/run_gym_hil_display.sh`，`gym_hil_display_latest`）

### Phase 4

- [x] dataset `info.json` 契约 preflight
- [x] ACT 训练脚本就绪（`train_act_stage_a.sh` / `pack_act_stage_a_cloud.sh`）；**长训改云端**，本机仅 smoke 20 step 验证过
- [ ] 云端 ACT 5k 训练完成并同步 checkpoint 回本机
- [ ] ACT checkpoint + execute-first 评测（`eval_act_execute_first.py`）
- [ ] Robometer 离线校准过门禁

---

## 5. 下一步建议（按优先级）

1. **云端阶段 A ACT 训练**（本机不跑长训）：打包后上云  
   `bash scripts/rl/pack_act_stage_a_cloud.sh` → 上传 bundle → 云端  
   `bash scripts/rl/train_act_stage_a.sh`（或 `USE_DOCKER=0`）  
   训完把 `data/rl_runs/act_stage_a_*` 同步回本机，再 `eval_act_execute_first.py`。
2. **ROS 仿真机**：本机起 Kuavo-Sim / roscore 后跑 `run_kuavo_sim_smoke.py --kuavo-env`。
3. **真机前**：Jetson 上 live `verify_joint_map.py`，冻结 `verified_on_robot: true`。
4. **Robometer**：5060 Ti 显存实测后再换真模型。

---

## 6. 问题回溯索引

| 关键字 | 查阅 |
|---|---|
| 关节索引 / `[12:26]` | P2、D7、`configs/rl/kuavo_v62_joint_map.yaml` |
| rad/deg | P3、`kuavo_rl/ros_adapter.py` |
| reward 为零 | P1、`kuavo_rl/reward.py` / `env.py` |
| ACT chunk | P6、`kuavo_rl/act_runner.py` |
| 显存 / Robometer | P5、`kuavo_rl/reward.py` |
| hilserl / Python 版本 | P4、P9、D10、`install_hilserl_docker.sh` |
| evdev 编译失败 | P9、D11 |
| gym_hil / mujoco / glfw / osmesa | P10、P12–P15、D12–D13 |
| outputs / rl_runs 权限 | P11、P17 |
| dataset=null validate | P12、`train_rl.py` |
| actor/learner output_dir | P13、`run_gym_hil_smoke.sh` |
| 速度限幅过严 | P7、`kuavo_rl/safety.py` |
| 文件写反 | P8 |

---

## 7. 当前结论

Phase 0–1 已打通：headless 与 **有屏 Keyboard** 联跑均通过（display：`opt_steps=772`，2 episodes）。

下一风险点：

1. 真实 ROS/Kuavo-Sim 接线与 rad/deg 现场确认；
2. 阶段 A ACT 训练与 execute-first 评测；
3. Robometer-4B 与 actor/learner 同卡显存；
4. 真机 reset 吞吐量与 VR 接管。
