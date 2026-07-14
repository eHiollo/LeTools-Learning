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
| Phase 3 Robometer / VR 接管 / replay | 🟡 接管与存储已通 | MuJoCo VR 分臂接管、人工 reward、HIL replay 已落地；Robometer 门禁未过 |
| Phase 4 数据审计 / ACT 基线脚本 | 🟢 脚本已落地 | preflight / ACT execute-first / sim smoke 通过 |
| Phase 5 仿真闭环 | 🟡 对照 harness 通 | ACT/SAC Sim smoke + Phase5 四臂对照 harness；SAC 尚未证明优于 ACT |
| Phase 6+ 影子/真机 | ⬜ 未开始 | 须过 Phase 5 验收 |

图例：🟢 已完成代码 / 🟡 部分 / ⬜ 未开始 / 🔴 阻断

---

## 1. 实现决策（相对手册的落地选择）

| # | 决策 | 原因 |
|---|---|---|
| D1 | `kuavo_rl` 放主仓库，不改 `third_party/lerobot` | 手册硬约束；便于升级上游 |
| D14 | Phase1 所需 upstream 行为用 **运行时 monkeypatch**（`kuavo_rl/lerobot_patches.py` + `hilserl_cli`） | 曾误改 submodule；已回退；入口改为 `python -m kuavo_rl.hilserl_cli {learner\|actor}` |
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
| `robometer_scorer.py` | Robometer-4B 懒加载打分（离线/worker 共用） |
| `calibration_metrics.py` | 3.4 门禁：Spearman / AUC |
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
- `score_rollouts_robometer.py` / `probe_robometer_vram.py` / `run_robometer_calibration.sh`
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
| P12 | `eval_freq` 字段已改名；`dataset=null` 触发 `TrainPipelineConfig.validate` AttributeError | learner/actor 起不来 | 配置改用 `env_eval_freq`；**运行时 patch** `TrainRLServerPipelineConfig.validate`（不改 submodule） |
| P18 | 曾直接改 `third_party/lerobot`（train_rl / gym_manipulator） | 违反 D1 | 已 `git checkout` 回退；逻辑迁至 `kuavo_rl/lerobot_patches.py` |
| P19 | Kuavo-Sim 工作空间 devel 指向 `/root/kuavo_ws`；宿主机直接 launch 缺 OpenVINO/LCM；默认 mujoco 无相机 publisher | 阻塞真实 sim smoke | 已解决：软链 `/root/kuavo_ws`→本机 workspace；**在 `./docker/run.sh` 容器内用 `setup.zsh` 启动**；smoke 用 joint-only 配置（见下） |
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

### 2026-07-13 — Kuavo-Sim 真实闭环 smoke（关节，无相机）

**前置**

1. `sudo ln -s /home/fulin/VSCode/kuavo-ros-control /root/kuavo_ws`（devel 仍指向该路径）
2. 容器：`cd /home/fulin/VSCode/kuavo-ros-control && ./docker/run.sh`
3. 容器内（zsh）：`source /opt/ros/noetic/setup.zsh && source /root/kuavo_ws/devel/setup.zsh && roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch`
4. 宿主机 `data` 环境：`numpy==1.26.4`（兼容 ROS `cv_bridge`）、`PYTHONPATH` 含 `kuavo_humanoid_sdk` + apriltag python 路径

**做了什么**

1. 修复 `scripts/rl/run_kuavo_sim_smoke.py`：正确 `load_kuavo_config` + `gym.make(..., config=)`
2. 新增 `configs/deploy/total/deploy_sim_smoke_total.yaml`（仅 `joint_q`/`leju_claw`；默认 mujoco 无 `/cam_*` publisher）
3. 新增 `scripts/rl/run_kuavo_sim_smoke.sh`
4. 修复 `kuavo_rl/kuavo_bridge.py`：unwrap `gym.make` 包装层以调用 `get_obs`/`exec_action`

**结果**

| 项 | 值 |
|---|---|
| 状态 | OK |
| `mode` | `kuavo_sim`（非 mock_fallback） |
| steps | 5 |
| `last_fault` | `NONE` |
| `discarded_per_step` | 9（ACT execute-first） |
| 产物 | `data/rl_runs/kuavo_sim_smoke/manifest.json` |

**复验**

```bash
# 容器内仿真已启动后，宿主机：
bash scripts/rl/run_kuavo_sim_smoke.sh 10
cat data/rl_runs/kuavo_sim_smoke/manifest.json  # mode=kuavo_sim
```

**未完成 / 下一步**

- 默认 mujoco launch **无原生 RGB**；已加 s62 模型相机 + `scripts/rl/run_sim_rgb_cameras.sh` 影子渲染发布 `/cam_*`
- 仿真必须用 **`robot_version:=62`**（当前若仍是 42 需重启）
- 下一步：接 ACT `005000` 做带相机的 sim execute-first 评测

### 2026-07-13 — s62 + RGB 相机接入

**事实**

- 官方 `run_mujoco_camera:=true` 只开腰部 **depth**（RayCaster），不出 `/cam_h|/cam_l|/cam_r` RGB。
- 已在 `biped_s62.xml` 增加 `cam_h` / `cam_l` / `cam_r` / `waist_camera`。
- 新增宿主机 RGB 发布：`bash scripts/rl/run_sim_rgb_cameras.sh`（跟 `/sensors_data_raw` 同步渲染）。

**你需要做的（重启仿真）**

容器内 Ctrl+C 停掉当前 launch，然后：

```zsh
export ROBOT_VERSION=62
source /opt/ros/noetic/setup.zsh
source /root/kuavo_ws/devel/setup.zsh
# 或: bash /root/kuavo_ws/start_kuavo_sim_v62_with_cameras.sh
roslaunch humanoid_controllers load_kuavo_mujoco_sim.launch \
  robot_version:=62 run_mujoco_camera:=true joystick_type:=sim
```

宿主机另开终端：

```bash
bash scripts/rl/run_sim_rgb_cameras.sh
# 验收：rostopic hz /cam_h/color/image_raw/compressed
bash scripts/rl/run_kuavo_sim_smoke.sh 5
```

### 2026-07-13 — ACT `005000` Kuavo-Sim execute-first 闭环

**架构**（宿主机 Py3.10 无法 import 仓库 lerobot 0.6.1）：

| 端 | 角色 |
|---|---|
| Docker `letools-train:hilserl` | ACT 推理 TCP 服务（pre/post + `predict_action_chunk`） |
| 宿主机 `data` + ROS | `Kuavo-Sim` + `ActExecuteFirstRunner` 远程策略 |

**命令**

```bash
# 已起 v62 轮臂仿真 + RGB 后：
bash scripts/rl/run_act_infer_server.sh          # Docker :8765
bash scripts/rl/run_act_kuavo_sim_eval.sh 10     # 主机闭环
cat data/rl_runs/act_kuavo_sim_eval/manifest.json
```

**结果**

| 项 | 值 |
|---|---|
| 状态 | OK |
| `mode` | `kuavo_sim` |
| `policy` | `remote` |
| chunk | `(10, 16)`，discard=9 |
| steps | 10 |
| `last_fault` | `NONE` |
| 产物 | `data/rl_runs/act_kuavo_sim_eval/manifest.json` |

**新增/改动**

- `kuavo_rl/act_policy.py`：本地/远程 ACT 适配；图像 float[0,1]；跨 NumPy 版本字节打包
- `scripts/rl/act_infer_server.py` + `run_act_infer_server.sh`
- `scripts/rl/run_act_kuavo_sim_eval.sh`；`eval_act_execute_first.py` 支持 `--policy remote` + deploy config
- bridge 将相机 resize 到 `848×480`；RGB 发布默认改为 848×480
- `configs/rl/kuavo_hilserl_sim_act.yaml`（放宽 consecutive clips）

**下一步**

- 更长 episode / 影子模式；关节 map live 校验；Stage B SAC

### 2026-07-13 — 长 episode + 影子模式 + joint-map live

**结果**

| 项 | 值 |
|---|---|
| 长闭环 | 50 步，`last_fault=NONE`，`data/rl_runs/act_kuavo_sim_eval_long/manifest.json` |
| 影子模式 | `SHADOW=1` 20 步，`shadow_mode=true`，不下发动作，`act_kuavo_sim_shadow/manifest.json` |
| joint-map live | `ok=true`，当前仿真 `raw_joint_dim=20` → 切片 `[4:18]`；夹爪可读；`joint_map_live/manifest.json` |

**注意**：轮臂仿真传感器为 **20-D**（非 biped 28-D/`[12:26]`）。deploy/obs 路径已按表切片；真机仍须再跑 `--live` 确认维度。

**用法**

```bash
bash scripts/rl/run_act_infer_server.sh
MANIFEST=data/rl_runs/act_kuavo_sim_eval_long/manifest.json bash scripts/rl/run_act_kuavo_sim_eval.sh 50
SHADOW=1 bash scripts/rl/run_act_kuavo_sim_eval.sh 20
# ROS 已 source：
PYTHONPATH=... python scripts/rl/verify_joint_map.py --live --out data/rl_runs/joint_map_live/manifest.json
```

**下一步**：Stage B SAC / 真机前再验 28-D joint-map

### 2026-07-13 — Stage B SAC mock smoke（gaussian_actor）

**结果**

| 项 | 值 |
|---|---|
| 状态 | OK |
| 环境 | `KuavoHILSerlEnv` + `MockBackend`（`name=kuavo_hilserl`） |
| 策略/算法 | `gaussian_actor` + `sac`（单步 16-D，无 ACT chunk） |
| 启动 | `bash scripts/rl/run_kuavo_sac_smoke.sh` |
| 产物 | `data/rl_runs/kuavo_sac_smoke_latest/manifest.json` |

**新增**

- `kuavo_rl/hilserl_processors.py` + `lerobot_patches` 路由 `kuavo_hilserl`
- `configs/rl/kuavo_sac_smoke.json`（128×128 三相机 smoke）
- Mock 支持可配置 `image_shape_chw`

**下一步**：Kuavo-Sim 上接 ROS backend 的 Stage B；对照 ACT 基线

### 2026-07-13 — Robometer 离线打分冒烟 + 显存探针

**结果（冒烟通，手册 3.4 门禁未过）**

| 项 | 值 |
|---|---|
| 权重 | ModelScope 本地：`data/models/Robometer-4B` + `Qwen3-VL-4B-Instruct` |
| 视频解码 | Docker 内 PyAV（AV1）；OpenCV/torchcodec 不可用 |
| 离线报告 | `data/reward_calibration/offline_scores.json`（`status=SCORED`） |
| 延迟 | ~35 s/条（4 帧）/ ~74 s/条（8 帧），5060 Ti |
| 门禁 | Spearman/AUC 未同时过线；失败集为合成 hard-neg，非人工标注 |
| 显存 | `vram_budget.json` peak ≈ **8.7 GB**，`co_resident_possible_if_actor_learner_fit` |
| 在线 | **保持** `robometer_mode=disabled` / 确定性 reward |

**用法**

```bash
# Docker hilserl + 本地权重
python -u scripts/rl/score_rollouts_robometer.py --max-success-eps 2 --max-frames 4 \
  --model-id data/models/Robometer-4B --out data/reward_calibration/offline_scores.json
python -u scripts/rl/probe_robometer_vram.py --model-id data/models/Robometer-4B \
  --out data/reward_calibration/vram_budget.json
```

**下一步**：人工标注成功/失败集后再冲 3.4；当前可进 Phase 5 ACT vs SAC 对照（确定性 reward）

### 2026-07-13 — Phase 5 对照 harness（确定性 reward）

**结果**

| 项 | 值 |
|---|---|
| 入口 | `bash scripts/rl/run_phase5_contrast.sh 20` |
| 模式 | MockBackend；`use_robometer=false` |
| 四臂 | zero / random / ACT execute-first harness / SAC explore proxy |
| `harness_ok` | **true**（action_dim=16；ACT discard=9） |
| `enter_real_stage_b` | **false**（无训练好的 SAC 成功率，不可解释改善） |
| 产物 | `data/rl_runs/phase5_contrast_latest/{manifest.json,summary.md}` |
| 附带 Sim 参考 | ACT `act_kuavo_sim_eval`（fault=NONE）；SAC `kuavo_sac_sim_latest`（status=ok，reward 全 0） |

**说明**：对照 harness 验证契约与入口；**不**等于手册“SAC 相对 ACT 有可解释改善”。Robometer 门禁仍未过。

**下一步**：Kuavo-Sim 上更长 ACT 评测；SAC 需有效 reward/训练后再做真对照。

---

## 4. 验收检查表（代码侧）

### Phase 2

- [x] mock backend 单测可无硬件运行
- [x] action contract：16-D、顺序、单位转换单测
- [x] SafetyGate：NaN / shape / 越界 / 差分限速 / stop
- [x] episode：success / timeout / safety fault 边界
- [x] ACT「预测 chunk、只执行第 1 步」单测
- [x] Kuavo obs bridge（torch batch / HWC）单测
- [x] 接真实 Kuavo-Sim smoke（关节闭环，`mode=kuavo_sim`）；带相机 / 100 episode / v62 重启后复验仍待做

### Phase 0 / 1

- [x] Docker 镜像可 import grpc / gym_hil / TrainRLServerPipelineConfig
- [x] `python -m lerobot.rl.actor --help` / `learner --help`
- [x] 提交持久镜像 `letools-train:hilserl`
- [x] gym_hil baseline 短跑（learner+actor 联跑，headless）
- [x] gym_hil Keyboard 有显示器人工干预（`scripts/rl/run_gym_hil_display.sh`，`gym_hil_display_latest`）

### Phase 4

- [x] dataset `info.json` 契约 preflight
- [x] ACT 训练脚本就绪（`train_act_stage_a.sh` / `pack_act_stage_a_cloud.sh`）；**长训改云端**，本机仅 smoke 20 step 验证过
- [x] 云端 ACT 5k 训练完成并同步 checkpoint 回本机（`data/rl_runs/act_stage_a_latest` → `checkpoints/005000`）
- [x] ACT checkpoint + execute-first 离线评测通过（chunk=(10,16)、discard=9、`execute_first_ok`）
- [x] Robometer 离线真打分冒烟（`SCORED`，合成负样本；**手册门禁未过**）
- [ ] Robometer 离线校准过门禁（需人工标注失败集）

### Phase 5（仿真闭环）

- [x] Kuavo-Sim `--kuavo-env` smoke（`mode=kuavo_sim`）
- [x] ACT runner 接真 checkpoint `005000` 跑仿真 episode（Docker 远程推理 + 主机 ROS）
- [x] 阶段 B gaussian_actor+SAC **mock** smoke（learner+actor，单步 16-D）
- [x] 阶段 B 接 Kuavo-Sim ROS（`KUAVO_HILSERL_BACKEND=proxy`，`STAGEB_SIM_OK` / `kuavo_sac_sim_latest`）
- [x] Robometer 显存预算记录（`vram_budget.json` ≈ 8.7 GB）
- [ ] Robometer 异步打分过 3.4 门禁（在线 `episode_end` 仍关）
- [x] Phase5 对照 harness（zero/random/ACT/SAC proxy，确定性 reward；`phase5_contrast_latest`）
- [ ] ACT vs SAC 可解释改善对照（需训练 SAC + 有效成功指标）

---

## 5. 下一步建议（按优先级）

1. **Kuavo-Sim 更长 ACT 评测**（成功率基线），再谈 SAC 训练目标。
2. **SAC 有效 reward**：确定性事件 / 人工按键；Robometer 待人工失败集后再开。
3. **真机前**：影子模式 + `verify_joint_map.py --live`。

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
| dataset=null validate | P12、P18、`kuavo_rl/lerobot_patches.py` |
| actor/learner output_dir | P13、`run_gym_hil_smoke.sh` |
| 速度限幅过严 | P7、`kuavo_rl/safety.py` |
| 文件写反 | P8 |

---

## 7. 当前结论

阶段 A（ACT Sim 闭环）与阶段 B（gaussian_actor+SAC）均已打通：mock smoke 与 **Kuavo-Sim ROS proxy**（`STAGEB_SIM_OK`，`kuavo_sac_sim_latest`）均已跑通；actor 单步 16-D，未混入 ACT chunk。

Robometer 冒烟已通：本地权重加载、PyAV 读帧、真推理打分、显存峰值约 8.7 GB 均已验证；单条延迟约 35–75 s（不可进 10 Hz 同步路径）。3.4 门禁未过（合成负样本 + 指标未齐），在线 reward 保持确定性旁路。

Phase5 对照 harness 已通（`phase5_contrast_latest`，`harness_ok=true`）；**尚未**证明 SAC>ACT，禁止据此进真机阶段 B。

下一风险点：

1. 人工标注失败集后再过 Robometer 3.4；否则不得开 `episode_end`；
2. Sim 上更长 ACT 基线 + 可训练的 SAC reward；
3. 真机 joint-map / 影子模式放行。

### 2026-07-13 — Quest3 VR 人工接管接入（真机前准备）

**目标**：复用 Kuavo 5W v62 现有 Quest3 → IK → `/kuavo_arm_traj` 遥操作链，不在 LeTools 中重复实现 UDP、骨骼解析或 IK；LeTools 只负责接收人工动作、控制接管时的发布仲裁、reward 事件和数据审计。

**已完成**

1. 新增 `kuavo_rl/ros_teleop.py`：
   - 订阅 `/quest_joystick_data` 和 `/kuavo_arm_traj`；
   - 对齐原 Quest3 遥操作：使用左右 grip 分别激活对应手臂，阈值默认为 `0.8`；
   - 将 Kuavo `/kuavo_arm_traj` 的 14-D 角度值（degree）转换为 canonical 16-D action（radian）；
   - action 顺序保持 `L7,left_claw,R7,right_claw`；
   - 夹爪暂时保持 reference action，避免在未确认 qiangnao/夹爪消息映射前误发手指命令；
   - `/kuavo_arm_traj` 超过 `0.20s` 未更新时自动取消人工接管；
   - 沿用 Kuavo Quest3 FSM 的“双左键”急停手势；
   - success/failure/abort 按键映射改为显式配置，默认不猜测。
2. `kuavo_rl/adapter.py` 支持注入自定义 `TeleopAdapter`。
3. `kuavo_rl/env.py` 在人工接管时不再重复发布 LeTools 策略动作，避免与现有 VR IK 节点同时写 `/kuavo_arm_traj`；同时把 `teleop_source`、`teleop_age_s` 写入每步 `info`。
4. `scripts/rl/eval_act_execute_first.py` 增加 `--ros-teleop`，并从配置的 `teleop:` 段读取 topic/deadman/按键设置。
5. `configs/rl/kuavo_hilserl_real_mvp.yaml` 默认关闭在线 Robometer，与当前 3.4 门禁未通过的结论一致；增加 Quest3 遥操作配置模板。
6. 新增无 ROS 单测：
   - `kuavo_rl/tests/test_ros_teleop.py`
   - `kuavo_rl/tests/test_teleop_override.py`

**验证结果**

```text
核心 HIL-SERL + 新增遥操作测试：32 passed
```

当前 `data` 环境完整收集还剩 3 个既有 `test_lerobot_patches.py` 失败，原因是该环境没有 `lerobot.rl.train_rl` 模块；不属于本次遥操作改动。

**当前限制 / 真机前必须确认**

- 尚未在真实机器人上运行 ROS 遥操作适配器；目前只完成无 ROS 单测。
- success/failure/abort 的 Quest3 按键不能直接采用配置中的注释示例，必须现场确认 `JoySticks.msg` 语义后再启用。
- 14-D 手臂与 2-D 夹爪的真实映射仍需结合 `verify_joint_map.py --live` 和实际 end-effector 类型确认。
- 当前 LeTools 在 grip 接管时跳过发布，但尚未引入独立 ROS topic mux；真机联调时必须确认 VR IK 与策略不会同时产生有效控制命令。
- Robometer 仍保持离线/关闭，不能在 3.4 门禁通过前打开在线 `episode_end`。

**下一步**

1. 在不上策略动作的前提下启动 v62 Quest3 遥操作，确认 `/quest_joystick_data`、`/kuavo_arm_traj` 频率和单位。
2. 运行 `verify_joint_map.py --live`，确认 14-D 手臂顺序、20/28-D raw state 和夹爪映射。
3. 确认 success/failure/abort 按键并写入 `configs/rl/kuavo_hilserl_real_mvp.yaml`。
4. 真机先做 shadow，再做低速、短时 ACT 接管测试；确认控制权切换和急停后再进入 SAC。

### 2026-07-14 — VR 设备联调暂缓

已将 VR 人工接管恢复后的检查、启动命令、deadman/reward 配置和真机前验收整理到：

- `Kuavo_HILSERL_VR人工接管联调方案.md`

当前 Quest3/VR 设备存在问题，暂不进行 ROS VR 联调和策略接管测试。设备恢复后必须从话题频率、消息字段、degree/radian、joint-map、deadman 和急停检查重新开始，不跳过 shadow 阶段。

VR 暂缓期间继续推进：数据与 joint-map 静态审计、人工 reward/replay 离线测试、ACT checkpoint 评测、SAC reward 设计和 Robometer 离线标注校准。

### 2026-07-14 — Reward 设计方案整理

已新增 `Kuavo_HILSERL_Reward设计与实现方案.md`，记录：

- LeRobot 基础 reward 默认返回 0 的原因；
- RewardClassifier 需要 Kuavo 任务专用标注和权重，不能直接套用；
- 当前 Kuavo 确定性人工 reward 语义；
- 人工标签、Kuavo 专用 classifier、progress shaping、Robometer 的分层使用方式；
- SAC 训练前的 reward 数据和校准门禁。

当前原则：真机 MVP 使用人工 success/failure/abort + SafetyGate reward；Robometer 保持离线/关闭，直到 3.4 门禁通过。


### 2026-07-14 — Quest3 VR 接管、人工 reward 与 HIL replay 联调完成

**完成项**

1. MuJoCo + Quest3/IK + ACT 联调通过：
   - 左/右 grip 分别接管对应手臂；松开后 ACT 恢复；
   - 左手两个按键同时按下触发急停；
   - 接管期间 LeTools 停止发布策略动作，不与既有 VR IK 链路抢控制权。
2. 人工 reward 固化为不占用遥操作按键的 B 手势：
   - `right_second_button_pressed`（B）单击 success；
   - 双击（≤0.35s）failure；
   - 长按（≥1.2s）abort；
   - B 字段已现场确认。
3. 新增 HIL replay 落盘：
   - 轻量审计：`data/rl_runs/hilserl_episodes/hilserl_vr/transitions.jsonl`；
   - 可训练 replay：`.../replay/episodes/<episode_id>/transitions.jsonl`、state `.npy`、三路相机帧 `.jpg/.npy`；
   - 每条 transition 含 `obs, action, reward, next_obs, terminated, truncated, is_intervention`。
4. 修正人工 action 的训练语义：
   - `extras.teleop_raw_action` 仅保留原始 VR IK target，禁止直接作为训练标签；
   - 主 `action` 以实测 state 为锚，仅覆盖 grip 接管侧并经 SafetyGate 限幅；
   - 保存 `intervention_mask` 与 `intervention_segment_id/step`，供后续导入 replay 时屏蔽接管拼接段；
   - 未接管侧保持实测状态，不再复用陈旧 VR action。

**验证**

```text
VR / replay / ACT 相关回归：16 passed
```

完整测试在两个本地环境仍有既有依赖差异：`letools-rl` 缺 OpenCV（既有图像 resize 测试失败）；`data` 环境缺 `lerobot.rl.train_rl`（3 个 `lerobot_patches` 测试失败）。两者均非本轮 VR/replay 改动引入。

**下一步**

1. 用这套格式采集带 B reward 的仿真/真机 HIL episode；
2. 编写 replay importer，将正式数据接入 HIL-SERL online buffer，并按 `intervention_mask`、segment warmup 过滤；
3. 先 shadow、再低速真机验证，再启动 SAC 微调与 ACT 对照。
