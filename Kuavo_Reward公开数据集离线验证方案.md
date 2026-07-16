# Kuavo Reward 公开数据集离线验证方案

> 更新：2026-07-16
> 分支：`reward`（本方案及后续实现均在此分支进行，与主体 HIL-SERL / 实机闭环框架解耦）
> 关联文档：`Kuavo_HILSERL_Reward设计与实现方案.md`（主体 R0–R4 路线；本文仅覆盖「公开数据离线验证」旁路）

---

## 0. 与主体框架的关系

| 项 | 说明 |
|---|---|
| **独立性** | 本工作验证的是 LeRobot 上游 `lerobot.rewards`（Robometer / TOPReward / SARM / classifier）在公开数据上的区分力与工程可跑性，**不改动**实机控制环、SAC、VR 接管、SafetyGate。 |
| **Git 分支** | 统一在分支 **`reward`** 上开发与提交。`main` 保持主体框架稳定；验证脚本、公开数据目录、报告产物默认只进 `reward`。 |
| **合并策略** | 仅当验证结论需要回写主体文档/脚本（例如校准门禁脚本泛化、推荐默认 `image_key`）时，再开 PR 合入 `main`。验证用公开数据本身不强制入库。 |
| **不能替代什么** | 公开数据过门禁 ≠ Kuavo 可上线模型正奖励。Kuavo 真机/仿真人工标签的 R0 评测集仍是上线唯一依据（见主体方案）。 |

### 0.1 本文的评测性质

P0 只有 50 个 episode，适合验证数据读取、评分链路和发现明显失效模式，不足以给出稳健的跨数据集模型排名。所有 prompt、相机视角、抽帧数、阈值必须在运行前写入 manifest；若用 P0 调过这些超参数，P0 结果只能标为“开发集结果”，需要在 P1 或后续固定 Kuavo R0 集复验。

```text
main（主体 HIL-SERL）          reward（本方案）
─────────────────          ────────────────────────
人工 B 事件 / VR / SAC        公开 HF 数据集下载
Kuavo R0 采集与门禁            Robometer/TOPReward/SARM 离线赛马
在线 reward 启用/回退          报告与结论（不进控制环）
```

---

## 1. 最终目标

在**不采集 Kuavo 新数据**的前提下，用 Hugging Face 等公开 LeRobot 格式数据集，完成一次可复现的离线 reward 赛马，回答三个问题：

1. **API 是否可用**：`make_reward_model` / processor / `compute_reward`、以及现有 `score_rollouts_robometer.py` 扩展路径能否在公开数据上端到端跑通。
2. **信号是否有区分力**：在带成功/失败标签的公开集上，Robometer、TOPReward（及可选 SARM）能否达到约定门禁；失败轨迹 progress 是否异常爬升。
3. **工程预算是否可接受**：单 episode 延迟、峰值显存、吞吐，用于后续决定「仅离线旁路」还是「可考虑异步影子」。

**明确不做：**

- 不把公开集上的分数接进 SAC / 在线 reward。
- 不宣称跨本体（Franka / SO100 → Kuavo）可迁移为上线结论。
- 不阻塞主体分支的 R1 人工稀疏 reward 与数据采集。

---

## 2. 需要做的事（工作项）

### W1. 分支与目录约定

- [x] 从 `main` 检出分支 `reward`（本文编写时已创建）。
- [ ] 约定产物目录（均相对仓库根目录）：

```text
data/public_reward/                 # 公开数据集本地根（可 gitignore）
  ├── exylos_pick_place/            # 主评测集（成功/失败标签）
  ├── libero_10_image/              # 官方脚本示例集（可选）
  └── README.md                     # 下载来源、commit/revision、许可证
data/reward_public_eval/            # 本方案评测报告（进分支或按需提交）
  ├── datasets_manifest.json
  ├── robometer_*.json
  ├── topreward_*.json
  ├── sarm_*.json                   # 可选
  ├── race_summary.md
  └── latency_vram.json
scripts/reward_public/              # 本方案专用脚本（与 kuavo_rl 主体解耦）
  ├── download_datasets.sh
  ├── score_robometer.py            # 可基于 scripts/rl/score_rollouts_robometer.py 泛化
  ├── score_topreward.py
  ├── score_sarm.py                 # 可选
  └── make_race_report.py
```

### W2. 选定并下载公开数据集

按优先级：

| 优先级 | 数据集 | 用途 | 备注 |
|---:|---|---|---|
| P0 | [ExylosAi/pick_and_place_sample](https://huggingface.co/datasets/ExylosAi/pick_and_place_sample) | **主考卷**：success AUC、失败 progress 检查 | ~50 ep；30 成功 / 20 失败；LeRobot 兼容 |
| P1 | [HaptalAI/so100-curated](https://huggingface.co/datasets/HaptalAI/so100-curated) | 第二考卷 / 敏感性 | 候选；下载后先核对是否有可用且可信的 episode outcome 标签，再纳入。 |
| P2 | [lerobot/libero_10_image](https://huggingface.co/datasets/lerobot/libero_10_image) | TOPReward 官方 labeling 通路验证 | 官方 `compute_rabc_weights` 示例 |
| P3 | [nvidia/LIBERO_LeRobot_v3](https://huggingface.co/datasets/nvidia/LIBERO_LeRobot_v3) | 规模扩展（可选） | 体积大，按 suite 子集下载 |

下载后写入 `datasets_manifest.json`：`repo_id`、本地路径、revision、episode 数、相机 key、是否含 success/failure 字段。

### W3. 适配打分流水线

对每个 reward 模型统一输入/输出契约：

**输入（每 episode）：**

- 视频帧序列（按 `image_key` 读取；公开集多为 `observation.images.top` 等，**不是** Kuavo 的 `head_cam_h`）
- 任务文本 `task`（缺失则用数据集 card / 默认英文任务句，并记入报告）

**输出（写入 JSON）：**

- `final_progress` / `final_success`（或 TOPReward 的 log-prob / 阈值后二值）
- 可选 `progress_curve`
- `latency_s`、设备、模型路径、`image_key`、`max_frames`

模型清单：

| 模型 | 入口 | 本方案要求 |
|---|---|---|
| Robometer-4B | `lerobot/Robometer-4B` 或 `data/models/Robometer-4B` | **必做**；复用/泛化现有校准脚本 |
| TOPReward | `reward_model.type=topreward` + Qwen3-VL | 条件项；先在 5 条 episode 冒烟，显存、任务文本与视频输入均正常后才跑 P0。 |
| SARM | 上游 SARM + progress parquet | 本轮不纳入赛马。SARM 需要训练 reward model 和阶段标注，不是可直接调用的零样本模型。 |
| RewardClassifier | ResNet10 类 | **本阶段不做训练**；仅记录「公开集无 Kuavo 正负帧标签，不适合直接训 Kuavo classifier」 |

### W4. 离线赛马与门禁计算

在 **同一份固定 episode 列表**（写入 manifest，不可事后偷偷换集）上：

1. 计算 success vs failure 的 AUC（终局分数）；AUC 对单调变换不敏感，适合跨模型比较。
2. 对成功子集，在固定的等间隔 anchors 上计算 progress–时间 Spearman，并人工抽查曲线是否与任务阶段相符。
3. 对失败子集报告终局 score 分布与“误判完成率”。阈值必须预先写入 manifest，或在独立校准集确定；不同模型的 raw score 标尺不同，禁止直接用统一的 `progress >= 0.9` 比较。
4. 汇总端到端延迟、峰值显存、每 episode 输入帧数与模型版本。TOPReward 只在稀疏 anchors（例如 15）运行，不做全帧密集 VLM 推理。

### W5. 产出结论文档

生成 `data/reward_public_eval/race_summary.md`，必须包含：

- 各模型是否通过本方案门禁；
- 推荐后续用途（仅旁路排序 / 不可用 / 需换数据或 prompt）；
- 与主体方案的衔接建议（例如：公开集验证通过后，仍须在 Kuavo R0 上重跑同样门禁）。

---

## 3. 怎么做（执行步骤）

### 3.1 环境与算力分流

```bash
git checkout reward

# 依赖：LeRobot + robometer/topreward extras；GPU 推荐
# 模型：已有 data/models/Robometer-4B 可复用；TOPReward 默认下载 Qwen3-VL-8B-Instruct。
```

| 工作 | 本地建议 | 云端建议 | 原因 |
|---|---|---|---|
| 数据结构检查、manifest、5 条 episode 冒烟 | 可在本地完成 | 不需要 | 主要是 I/O 与格式调试。 |
| Robometer-4B 全 P0 打分 | 有空闲 16 GB GPU 可尝试 | 推荐 16–24 GB GPU | 4B 权重约 8 GB（bf16）外加视觉/运行时开销；更适合离线批处理。 |
| TOPReward（Qwen3-VL-8B） | 不建议占用控制机 | 推荐 ≥24 GB GPU，优先 40 GB | 权重约 16 GB，视频 token 与运行时峰值会继续增加；16 GB 显存存在 OOM 风险。 |
| SARM 训练/消融 | 不做 | 若另开任务，使用 ≥24 GB GPU | 这是训练任务，不是本轮零样本评测。 |

云端只处理公开数据、模型缓存和报告产物；不要上传 Kuavo 真机数据、ROS 环境密钥或控制相关配置。固定容器镜像/`pip freeze`、CUDA、GPU 型号、模型 revision 和数据 revision，写入 `datasets_manifest.json`。

### 3.2 下载 P0 数据

```bash
mkdir -p data/public_reward
huggingface-cli download ExylosAi/pick_and_place_sample \
  --repo-type dataset \
  --local-dir data/public_reward/exylos_pick_place
```

检查 `info.json`（LeRobot v2）或 `meta/info.json`（LeRobot v3）、`episodes.jsonl` 与 `annotations.json`：fps、features、相机 key、episode 数及 success/failure 字段。P0 当前 card 声明 5 路相机、30 success / 20 failure；实际运行以下载 revision 的文件为准。把 Hub commit/revision 固定写入 manifest。

### 3.3 Robometer 打分

1. 扩展或新建 `scripts/reward_public/score_robometer.py`：支持任意 `dataset` 根目录、可配置 `image_key`、按标签文件划分 success/failure（禁止仅用「时间反序」当唯一 hard-negative）。
2. 对 Exylos 的固定 episode 列表打分。P0 总共仅 30 success / 20 failure；若不调参可全量评测并标注为探索性结果。若调 prompt/image_key/max_frames，则这些选择只能用开发列表，最终门禁须在独立 P1 或 Kuavo R0 重跑。
3. 输出 `data/reward_public_eval/robometer_exylos.json`。

参考命令形态：

```bash
python scripts/reward_public/score_robometer.py \
  --dataset data/public_reward/exylos_pick_place \
  --image-key observation.images.<从info.json确认> \
  --model-id data/models/Robometer-4B \
  --out data/reward_public_eval/robometer_exylos.json
```

### 3.4 TOPReward 打分

1. 先用官方脚本验证通路：

```bash
# 先运行 5 条公开 episode；通过后才扩大。使用已安装的 LeRobot module，
# 不依赖仓库内源码的相对路径。
python -m lerobot.rewards.topreward.compute_rabc_weights \
  --dataset-repo-id data/public_reward/exylos_pick_place \
  --episodes 0 1 2 3 4 \
  --num-samples 15 \
  --device cuda
```

2. 再对 Exylos 本地路径封装 `score_topreward.py`，输出与 Robometer 可比的终局分数与延迟。

### 3.5 SARM（另开训练实验，不计入本轮 DoD）

SARM 需要先定义阶段标注策略，再训练其 reward model，之后才能生成 `sarm_progress.parquet`。P0 虽有 phase annotations，适合作为未来可行性输入，但不能把它与 Robometer/TOPReward 的零样本打分放在同一“直接运行”工作量里。本轮只保留接口调研，不训练、不计入赛马赢家。

### 3.6 汇总报告

```bash
# TOPReward 冒烟未通过时，只传入 Robometer 报告，并在摘要中写明原因。
python scripts/reward_public/make_race_report.py \
  --inputs data/reward_public_eval/robometer_exylos.json \
  --out data/reward_public_eval/race_summary.md
```

### 3.7 Git 工作流

```bash
# 始终在 reward 分支
git checkout reward

# 提交：方案文档、脚本、manifest、摘要报告
# 不强制提交：大体积公开数据集、模型权重
# .gitignore 建议忽略 data/public_reward/** 的视频与权重，保留 README/manifest
```

合入 `main` 的条件（可选）：脚本已稳定、主体校准脚本可安全复用同一门禁函数、且有明确评审。

---

## 4. 验收标准

### 4.1 工程验收（必须全部满足）

| ID | 标准 | 证据 |
|---|---|---|
| E1 | 工作在 `reward` 分支完成，未污染 `main` 上未评审的控制环/SAC 代码 | `git branch`、PR 说明 |
| E2 | P0 数据集可本地加载，相机 key 与标签字段已写入 `datasets_manifest.json` | manifest 文件 |
| E3 | Robometer 必须产出可解析 JSON；TOPReward 若资源冒烟通过则同样产出，否则记录失败原因与资源证据 | `robometer_*.json`、可选 `topreward_*.json`、日志 |
| E4 | 评测集 episode 列表固定（hash 或显式 id 列表），报告可复现 | manifest + 报告中的 episode id |
| E5 | `race_summary.md` 给出明确结论与「是否可用于 Kuavo 上线」的否定/限定表述 | 摘要文档 |

### 4.2 指标门禁（公开集赛马；与主体手册对齐但**仅作旁路结论**）

在 **Exylos（或同等带标签集）** 的预注册固定 episode 列表上：

| 指标 | 通过线 | 说明 |
|---|---:|---|
| Success vs Failure AUC | ≥ 0.85 | 终局 success 概率或可比分数 |
| 成功轨迹 progress–时间 Spearman | ≥ 0.70 | 过低则 progress 不可信 |
| 失败轨迹「误判完成」率 | 报告并讨论 | 用运行前固定、或由独立校准集确定的模型内阈值统计；人工抽查至少 10 条失败。禁止跨模型共用 raw `0.9` 阈值。 |
| 单 episode 推理延迟 | 如实记录 p50/p95；**不设上线阈值** | 公开验证阶段只要求可离线跑完 |

**通过定义：**

- **框架验证通过**：E1–E5 满足；其中 E3 的最低要求是 Robometer 成功产生可解析报告与指标，TOPReward 是否运行不阻塞框架验证。
- **探索性赛马赢家**：在预注册配置下 AUC 更高、且模型内阈值下失败误判更少者记为旁路优选；写入 summary。若配置在 P0 上调过，必须标注为开发结论。
- **全部未过线**：仍算完成「验证实验」，结论为「公开集上不可用 / 需调 prompt、image_key、max_frames 或换数据」；**禁止**据此开启在线模型正奖励。

### 4.3 明确非验收项

- Kuavo 真机/仿真 R0 人工集门禁（属主体方案）。
- HIL replay importer、SAC 训练回报提升。
- RewardClassifier 训练精度。
- 将公开数据合并进 `data/lerobot/lerobot_merged` 用于策略训练（除非另开任务）。

---

## 5. 风险与注意

1. **域差**：桌面单臂/Franka ≠ Kuavo 双臂胸前搬运；公开集结论只证明「模型+流水线」，不证明「Kuavo 任务」。
2. **标签噪声**：社区集 failure 定义可能是速度尖峰/动作失配，未必是任务失败；Exylos 优先。
3. **合成负样本陷阱**：主体已有校准中 `synthetic_hard_negative` 导致门禁失真；本方案 **禁止** 把时间反序当作唯一失败集。
4. **资源**：Exylos 本体约 3.5 GB；Robometer-4B 需约 8 GB 权重加运行时开销；TOPReward 默认 Qwen3-VL-8B 首次权重下载约 16 GB，推荐云端 ≥24 GB GPU。均先跑 5 条 episode 冒烟，再扩大。
5. **阈值泄漏**：不能在同一 50 条 P0 上反复选择 prompt、image_key、阈值后还把 AUC/FPR 称作独立门禁；需预注册，或改在 P1/Kuavo R0 复验。
6. **许可证**：下载前确认各数据集 License，仅内部验证亦应在 README 注明来源。

---

## 6. 时间盒建议

| 阶段 | 内容 | 建议产出 |
|---|---|---|
| D1 | 分支目录、下载 Exylos、manifest | 可加载数据 |
| D2 | Robometer 全流程 + 门禁数字 | `robometer_exylos.json` |
| D3 | TOPReward 5 条云端冒烟；仅资源与输入通过后再跑 Exylos | 日志；可选 `topreward_exylos.json` |
| D4 | 赛马报告、是否扩展 Haptal/LIBERO | `race_summary.md` |

---

## 7. 完成定义（DoD）

同时满足：

1. `reward` 分支上存在本方案文档与可运行打分/汇总脚本；
2. P0 的 Robometer 离线报告已生成；TOPReward 若未通过资源冒烟，报告须明确记录原因而非静默跳过；
3. `race_summary.md` 写清：谁过线、谁不过、对主体 R3/R4 的建议（通常为：继续禁用在线模型 reward，仅保留离线旁路能力）；
4. 未将公开验证结论误写为「可上线 Kuavo 正奖励」。

---

## 8. 下一步（本分支内）

1. 落地 `scripts/reward_public/` 与 `.gitignore` 规则。
2. 下载 Exylos，写 `datasets_manifest.json`。
3. 泛化 Robometer 打分脚本并跑通门禁。
4. 在云端对 TOPReward 跑 5 条 episode 冒烟；通过后才接入完整 P0。
5. 生成 `race_summary.md`，决定是否扩展 Haptal/LIBERO。
6. 若需合入主体：单独 PR，只带脚本/文档，不带大文件。
