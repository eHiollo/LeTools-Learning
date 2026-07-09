# 阶段 3（云端）执行状态

执行时间：2026-07-09

## 新增文件

| 路径 | 说明 |
|------|------|
| `deploy.env.example` | 环境变量模板（复制为 `deploy.env`） |
| `configs/train/cloud/` | 云端完整训练配置（ACT / Diffusion） |
| `scripts/cloud_bootstrap.sh` | Docker 基建 + smoke 一键脚本 |
| `scripts/cloud_train.sh` | 完整训练（Docker 优先，无权限时回退 conda） |
| `scripts/native_train.sh` | 本机 conda 训练（无 Docker 时使用） |
| `scripts/setup_native_env.sh` | 安装本机 conda 训练环境 |

## 快速开始

```bash
cd ~/robot-il/LeTools-Learning
cp deploy.env.example deploy.env   # 按需改 TASK_NAME
source deploy.env

# 有 Docker
bash scripts/cloud_bootstrap.sh

# 无 Docker（本机 conda）
bash scripts/setup_native_env.sh
bash scripts/native_train.sh --policy act --mode simple --config-root configs/train/smoke
bash scripts/cloud_train.sh diffusion
```

## 配置说明

- YAML 中数据路径使用容器路径 `/workspace/LeTools-Learning/...`
- `native_train.sh` 会自动替换为 `deploy.env` 中的 `REPO_ROOT` / `LEROBOT_DATA`
- 更换数据集：修改 `deploy.env` 的 `TASK_NAME`，并同步改 `configs/train/cloud/total/*_total.yaml` 的 `repo_id` 与路径后缀

## 云端实测（2026-07-09）

| 项 | 结果 |
|----|------|
| GPU | RTX 5880 Ada 48GB，`torch 2.9.1+cu128` |
| 数据集 | `lerobot_pada_v1_3_sample16_a10`（1 ep，91 帧） |
| ACT smoke | loss 101→33 ✅ |
| Diffusion 训练 | 已用 `native_train.sh` 后台启动 |
