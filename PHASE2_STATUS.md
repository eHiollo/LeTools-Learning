# 阶段 2 执行状态

执行时间：2026-07-09

## 验收标准

| 项 | 状态 | 说明 |
|----|------|------|
| dry-run | ✅ | 训练命令解析正确 |
| 10 步训练 | ✅ | loss 90 → 40 下降 |
| checkpoint | ✅ | `outputs/train/act_smoke_20260709_060544/checkpoints/000010/` |
| `model.safetensors` | ✅ | 已生成 |

## 训练输出目录

`outputs/train/act_smoke_20260709_060544/`

## 修复项

- smoke 配置补充 `policy.push_to_hub: false`（否则 lerobot 校验失败）

## 本机运行方式（无 docker compose）

```bash
cd ~/robot-il/LeTools-Learning
source deploy.env
chmod +x docker/run_train.sh

# smoke
./docker/run_train.sh python kuavo_model/train.py --policy act --mode simple --config-root configs/train/smoke
```

## 当前 Git 状态（上云前需 commit + push）

```
commit（克隆点）: 3bde39d9f2ab34deb15d4bd4eb55c5cf7502ca97
未提交: docker/, configs/train/smoke/, .dockerignore, .gitignore 等
```

## 上云前待办（本地用户）

1. `git add` + `git commit` + `git push`（含 docker 与 smoke 配置）
2. 填写 `deploy.env` 中 `ACR_NS`、`OSS_BUCKET`
3. `docker push` 镜像到 ACR，或让云端重新 build
4. `ossutil` 上传 `data/lerobot/<TASK_NAME>/` 到 OSS

## 云端可开始条件

以上 4 项完成后，通知云端 AI 按 `LeTools_部署与训练方案.md` 第 6.4 节执行。
