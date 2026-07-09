# 阶段 1 执行状态

执行时间：2026-07-09

## 验收标准

| 项 | 状态 | 说明 |
|----|------|------|
| `docker/Dockerfile.letools` | ✅ | 非交互安装脚本 `docker/setup_env_docker.sh` |
| 镜像构建 | ✅ | `letools-train:lerobot-0.4.2`（约 40GB） |
| 容器内 CUDA | ✅ | `torch.cuda.is_available() == True` |
| compose 文件 | ✅ | `docker/docker-compose.train.yml` |
| smoke 配置 | ✅ | `configs/train/smoke/` |

## 构建命令

```bash
cd ~/robot-il/LeTools-Learning
docker build -f docker/Dockerfile.letools --build-arg CUDA_VER=12.8 -t letools-train:lerobot-0.4.2 .
```

## 验证 CUDA

```bash
docker run --rm --gpus all letools-train:lerobot-0.4.2 bash -lc \
  "source /opt/conda/etc/profile.d/conda.sh && conda activate letools && \
   python -c \"import torch; print(torch.cuda.is_available())\""
```

## 阶段 2 smoke test（下一步）

```bash
cd ~/robot-il/LeTools-Learning
docker compose -f docker/docker-compose.train.yml run --rm train \
  bash -lc "source /opt/conda/etc/profile.d/conda.sh && conda activate letools && \
    python kuavo_model/train.py --policy act --mode simple --config-root configs/train/smoke"
```

## 构建说明

- **跳过** `requirements_ros_env.txt`：其中 ROS 包无法从 PyPI 安装；训练不依赖，部署时 `source /opt/ros/noetic/setup.bash`。
- **跳过** `flash-attn`：gr00t 策略按需另建镜像。
- 日志：`docker/build.log`

## 推送 ACR（上云前，填好 deploy.env 后）

```bash
source deploy.env
docker tag letools-train:lerobot-0.4.2 ${LETOOLS_IMAGE}
docker login ${ACR_REGISTRY}
docker push ${LETOOLS_IMAGE}
```
