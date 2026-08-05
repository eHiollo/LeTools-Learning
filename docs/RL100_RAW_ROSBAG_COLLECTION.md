# RL-100 原始 rosbag 采集

RL-100 实机采集默认使用 Brain 风格链路：录制时只让 `rosbag record` 异步写入原始消息；episode 结束后再按参考相机时间戳对齐到 YAML 中的 `fps: 10`，最后离线生成点云和 topic-native NPZ/Zarr。采集期间不创建 Gym 部署环境，也不会在线执行 freshness、深度解码、TF 查询或 FPS 点云采样。

## 机器人端采集

先按原来的方式启动 ROS、相机和 VR 遥操作，然后执行：

```bash
bash scripts/rl/run_rl100_zarr_collect.sh preflight
bash scripts/rl/run_rl100_zarr_collect.sh collect --confirm-live
```

每个已标注 episode 会产生：

```text
data/rl100/grasp_8_4_v2/raw_bags/<episode_id>.bag
data/rl100/grasp_8_4_v2/raw_bags/<episode_id>.json
```

`.json` 保存 success/failure/rerecord 标签；未成功结束的 bag 不会进入训练集。当前 upper-cams 配置的手势是：Y 开始、Y 双击重录、右摇杆下 success、右摇杆上 failure；若把 `reward_gesture` 改成 `button`，则使用 B 短按/长按。

## 离线转换与构建

采集完成后，在具备 ROS Python、`cv_bridge` 和 `tf2_ros` 的环境执行：

```bash
bash scripts/rl/run_rl100_zarr_collect.sh build \
  --config configs/rl/rl100_zarr_collect_upper_cams.yaml \
  --overwrite
```

转换规则固定为：参考相机 `camera_sync.reference_camera` → 时间降采样到 `fps` → state 使用不晚于参考帧的最新 `/sensors_data_raw` 与 `/dexhand/state` → action 使用不晚于参考帧的最新 `/kuavo_arm_traj` 与 `/control_robot_hand_position`；首条 command 之前不写帧，夹爪首条 command 之前填 YAML 的 `hand_default`。点云只在 bag 关闭后用录制的 `/tf`、`/tf_static` 生成。

如需保留旧的在线点云链路，仅在实验配置中把 `collection_mode` 改成 `online_pointcloud`。
