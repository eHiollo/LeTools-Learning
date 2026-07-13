#!/usr/bin/env bash
# Kuavo v62 = 轮臂 (LUNBI_V62). Must use wheel launch, NOT bipedal load_kuavo_mujoco_sim.launch.
# Run INSIDE the kuavo docker container.
set -euo pipefail

export ROBOT_VERSION=62

echo "[kuavo-sim] ROBOT_VERSION=${ROBOT_VERSION} (wheel / LUNBI_V62)"
echo "[kuavo-sim] launch=load_kuavo_mujoco_sim_wheel.launch run_mujoco_camera:=true"
echo "[kuavo-sim] After up, on HOST: bash scripts/rl/run_sim_rgb_cameras.sh"

source /opt/ros/noetic/setup.zsh 2>/dev/null || source /opt/ros/noetic/setup.bash
source /root/kuavo_ws/devel/setup.zsh 2>/dev/null || source /root/kuavo_ws/devel/setup.bash

exec roslaunch humanoid_controllers load_kuavo_mujoco_sim_wheel.launch \
  robot_version:=62 \
  run_mujoco_camera:=true \
  joystick_type:=sim
