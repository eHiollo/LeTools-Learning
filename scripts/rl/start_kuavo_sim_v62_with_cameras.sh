#!/usr/bin/env bash
# Kuavo v62 wheel sim with native mujoco RGB cameras.
# Run on the ROS/kuavo side (host or container that owns the sim).
#
# Usage:
#   export ROBOT_VERSION=62
#   bash scripts/rl/start_kuavo_sim_v62_with_cameras.sh
#
# Or manually:
#   export ROBOT_VERSION=62
#   roslaunch humanoid_controllers load_kuavo_mujoco_sim_wheel.launch publish_camera:=true
#
# Host eval then uses:
#   configs/deploy/total/deploy_sim_mujoco_native_cams.yaml
# Host-side scripts/rl/run_sim_rgb_cameras.sh is NO LONGER required.
set -euo pipefail

export ROBOT_VERSION="${ROBOT_VERSION:-62}"

echo "[kuavo-sim] ROBOT_VERSION=${ROBOT_VERSION} (wheel / LUNBI_V62)"
echo "[kuavo-sim] launch=load_kuavo_mujoco_sim_wheel.launch publish_camera:=true"
echo "[kuavo-sim] topics: /camera /left_wrist_camera /right_wrist_camera color/image_raw"

if [[ -f /root/kuavo_ws/devel/setup.bash || -f /root/kuavo_ws/devel/setup.zsh ]]; then
  # Inside kuavo docker
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.zsh 2>/dev/null || source /opt/ros/noetic/setup.bash
  # shellcheck disable=SC1091
  source /root/kuavo_ws/devel/setup.zsh 2>/dev/null || source /root/kuavo_ws/devel/setup.bash
else
  # Host with local kuavo-ros-control workspace
  # shellcheck disable=SC1091
  source /opt/ros/noetic/setup.bash
  # shellcheck disable=SC1091
  source "${KUAVO_WS:-/home/fulin/VSCode/kuavo-ros-control}/devel/setup.bash"
fi

exec roslaunch humanoid_controllers load_kuavo_mujoco_sim_wheel.launch \
  publish_camera:=true \
  joystick_type:=sim
