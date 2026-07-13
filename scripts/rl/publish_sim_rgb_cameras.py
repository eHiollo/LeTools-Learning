#!/usr/bin/env python3
"""
Shadow MuJoCo RGB camera publisher for Kuavo-Sim.

Standard `load_kuavo_mujoco_sim.launch` only optionally enables waist *depth*
(`run_mujoco_camera:=true`). RGB topics `/cam_{h,l,r}/color/image_raw/compressed`
are not published by that node.

This node loads the same scene XML, syncs qpos from `/sensors_data_raw` (+ optional
root pose), renders offscreen, and publishes CompressedImage on the deploy topics.

Requires: mujoco, cv_bridge/OpenCV, rospy, kuavo_msgs.
"""

from __future__ import annotations

import argparse
import os
import threading
from typing import Dict, List, Optional

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage

# Must set before importing mujoco. Host often lacks OSMesa; EGL works with NVIDIA.
if not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pip/conda install mujoco first") from exc

from kuavo_msgs.msg import sensorsData


DEFAULT_SCENE = (
    "/home/fulin/VSCode/kuavo-ros-control/src/kuavo_assets/models/biped_s62/xml/scene.xml"
)

# Named cameras added to biped_s62.xml → ROS compressed topics used by kuavo_deploy
CAMERA_TOPICS = {
    "cam_h": "/cam_h/color/image_raw/compressed",
    "cam_l": "/cam_l/color/image_raw/compressed",
    "cam_r": "/cam_r/color/image_raw/compressed",
}

# Map sensors_data_raw joint_q indices → mujoco hinge joint names (s62 wheeled layout).
# Freejoint (root) is filled separately; wheels default to 0 if not present in msg.
#
# When len(joint_q)==28 (classic biped legs+arms+head while still on wrong robot_version),
# we only drive arms+head so the image at least tracks upper body.
S62_HINGE_FROM_SENSORS: Dict[str, int] = {
    "knee_joint": 0,
    "leg_joint": 1,
    "waist_pitch_joint": 2,
    "waist_yaw_joint": 3,
    "zarm_l1_joint": 4,
    "zarm_l2_joint": 5,
    "zarm_l3_joint": 6,
    "zarm_l4_joint": 7,
    "zarm_l5_joint": 8,
    "zarm_l6_joint": 9,
    "zarm_l7_joint": 10,
    "zarm_r1_joint": 11,
    "zarm_r2_joint": 12,
    "zarm_r3_joint": 13,
    "zarm_r4_joint": 14,
    "zarm_r5_joint": 15,
    "zarm_r6_joint": 16,
    "zarm_r7_joint": 17,
}

# Classic 28-D biped: 12 legs + 14 arms + 2 head
BIPED28_ARM_HEAD: Dict[str, int] = {
    "zarm_l1_joint": 12,
    "zarm_l2_joint": 13,
    "zarm_l3_joint": 14,
    "zarm_l4_joint": 15,
    "zarm_l5_joint": 16,
    "zarm_l6_joint": 17,
    "zarm_l7_joint": 18,
    "zarm_r1_joint": 19,
    "zarm_r2_joint": 20,
    "zarm_r3_joint": 21,
    "zarm_r4_joint": 22,
    "zarm_r5_joint": 23,
    "zarm_r6_joint": 24,
    "zarm_r7_joint": 25,
    "zhead_1_joint": 26,
    "zhead_2_joint": 27,
}


class SimRgbCameraPublisher:
    def __init__(self, scene: str, width: int, height: int, fps: float):
        self.lock = threading.Lock()
        self.latest_q: Optional[np.ndarray] = None
        self.model = mujoco.MjModel.from_xml_path(scene)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.fps = fps
        self.cam_ids: Dict[str, int] = {}
        for name in CAMERA_TOPICS:
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                rospy.logwarn("camera %s not found in model; skip", name)
                continue
            self.cam_ids[name] = cid
        if not self.cam_ids:
            raise RuntimeError(
                f"no RGB cameras found in {scene}; expected {list(CAMERA_TOPICS)}"
            )

        self.jnt_qposadr: Dict[str, int] = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.jnt_qposadr[name] = int(self.model.jnt_qposadr[i])

        self.pubs = {
            name: rospy.Publisher(topic, CompressedImage, queue_size=1)
            for name, topic in CAMERA_TOPICS.items()
            if name in self.cam_ids
        }
        rospy.Subscriber("/sensors_data_raw", sensorsData, self._on_sensors, queue_size=1)
        rospy.loginfo(
            "SimRgbCameraPublisher ready cams=%s scene=%s",
            list(self.cam_ids),
            scene,
        )

    def _on_sensors(self, msg: sensorsData) -> None:
        q = np.asarray(msg.joint_data.joint_q, dtype=np.float64)
        with self.lock:
            self.latest_q = q

    def _apply_joint_q(self, joint_q: np.ndarray) -> None:
        # Keep freejoint / wheels at current (or home) unless we have a better source.
        if self.data.qpos.shape[0] >= 7:
            # Prefer standing height if still zero
            if abs(self.data.qpos[2]) < 1e-6:
                self.data.qpos[2] = 0.98
            if abs(self.data.qpos[3]) < 1e-6:
                self.data.qpos[3] = 1.0  # quat w

        mapping = BIPED28_ARM_HEAD if joint_q.shape[0] >= 28 else S62_HINGE_FROM_SENSORS
        for jname, idx in mapping.items():
            if idx >= joint_q.shape[0]:
                continue
            adr = self.jnt_qposadr.get(jname)
            if adr is None:
                continue
            self.data.qpos[adr] = float(joint_q[idx])

        # If sensors looks like s62 18-D hinge block, also apply S62 map
        if joint_q.shape[0] in (18, 22, 26, 30):
            for jname, idx in S62_HINGE_FROM_SENSORS.items():
                if idx >= joint_q.shape[0]:
                    continue
                adr = self.jnt_qposadr.get(jname)
                if adr is not None:
                    self.data.qpos[adr] = float(joint_q[idx])

        mujoco.mj_forward(self.model, self.data)

    def _publish_jpeg(self, pub, rgb: np.ndarray, stamp) -> None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = stamp
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        pub.publish(msg)

    def spin(self) -> None:
        rate = rospy.Rate(self.fps)
        while not rospy.is_shutdown():
            with self.lock:
                q = None if self.latest_q is None else self.latest_q.copy()
            if q is not None:
                self._apply_joint_q(q)
            stamp = rospy.Time.now()
            for name, cid in self.cam_ids.items():
                self.renderer.update_scene(self.data, camera=cid)
                rgb = self.renderer.render()
                self._publish_jpeg(self.pubs[name], rgb, stamp)
            rate.sleep()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    # Match ACT / dataset IMAGE_SHAPE_CHW=(3,480,848)
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    rospy.init_node("kuavo_sim_rgb_cameras", anonymous=False)
    pub = SimRgbCameraPublisher(args.scene, args.width, args.height, args.fps)
    pub.spin()


if __name__ == "__main__":
    main()
