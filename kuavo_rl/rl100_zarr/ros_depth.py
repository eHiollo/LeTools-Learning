"""Optional ROS depth subscribers for multi-camera point clouds.

Isolated from ``hil_recording`` topic profiles — only used by RL-100 collect.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kuavo_rl.rl100_zarr.config import CameraPCConfig, RL100CollectConfig
from kuavo_rl.rl100_zarr.pointcloud import build_rl100_point_cloud, depth_to_point_cloud


@dataclass
class _CamSample:
    depth_m: np.ndarray | None = None
    intrinsics: tuple[float, float, float, float] | None = None
    stamp: float = 0.0
    frame_id: str = ""


class DepthPointCloudHub:
    """Subscribe to configured depth + camera_info topics; fuse on demand."""

    def __init__(self, config: RL100CollectConfig):
        self.config = config
        self._lock = threading.Lock()
        self._samples: dict[str, _CamSample] = {
            c.name: _CamSample(frame_id=c.frame_id) for c in config.cameras if c.enabled
        }
        self._subs: list[Any] = []
        self._tf_buffer: Any | None = None
        self._tf_listener: Any | None = None
        self._ros: Any | None = None
        self._bridge: Any | None = None

    def start(self) -> None:
        if self._ros is not None:
            return
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import CameraInfo, Image

        try:
            import tf2_ros
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("tf2_ros required for multi-camera extrinsics") from exc

        self._ros = rospy
        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        for cam in self.config.cameras:
            if not cam.enabled:
                continue
            self._subs.append(
                rospy.Subscriber(
                    cam.depth_topic,
                    Image,
                    callback=lambda msg, name=cam.name: self._on_depth(name, msg),
                    queue_size=1,
                )
            )
            self._subs.append(
                rospy.Subscriber(
                    cam.camera_info_topic,
                    CameraInfo,
                    callback=lambda msg, name=cam.name: self._on_info(name, msg),
                    queue_size=1,
                )
            )

    def close(self) -> None:
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:  # noqa: BLE001
                pass
        self._subs = []
        self._tf_listener = None
        self._tf_buffer = None
        self._ros = None
        self._bridge = None

    def _on_info(self, name: str, msg: Any) -> None:
        K = getattr(msg, "K", None)
        if K is None or len(K) < 8:
            return
        fx, fy, cx, cy = float(K[0]), float(K[4]), float(K[2]), float(K[5])
        with self._lock:
            sample = self._samples.setdefault(name, _CamSample())
            sample.intrinsics = (fx, fy, cx, cy)
            if getattr(msg, "header", None) is not None and msg.header.frame_id:
                sample.frame_id = str(msg.header.frame_id)

    def _on_depth(self, name: str, msg: Any) -> None:
        if self._bridge is None:
            return
        try:
            encoding = str(getattr(msg, "encoding", "")).lower()
            if "16" in encoding or encoding in {"mono16", "16uc1"}:
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth_m = np.asarray(depth, dtype=np.float32) * 0.001
            else:
                depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                depth_m = np.asarray(depth, dtype=np.float32)
                # Heuristic: large values likely millimeters.
                if np.nanmax(depth_m) > 20.0:
                    depth_m = depth_m * 0.001
        except Exception:  # noqa: BLE001
            return
        with self._lock:
            sample = self._samples.setdefault(name, _CamSample())
            sample.depth_m = depth_m
            sample.stamp = time.time()
            if getattr(msg, "header", None) is not None and msg.header.frame_id:
                sample.frame_id = str(msg.header.frame_id)

    def _lookup_T_base_cam(self, frame_id: str) -> np.ndarray | None:
        if self._tf_buffer is None or not frame_id:
            return None
        try:
            import rospy

            tf = self._tf_buffer.lookup_transform(
                self.config.base_frame,
                frame_id,
                rospy.Time(0),
                rospy.Duration(0.05),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            return _quat_pose_to_mat(
                (t.x, t.y, t.z),
                (q.x, q.y, q.z, q.w),
            )
        except Exception:  # noqa: BLE001
            return None

    def get_point_cloud(self, *, require_all_enabled: bool = False) -> np.ndarray:
        clouds: list[np.ndarray] = []
        missing: list[str] = []
        with self._lock:
            snapshot = {k: _CamSample(v.depth_m.copy() if v.depth_m is not None else None,
                                      v.intrinsics, v.stamp, v.frame_id)
                        for k, v in self._samples.items()}

        for cam in self.config.cameras:
            if not cam.enabled:
                continue
            sample = snapshot.get(cam.name)
            if sample is None or sample.depth_m is None or sample.intrinsics is None:
                missing.append(cam.name)
                continue
            frame = sample.frame_id or cam.frame_id
            T = self._lookup_T_base_cam(frame)
            if T is None:
                missing.append(f"{cam.name}:tf")
                continue
            clouds.append(
                depth_to_point_cloud(sample.depth_m, sample.intrinsics, T)
            )

        if require_all_enabled and missing:
            raise RuntimeError(f"missing depth/tf for cameras: {missing}")
        if not clouds:
            raise RuntimeError(
                "no usable depth cameras; check topics / tf / camera_info "
                f"(missing={missing})"
            )

        x_range, y_range, z_range = self.config.workspace_ranges()
        return build_rl100_point_cloud(
            clouds,
            num_points=self.config.num_points,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
        )

    def preflight(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """Wait briefly and report which cameras have depth+info (+tf attempt)."""
        deadline = time.time() + timeout_s
        report: dict[str, Any] = {"cameras": {}, "ok": False}
        while time.time() < deadline:
            with self._lock:
                for name, sample in self._samples.items():
                    report["cameras"][name] = {
                        "has_depth": sample.depth_m is not None,
                        "has_intrinsics": sample.intrinsics is not None,
                        "frame_id": sample.frame_id,
                        "age_s": (time.time() - sample.stamp) if sample.stamp else None,
                    }
            if any(v["has_depth"] and v["has_intrinsics"] for v in report["cameras"].values()):
                break
            time.sleep(0.1)
        # tf probe
        for cam in self.config.cameras:
            if not cam.enabled:
                continue
            info = report["cameras"].setdefault(cam.name, {})
            frame = info.get("frame_id") or cam.frame_id
            T = self._lookup_T_base_cam(frame) if frame else None
            info["has_tf"] = T is not None
        report["ok"] = any(
            c.get("has_depth") and c.get("has_intrinsics") and c.get("has_tf")
            for c in report["cameras"].values()
        )
        return report


def _quat_pose_to_mat(
    xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return T
