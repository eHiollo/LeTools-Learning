"""Optional ROS depth subscribers for multi-camera point clouds.

Isolated from ``hil_recording`` topic profiles — only used by RL-100 collect.
Supports raw ``sensor_msgs/Image`` and Kuavo ``compressedDepth`` (PNG in CompressedImage).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from kuavo_rl.rl100_zarr.config import RL100CollectConfig
from kuavo_rl.rl100_zarr.pointcloud import build_rl100_point_cloud, depth_to_point_cloud

_PNG_MAGIC = bytes([137, 80, 78, 71, 13, 10, 26, 10])


@dataclass
class _CamSample:
    depth_m: np.ndarray | None = None
    intrinsics: tuple[float, float, float, float] | None = None
    stamp: float = 0.0
    frame_id: str = ""


def _decode_compressed_depth(msg: Any) -> np.ndarray | None:
    """Decode Kuavo/RealSense compressedDepth → depth meters (H, W)."""
    import cv2

    data = bytes(getattr(msg, "data", b"") or b"")
    idx = data.find(_PNG_MAGIC)
    if idx < 0:
        return None
    image = cv2.imdecode(np.frombuffer(data[idx:], np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    depth = np.asarray(image)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    # RealSense compressedDepth is typically uint16 millimeters.
    if np.issubdtype(depth.dtype, np.integer) or float(np.nanmax(depth)) > 20.0:
        return depth.astype(np.float32) * 0.001
    return depth.astype(np.float32)


def _depth_m_from_image_msg(bridge: Any, msg: Any) -> np.ndarray | None:
    try:
        encoding = str(getattr(msg, "encoding", "")).lower()
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_m = np.asarray(depth, dtype=np.float32)
        if "16" in encoding or encoding in {"mono16", "16uc1"}:
            depth_m = depth_m * 0.001
        elif np.nanmax(depth_m) > 20.0:
            depth_m = depth_m * 0.001
        return depth_m
    except Exception:  # noqa: BLE001
        return None


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
        from sensor_msgs.msg import CameraInfo, CompressedImage, Image

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
            msg_type = (cam.depth_msg_type or "compressed_depth").strip().lower()
            if msg_type in {"compressed_depth", "compressed", "compresseddepth"}:
                depth_cls = CompressedImage
                cb = lambda msg, name=cam.name: self._on_depth_compressed(name, msg)
            elif msg_type in {"image", "raw", "sensor_image"}:
                depth_cls = Image
                cb = lambda msg, name=cam.name: self._on_depth_image(name, msg)
            else:
                raise ValueError(
                    f"camera {cam.name}: unknown depth_msg_type={cam.depth_msg_type!r} "
                    "(use compressed_depth or image)"
                )
            self._subs.append(
                rospy.Subscriber(cam.depth_topic, depth_cls, callback=cb, queue_size=1)
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

    def _store_depth(self, name: str, depth_m: np.ndarray, msg: Any) -> None:
        with self._lock:
            sample = self._samples.setdefault(name, _CamSample())
            sample.depth_m = depth_m
            sample.stamp = time.time()
            if getattr(msg, "header", None) is not None and msg.header.frame_id:
                sample.frame_id = str(msg.header.frame_id)

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

    def _on_depth_image(self, name: str, msg: Any) -> None:
        if self._bridge is None:
            return
        depth_m = _depth_m_from_image_msg(self._bridge, msg)
        if depth_m is None:
            return
        self._store_depth(name, depth_m, msg)

    def _on_depth_compressed(self, name: str, msg: Any) -> None:
        depth_m = _decode_compressed_depth(msg)
        if depth_m is None:
            return
        self._store_depth(name, depth_m, msg)

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

    def get_point_cloud(self, *, require_all_enabled: bool | None = None) -> np.ndarray:
        require_all = (
            self.config.require_all_cameras
            if require_all_enabled is None
            else bool(require_all_enabled)
        )
        clouds: list[np.ndarray] = []
        missing: list[str] = []
        with self._lock:
            snapshot = {
                k: _CamSample(
                    v.depth_m.copy() if v.depth_m is not None else None,
                    v.intrinsics,
                    v.stamp,
                    v.frame_id,
                )
                for k, v in self._samples.items()
            }

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
            clouds.append(depth_to_point_cloud(sample.depth_m, sample.intrinsics, T))

        if require_all and missing:
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
            raise_on_empty=self.config.fail_on_empty_pointcloud,
            min_points=self.config.min_workspace_points,
        )

    def preflight(self, timeout_s: float = 5.0) -> dict[str, Any]:
        """Wait briefly and report which cameras have depth+info (+tf attempt)."""
        deadline = time.time() + timeout_s
        report: dict[str, Any] = {
            "cameras": {},
            "ok": False,
            "require_all_cameras": self.config.require_all_cameras,
        }
        while time.time() < deadline:
            with self._lock:
                for name, sample in self._samples.items():
                    report["cameras"][name] = {
                        "has_depth": sample.depth_m is not None,
                        "has_intrinsics": sample.intrinsics is not None,
                        "frame_id": sample.frame_id,
                        "age_s": (time.time() - sample.stamp) if sample.stamp else None,
                    }
            ready = [
                v
                for v in report["cameras"].values()
                if v["has_depth"] and v["has_intrinsics"]
            ]
            if self.config.require_all_cameras:
                enabled_n = sum(1 for c in self.config.cameras if c.enabled)
                if len(ready) >= enabled_n:
                    break
            elif ready:
                break
            time.sleep(0.1)
        for cam in self.config.cameras:
            if not cam.enabled:
                continue
            info = report["cameras"].setdefault(cam.name, {})
            frame = info.get("frame_id") or cam.frame_id
            T = self._lookup_T_base_cam(frame) if frame else None
            info["has_tf"] = T is not None
            info["depth_topic"] = cam.depth_topic
            info["depth_msg_type"] = cam.depth_msg_type
        cams_ok = [
            c.get("has_depth") and c.get("has_intrinsics") and c.get("has_tf")
            for name, c in report["cameras"].items()
            if any(cam.name == name and cam.enabled for cam in self.config.cameras)
        ]
        if self.config.require_all_cameras:
            report["ok"] = bool(cams_ok) and all(cams_ok)
        else:
            report["ok"] = any(cams_ok)
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
