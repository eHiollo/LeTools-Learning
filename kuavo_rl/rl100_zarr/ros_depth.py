"""ROS depth buffering, causal approximate synchronization and point clouds.

The RL-100 real robot path uses raw ``sensor_msgs/Image`` depth topics by
default.  CompressedDepth remains supported for explicit legacy configurations,
but it is not the synchronization default because its transport/decoding delay
can make a latest-frame fusion pair different sampling instants.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from kuavo_rl.rl100_zarr.config import RL100CollectConfig
from kuavo_rl.rl100_zarr.pointcloud import build_rl100_point_cloud, depth_to_point_cloud

_PNG_MAGIC = bytes([137, 80, 78, 71, 13, 10, 26, 10])
_DEPTH_SUBSCRIBER_BUFF_SIZE = 16 * 1024 * 1024


def depth_subscriber_kwargs() -> dict[str, int]:
    """Keep only the latest depth frame without a small TCP receive backlog."""

    return {"queue_size": 1, "buff_size": _DEPTH_SUBSCRIBER_BUFF_SIZE}


class CameraSyncError(RuntimeError):
    """Structured failure raised when a synchronized camera bundle is invalid."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        self.code = str(code)
        self.details = dict(details)
        super().__init__(f"camera_sync[{self.code}]: {message}")


@dataclass(frozen=True)
class TimedDepthFrame:
    camera_name: str
    depth_m: np.ndarray
    intrinsics: tuple[float, float, float, float] | None
    frame_id: str
    header_stamp_s: float
    received_wall_s: float
    received_monotonic_s: float
    sequence: int | None = None


@dataclass(frozen=True)
class SynchronizedDepthBundle:
    frames: dict[str, TimedDepthFrame]
    reference_camera: str
    reference_stamp_s: float
    max_header_skew_s: float
    max_receive_skew_s: float
    oldest_received_age_s: float
    camera_header_stamps: dict[str, float]
    camera_received_wall_s: dict[str, float]
    camera_received_monotonic_s: dict[str, float]
    camera_received_ages_s: dict[str, float]


@dataclass
class _CamSample:
    """Latest-frame compatibility view plus timing fields used by preflight."""

    depth_m: np.ndarray | None = None
    intrinsics: tuple[float, float, float, float] | None = None
    stamp: float = 0.0
    received_at: float = 0.0  # legacy wall-clock alias
    frame_id: str = ""
    received_wall_s: float = 0.0
    received_monotonic_s: float = 0.0
    sequence: int | None = None


@dataclass(frozen=True)
class PointCloudSample:
    """Fused cloud and timing metadata shared by collection and deployment."""

    points: np.ndarray
    fused_stamp_s: float
    received_at_s: float
    oldest_age_s: float
    max_camera_skew_s: float
    camera_stamps: dict[str, float]
    valid_points: int
    reference_camera: str = ""
    max_receive_skew_s: float = 0.0
    camera_received_wall_s: dict[str, float] = field(default_factory=dict)
    camera_received_monotonic_s: dict[str, float] = field(default_factory=dict)
    camera_received_ages_s: dict[str, float] = field(default_factory=dict)
    generated_wall_s: float = 0.0
    generated_monotonic_s: float = 0.0

    @property
    def reference_stamp_s(self) -> float:
        return float(self.fused_stamp_s)

    @property
    def max_header_skew_s(self) -> float:
        return float(self.max_camera_skew_s)

    @property
    def camera_header_stamps(self) -> dict[str, float]:
        return self.camera_stamps


def _stamp_s(msg: Any) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    try:
        value = float(stamp.to_sec()) if stamp is not None else 0.0
    except Exception:  # noqa: BLE001
        try:
            value = float(stamp)
        except Exception:  # noqa: BLE001
            value = 0.0
    return value if np.isfinite(value) and value > 0.0 else 0.0


def _copy_frame(frame: TimedDepthFrame) -> TimedDepthFrame:
    return TimedDepthFrame(
        camera_name=frame.camera_name,
        depth_m=np.asarray(frame.depth_m, dtype=np.float32).copy(),
        intrinsics=frame.intrinsics,
        frame_id=frame.frame_id,
        header_stamp_s=float(frame.header_stamp_s),
        received_wall_s=float(frame.received_wall_s),
        received_monotonic_s=float(frame.received_monotonic_s),
        sequence=frame.sequence,
    )


def _candidates_before(
    frames: Sequence[TimedDepthFrame],
    *,
    cutoff_monotonic_s: float,
    now_monotonic_s: float,
    max_received_age_s: float,
) -> list[TimedDepthFrame]:
    candidates: list[TimedDepthFrame] = []
    for frame in frames:
        if frame.received_monotonic_s > cutoff_monotonic_s + 1e-9:
            continue
        if frame.intrinsics is None or not frame.frame_id:
            continue
        age = now_monotonic_s - frame.received_monotonic_s
        if age < -1e-3 or age > max_received_age_s:
            continue
        if not np.isfinite(frame.header_stamp_s) or frame.header_stamp_s <= 0.0:
            continue
        candidates.append(frame)
    return candidates


def select_synchronized_frames(
    histories: Mapping[str, Sequence[TimedDepthFrame]],
    *,
    cutoff_monotonic_s: float,
    reference_camera: str,
    max_received_age_s: float,
    max_header_skew_s: float,
    max_receive_skew_s: float | None = None,
    require_same_time_epoch: bool = True,
    now_monotonic_s: float | None = None,
) -> SynchronizedDepthBundle:
    """Select a causal, nearest-timestamp frame bundle from camera histories.

    The function has no ROS dependency and is intentionally deterministic.  It
    first chooses the newest valid reference-camera frame before the sampler
    cutoff, then chooses each other camera's frame nearest to that header stamp.
    """

    if not histories:
        raise CameraSyncError("no_cameras", "no camera histories are available")
    if reference_camera not in histories:
        raise CameraSyncError("reference_missing", f"reference camera {reference_camera!r} is missing")
    if max_received_age_s <= 0.0 or max_header_skew_s <= 0.0:
        raise ValueError("camera synchronization limits must be positive")
    now_mono = time.monotonic() if now_monotonic_s is None else float(now_monotonic_s)
    candidates: dict[str, list[TimedDepthFrame]] = {}
    for name, frames in histories.items():
        candidates[name] = _candidates_before(
            frames,
            cutoff_monotonic_s=float(cutoff_monotonic_s),
            now_monotonic_s=now_mono,
            max_received_age_s=float(max_received_age_s),
        )

    reference_candidates = candidates.get(reference_camera, [])
    if not reference_candidates:
        raise CameraSyncError(
            "reference_stale",
            f"reference camera {reference_camera!r} has no fresh frame before cutoff",
            reference_camera=reference_camera,
        )
    reference = max(reference_candidates, key=lambda frame: frame.received_monotonic_s)
    selected: dict[str, TimedDepthFrame] = {reference_camera: reference}
    for name, values in candidates.items():
        if name == reference_camera:
            continue
        if not values:
            raise CameraSyncError("camera_missing_or_stale", f"camera {name!r} has no fresh frame")
        selected[name] = min(
            values,
            key=lambda frame: (
                abs(frame.header_stamp_s - reference.header_stamp_s),
                frame.header_stamp_s,
                frame.received_monotonic_s,
            ),
        )

    stamps = {name: float(frame.header_stamp_s) for name, frame in selected.items()}
    received_wall = {name: float(frame.received_wall_s) for name, frame in selected.items()}
    received_mono = {name: float(frame.received_monotonic_s) for name, frame in selected.items()}
    ages = {name: max(0.0, now_mono - value) for name, value in received_mono.items()}
    if require_same_time_epoch:
        magnitudes = [abs(value) for value in stamps.values() if abs(value) > 0.0]
        if magnitudes and max(magnitudes) / min(magnitudes) > 1000.0:
            raise CameraSyncError(
                "time_epoch",
                "camera header stamps are not in the same time epoch",
                camera_stamps=stamps,
            )
    header_skew = max(stamps.values()) - min(stamps.values())
    receive_skew = max(received_mono.values()) - min(received_mono.values())
    if header_skew > max_header_skew_s:
        raise CameraSyncError(
            "header_skew",
            f"selected camera header skew {header_skew:.3f}s > {max_header_skew_s:.3f}s",
            camera_stamps=stamps,
            max_header_skew_s=header_skew,
        )
    if max_receive_skew_s is not None and receive_skew > max_receive_skew_s:
        raise CameraSyncError(
            "receive_skew",
            f"selected camera receive skew {receive_skew:.3f}s > {max_receive_skew_s:.3f}s",
            max_receive_skew_s=receive_skew,
        )
    return SynchronizedDepthBundle(
        frames={name: _copy_frame(frame) for name, frame in selected.items()},
        reference_camera=reference_camera,
        reference_stamp_s=float(reference.header_stamp_s),
        max_header_skew_s=float(header_skew),
        max_receive_skew_s=float(receive_skew),
        oldest_received_age_s=max(ages.values()),
        camera_header_stamps=stamps,
        camera_received_wall_s=received_wall,
        camera_received_monotonic_s=received_mono,
        camera_received_ages_s=ages,
    )


def _decode_compressed_depth(msg: Any) -> np.ndarray | None:
    """Decode Kuavo/RealSense compressedDepth -> depth meters (H, W)."""

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
    if depth.size == 0 or not np.isfinite(depth).all():
        return None
    if np.issubdtype(depth.dtype, np.integer) or float(np.nanmax(depth)) > 20.0:
        return depth.astype(np.float32) * 0.001
    return depth.astype(np.float32)


def _depth_m_from_image_msg(bridge: Any, msg: Any) -> np.ndarray | None:
    try:
        encoding = str(getattr(msg, "encoding", "")).lower()
        depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_m = np.asarray(depth, dtype=np.float32)
        if depth_m.ndim != 2 or depth_m.size == 0 or not np.isfinite(depth_m).all():
            return None
        if "16" in encoding or encoding in {"mono16", "16uc1"}:
            depth_m = depth_m * 0.001
        elif float(np.nanmax(depth_m)) > 20.0:
            depth_m = depth_m * 0.001
        return depth_m
    except Exception:  # noqa: BLE001
        return None


class DepthPointCloudHub:
    """Subscribe to depth topics, select synchronized frames and fuse clouds."""

    def __init__(self, config: RL100CollectConfig):
        self.config = config
        self._lock = threading.RLock()
        enabled = [c for c in config.cameras if c.enabled]
        self._samples: dict[str, _CamSample] = {
            c.name: _CamSample(frame_id=c.frame_id) for c in enabled
        }
        self._histories: dict[str, deque[TimedDepthFrame]] = {
            c.name: deque(maxlen=int(config.camera_buffer_size)) for c in enabled
        }
        self._receive_monotonic: dict[str, deque[float]] = {
            c.name: deque(maxlen=4096) for c in enabled
        }
        self._last_header_stamp: dict[str, float] = {c.name: 0.0 for c in enabled}
        self.invalid_counts: Counter[str] = Counter()
        self._subs: list[Any] = []
        self._tf_buffer: Any | None = None
        self._tf_listener: Any | None = None
        self._ros: Any | None = None
        self._bridge: Any | None = None
        self._sequence = 0

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
            msg_type = (cam.depth_msg_type or "image").strip().lower()
            if msg_type in {"compressed_depth", "compressed", "compresseddepth"}:
                depth_cls = CompressedImage
                callback = lambda msg, name=cam.name: self._on_depth_compressed(name, msg)
            elif msg_type in {"image", "raw", "sensor_image"}:
                depth_cls = Image
                callback = lambda msg, name=cam.name: self._on_depth_image(name, msg)
            else:
                raise ValueError(
                    f"camera {cam.name}: unknown depth_msg_type={cam.depth_msg_type!r} "
                    "(use image or compressed_depth)"
                )
            self._subs.append(
                rospy.Subscriber(
                    cam.depth_topic,
                    depth_cls,
                    callback=callback,
                    **depth_subscriber_kwargs(),
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

    def _store_depth(self, name: str, depth_m: np.ndarray, msg: Any) -> None:
        wall = time.time()
        monotonic = time.monotonic()
        stamp = _stamp_s(msg)
        if stamp <= 0.0:
            self.invalid_counts["header_stamp_invalid"] += 1
            if self.config.camera_sync_mode == "buffered_header":
                return
            stamp = wall
        if not np.isfinite(depth_m).all() or depth_m.ndim != 2:
            self.invalid_counts["depth_invalid"] += 1
            return
        header = getattr(msg, "header", None)
        frame_id = str(getattr(header, "frame_id", "") or "")
        with self._lock:
            previous_stamp = self._last_header_stamp.get(name, 0.0)
            if (
                self.config.camera_require_monotonic_header
                and previous_stamp > 0.0
                and stamp < previous_stamp - 1e-6
            ):
                self.invalid_counts[f"{name}_header_backwards"] += 1
                return
            self._last_header_stamp[name] = stamp
            sample = self._samples.setdefault(name, _CamSample())
            self._sequence += 1
            sequence = self._sequence
            sample.depth_m = np.asarray(depth_m, dtype=np.float32).copy()
            sample.stamp = stamp
            sample.received_at = wall
            sample.received_wall_s = wall
            sample.received_monotonic_s = monotonic
            sample.sequence = sequence
            if frame_id:
                sample.frame_id = frame_id
            frame_id = sample.frame_id or next(
                (c.frame_id for c in self.config.cameras if c.name == name), ""
            )
            self._histories.setdefault(name, deque(maxlen=int(self.config.camera_buffer_size))).append(
                TimedDepthFrame(
                    camera_name=name,
                    # The callback owns this array and never mutates it after
                    # enqueueing; avoid a second full-resolution copy per frame.
                    depth_m=sample.depth_m,
                    intrinsics=sample.intrinsics,
                    frame_id=frame_id,
                    header_stamp_s=stamp,
                    received_wall_s=wall,
                    received_monotonic_s=monotonic,
                    sequence=sequence,
                )
            )
            self._receive_monotonic.setdefault(name, deque(maxlen=4096)).append(monotonic)

    def _on_info(self, name: str, msg: Any) -> None:
        K = getattr(msg, "K", None)
        if K is None or len(K) < 8:
            self.invalid_counts["camera_info_invalid"] += 1
            return
        intrinsics = (float(K[0]), float(K[4]), float(K[2]), float(K[5]))
        if not np.isfinite(intrinsics).all() or intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
            self.invalid_counts["camera_info_invalid"] += 1
            return
        with self._lock:
            sample = self._samples.setdefault(name, _CamSample())
            sample.intrinsics = intrinsics
            header = getattr(msg, "header", None)
            if header is not None and getattr(header, "frame_id", ""):
                sample.frame_id = str(header.frame_id)

    def _on_depth_image(self, name: str, msg: Any) -> None:
        if self._bridge is None:
            return
        depth_m = _depth_m_from_image_msg(self._bridge, msg)
        if depth_m is not None:
            self._store_depth(name, depth_m, msg)
        else:
            self.invalid_counts["depth_decode_failed"] += 1

    def _on_depth_compressed(self, name: str, msg: Any) -> None:
        depth_m = _decode_compressed_depth(msg)
        if depth_m is not None:
            self._store_depth(name, depth_m, msg)
        else:
            self.invalid_counts["depth_decode_failed"] += 1

    def _lookup_T_base_cam(self, frame_id: str, stamp_s: float | None = None) -> np.ndarray | None:
        if self._tf_buffer is None or not frame_id:
            return None
        try:
            import rospy

            query_time = (
                rospy.Time.from_sec(float(stamp_s))
                if stamp_s is not None and self.config.camera_tf_at_image_stamp
                else rospy.Time(0)
            )
            tf = self._tf_buffer.lookup_transform(
                self.config.base_frame,
                frame_id,
                query_time,
                rospy.Duration(float(self.config.camera_tf_timeout_s)),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            return _quat_pose_to_mat((t.x, t.y, t.z), (q.x, q.y, q.z, q.w))
        except Exception:  # noqa: BLE001
            return None

    def _frame_from_sample(self, name: str, sample: _CamSample) -> TimedDepthFrame | None:
        if sample.depth_m is None or sample.intrinsics is None:
            return None
        wall = sample.received_wall_s or sample.received_at or time.time()
        now_wall = time.time()
        age = max(0.0, now_wall - wall) if wall > 1e8 else 0.0
        received_mono = sample.received_monotonic_s or max(0.0, time.monotonic() - age)
        stamp = sample.stamp if sample.stamp > 0.0 else wall
        frame_id = sample.frame_id or next(
            (c.frame_id for c in self.config.cameras if c.name == name), ""
        )
        return TimedDepthFrame(
            camera_name=name,
            depth_m=np.asarray(sample.depth_m, dtype=np.float32).copy(),
            intrinsics=sample.intrinsics,
            frame_id=frame_id,
            header_stamp_s=float(stamp),
            received_wall_s=float(wall),
            received_monotonic_s=float(received_mono),
            sequence=sample.sequence,
        )

    def _history_snapshot(self, names: Sequence[str]) -> dict[str, list[TimedDepthFrame]]:
        with self._lock:
            histories: dict[str, list[TimedDepthFrame]] = {}
            for name in names:
                values = list(self._histories.get(name, ()))
                if not values:
                    fallback = self._frame_from_sample(name, self._samples.get(name, _CamSample()))
                    if fallback is not None:
                        values = [fallback]
                # TimedDepthFrame instances are immutable from the hub's
                # perspective and callbacks replace, rather than mutate, the
                # underlying arrays. Only the selected bundle is copied below.
                histories[name] = values
            return histories

    def _select_bundle(
        self,
        *,
        require_all: bool,
        cutoff_monotonic_s: float | None = None,
        max_received_age_s: float | None = None,
        max_header_skew_s: float | None = None,
        max_receive_skew_s: float | None = None,
    ) -> SynchronizedDepthBundle:
        enabled = [c.name for c in self.config.cameras if c.enabled]
        histories = self._history_snapshot(enabled)
        if not require_all:
            histories = {name: value for name, value in histories.items() if value}
        if not histories or (require_all and len(histories) != len(enabled)):
            missing = [name for name in enabled if not histories.get(name)]
            raise CameraSyncError("camera_missing", f"missing camera histories: {missing}")
        if self.config.camera_sync_mode == "latest_legacy":
            # Explicit rollback mode: retain only each camera's latest frame,
            # while still applying the configured freshness/skew checks.
            histories = {
                name: [max(values, key=lambda frame: frame.received_monotonic_s)]
                for name, values in histories.items()
            }
        reference = self.config.camera_reference_camera
        if reference not in histories and len(histories) == 1:
            # Keep single-camera/unit-test configurations ergonomic while the
            # real three-camera config remains explicit about head_cam_h.
            reference = next(iter(histories))
        if reference not in histories:
            raise CameraSyncError("reference_missing", f"reference camera {reference!r} is unavailable")
        return select_synchronized_frames(
            histories,
            cutoff_monotonic_s=time.monotonic() if cutoff_monotonic_s is None else cutoff_monotonic_s,
            reference_camera=reference,
            max_received_age_s=(
                self.config.camera_max_received_age_s
                if max_received_age_s is None
                else float(max_received_age_s)
            ),
            max_header_skew_s=(
                self.config.camera_max_header_skew_s
                if max_header_skew_s is None
                else float(max_header_skew_s)
            ),
            max_receive_skew_s=(
                self.config.camera_max_receive_skew_s
                if max_receive_skew_s is None
                else float(max_receive_skew_s)
            ),
            require_same_time_epoch=self.config.camera_require_same_time_epoch,
        )

    def _fuse_bundle(self, bundle: SynchronizedDepthBundle, *, require_all: bool) -> PointCloudSample:
        transforms = self._bundle_transforms(bundle, require_all=require_all)
        clouds: list[np.ndarray] = []
        for name, frame in bundle.frames.items():
            transform = transforms.get(name)
            if transform is None:
                continue
            clouds.append(
                depth_to_point_cloud(
                    frame.depth_m,
                    frame.intrinsics,
                    transform,
                    candidate_count=self.config.pointcloud_candidate_pixels_per_camera,
                )
            )
        if not clouds:
            raise CameraSyncError("no_cloud", "no camera produced a point cloud")
        x_range, y_range, z_range = self.config.workspace_ranges()
        points = build_rl100_point_cloud(
            clouds,
            num_points=self.config.num_points,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            raise_on_empty=self.config.fail_on_empty_pointcloud,
            min_points=self.config.min_workspace_points,
        )
        generated_wall = time.time()
        generated_monotonic = time.monotonic()
        return PointCloudSample(
            points=points,
            fused_stamp_s=bundle.reference_stamp_s,
            received_at_s=generated_wall,
            oldest_age_s=bundle.oldest_received_age_s,
            max_camera_skew_s=bundle.max_header_skew_s,
            camera_stamps=bundle.camera_header_stamps,
            valid_points=int(points.shape[0]),
            reference_camera=bundle.reference_camera,
            max_receive_skew_s=bundle.max_receive_skew_s,
            camera_received_wall_s=bundle.camera_received_wall_s,
            camera_received_monotonic_s=bundle.camera_received_monotonic_s,
            camera_received_ages_s=bundle.camera_received_ages_s,
            generated_wall_s=generated_wall,
            generated_monotonic_s=generated_monotonic,
        )

    def _bundle_transforms(
        self,
        bundle: SynchronizedDepthBundle,
        *,
        require_all: bool,
    ) -> dict[str, np.ndarray]:
        transforms: dict[str, np.ndarray] = {}
        for name, frame in bundle.frames.items():
            transform = self._lookup_T_base_cam(
                frame.frame_id,
                frame.header_stamp_s if self.config.camera_tf_at_image_stamp else None,
            )
            if transform is None:
                if require_all:
                    raise CameraSyncError(
                        "tf_at_stamp",
                        f"TF unavailable for {name} at {frame.header_stamp_s:.6f}s",
                        camera=name,
                        stamp_s=frame.header_stamp_s,
                    )
                continue
            transforms[name] = transform
        return transforms

    def get_point_cloud_sample(
        self,
        *,
        require_all_enabled: bool | None = None,
        max_depth_age_s: float | None = None,
        max_camera_skew_s: float | None = None,
        cutoff_monotonic_s: float | None = None,
        max_header_skew_s: float | None = None,
        max_received_age_s: float | None = None,
        max_receive_skew_s: float | None = None,
    ) -> PointCloudSample:
        """Build one synchronized point cloud sample.

        ``max_depth_age_s`` and ``max_camera_skew_s`` are retained as explicit
        compatibility overrides. New callers should use the nested config and
        pass only ``cutoff_monotonic_s``.
        """

        require_all = self.config.require_all_cameras if require_all_enabled is None else bool(require_all_enabled)
        if max_received_age_s is None and max_depth_age_s is not None:
            max_received_age_s = float(max_depth_age_s)
        if max_header_skew_s is None and max_camera_skew_s is not None:
            max_header_skew_s = float(max_camera_skew_s)
        bundle = self._select_bundle(
            require_all=require_all,
            cutoff_monotonic_s=cutoff_monotonic_s,
            max_received_age_s=max_received_age_s,
            max_header_skew_s=max_header_skew_s,
            max_receive_skew_s=max_receive_skew_s,
        )
        return self._fuse_bundle(bundle, require_all=require_all)

    def get_point_cloud(self, *, require_all_enabled: bool | None = None) -> np.ndarray:
        """Compatibility wrapper for callers that do not need timing metadata."""

        return self.get_point_cloud_sample(require_all_enabled=require_all_enabled).points

    @staticmethod
    def _summary(values: Sequence[float]) -> dict[str, float | int | None]:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            return {"count": 0, "p50_s": None, "p95_s": None, "p99_s": None, "max_s": None}
        return {
            "count": int(array.size),
            "p50_s": float(np.quantile(array, 0.50)),
            "p95_s": float(np.quantile(array, 0.95)),
            "p99_s": float(np.quantile(array, 0.99)),
            "max_s": float(np.max(array)),
        }

    @staticmethod
    def _rate_summary(received_monotonic: Sequence[float]) -> dict[str, float | int | None]:
        values = np.asarray(received_monotonic, dtype=np.float64)
        if values.size < 2:
            return {"count": int(values.size), "hz": None, "interval_p50_s": None, "interval_p95_s": None}
        intervals = np.diff(values)
        intervals = intervals[intervals > 0.0]
        if intervals.size == 0:
            return {"count": int(values.size), "hz": None, "interval_p50_s": None, "interval_p95_s": None}
        return {
            "count": int(values.size),
            "hz": float(1.0 / np.mean(intervals)),
            "interval_p50_s": float(np.quantile(intervals, 0.50)),
            "interval_p95_s": float(np.quantile(intervals, 0.95)),
        }

    def profile_sync(self, duration_s: float) -> dict[str, Any]:
        """Profile the actual frame selector without doing point-cloud math."""

        deadline = time.monotonic() + max(0.0, float(duration_s))
        header_skews: list[float] = []
        receive_skews: list[float] = []
        ages: dict[str, list[float]] = {c.name: [] for c in self.config.cameras if c.enabled}
        failures: Counter[str] = Counter()
        attempts = 0
        last_signature: tuple[tuple[str, int | None], ...] | None = None
        warning_count = 0
        while time.monotonic() < deadline:
            try:
                bundle = self._select_bundle(require_all=self.config.require_all_cameras)
                signature = tuple(
                    (name, frame.sequence)
                    for name, frame in sorted(bundle.frames.items())
                )
                # A profile samples the synchronizer's decisions, not the
                # same cached bundle repeatedly between camera callbacks.
                if signature == last_signature:
                    time.sleep(0.01)
                    continue
                last_signature = signature
                self._check_bundle_tf(bundle)
                attempts += 1
                header_skews.append(bundle.max_header_skew_s)
                receive_skews.append(bundle.max_receive_skew_s)
                if bundle.max_header_skew_s > self.config.camera_warn_header_skew_s:
                    warning_count += 1
                for name, age in bundle.camera_received_ages_s.items():
                    ages.setdefault(name, []).append(age)
            except CameraSyncError as exc:
                attempts += 1
                failures[exc.code] += 1
            time.sleep(0.01)
        return {
            "duration_s": float(duration_s),
            "attempts": attempts,
            "successes": len(header_skews),
            "success_rate": float(len(header_skews) / attempts) if attempts else 0.0,
            "strict_sync_ok": bool(
                attempts
                and header_skews
                and len(header_skews) == attempts
                and max(header_skews) <= self.config.camera_max_header_skew_s
                and max(receive_skews) <= self.config.camera_max_receive_skew_s
            ),
            "header_skew": self._summary(header_skews),
            "receive_skew": self._summary(receive_skews),
            "header_warning_threshold_s": self.config.camera_warn_header_skew_s,
            "header_warning_count": warning_count,
            "camera_received_age": {
                name: self._summary(values) for name, values in ages.items()
            },
            "failures": dict(failures),
        }

    def _check_bundle_tf(self, bundle: SynchronizedDepthBundle) -> None:
        """Check TF availability at every selected image stamp, without fusion."""

        self._bundle_transforms(bundle, require_all=True)

    def preflight(self, timeout_s: float = 5.0, profile_s: float = 0.0) -> dict[str, Any]:
        """Wait for depth/info/TF and optionally profile strict synchronization."""

        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            with self._lock:
                ready_count = sum(
                    1
                    for name, sample in self._samples.items()
                    if sample.depth_m is not None and sample.intrinsics is not None
                )
            enabled_count = sum(1 for c in self.config.cameras if c.enabled)
            if (ready_count >= enabled_count if self.config.require_all_cameras else ready_count > 0):
                break
            time.sleep(0.1)

        now_wall = time.time()
        published_types: dict[str, str] = {}
        if self._ros is not None:
            try:
                published_types = dict(self._ros.get_published_topics())
            except Exception:  # noqa: BLE001
                published_types = {}
        with self._lock:
            report_cameras: dict[str, dict[str, Any]] = {}
            for cam in self.config.cameras:
                if not cam.enabled:
                    continue
                sample = self._samples.get(cam.name, _CamSample(frame_id=cam.frame_id))
                frame_id = sample.frame_id or cam.frame_id
                header_age = now_wall - sample.stamp if sample.stamp > 0.0 else None
                received_age = now_wall - (sample.received_wall_s or sample.received_at) \
                    if (sample.received_wall_s or sample.received_at) > 0.0 else None
                receive_times = list(self._receive_monotonic.get(cam.name, ()))
                header_wall_age = (
                    abs(now_wall - sample.stamp)
                    if sample.stamp >= 1.0e8
                    else None
                )
                expected_depth_type = (
                    "sensor_msgs/CompressedImage"
                    if cam.depth_msg_type.strip().lower()
                    in {"compressed_depth", "compressed", "compresseddepth"}
                    else "sensor_msgs/Image"
                )
                actual_depth_type = published_types.get(cam.depth_topic)
                depth_type_ok = actual_depth_type in {None, expected_depth_type}
                depth_shape = list(sample.depth_m.shape) if sample.depth_m is not None else None
                depth_bytes = int(sample.depth_m.nbytes) if sample.depth_m is not None else 0
                transform = (
                    self._lookup_T_base_cam(frame_id, sample.stamp)
                    if frame_id and sample.stamp > 0.0
                    else None
                )
                report_cameras[cam.name] = {
                    "has_depth": sample.depth_m is not None,
                    "has_intrinsics": sample.intrinsics is not None,
                    "frame_id": frame_id,
                    "header_stamp_s": sample.stamp or None,
                    "age_s": header_age,
                    "header_wall_age_s": header_wall_age,
                    "received_age_s": received_age,
                    "history_size": len(self._histories.get(cam.name, ())),
                    "receive_rate": self._rate_summary(receive_times),
                    "has_tf": transform is not None,
                    "depth_topic": cam.depth_topic,
                    "depth_msg_type": cam.depth_msg_type,
                    "ros_depth_type": actual_depth_type,
                    "expected_ros_depth_type": expected_depth_type,
                    "depth_type_ok": depth_type_ok,
                    "depth_shape": depth_shape,
                    "buffer_memory_bytes_estimate": depth_bytes * max(
                        len(self._histories.get(cam.name, ())), 1
                    ),
                }
        basic = [
            item["has_depth"]
            and item["has_intrinsics"]
            and item["has_tf"]
            and item["depth_type_ok"]
            for item in report_cameras.values()
        ]
        basic_ok = bool(basic) and (all(basic) if self.config.require_all_cameras else any(basic))
        report: dict[str, Any] = {
            "cameras": report_cameras,
            "ok": basic_ok,
            "basic_ok": basic_ok,
            "strict_sync_ok": None,
            "require_all_cameras": self.config.require_all_cameras,
            "sync_config": {
                "label": (
                    "UNSYNCHRONIZED_LEGACY"
                    if self.config.camera_sync_mode == "latest_legacy"
                    else "BUFFERED_HEADER"
                ),
                "mode": self.config.camera_sync_mode,
                "reference_camera": self.config.camera_reference_camera,
                "buffer_size": self.config.camera_buffer_size,
                "max_header_skew_s": self.config.camera_max_header_skew_s,
                "max_received_age_s": self.config.camera_max_received_age_s,
                "max_receive_skew_s": self.config.camera_max_receive_skew_s,
                "tf_at_image_stamp": self.config.camera_tf_at_image_stamp,
            },
            "invalid_counts": dict(self.invalid_counts),
        }
        if profile_s > 0.0 and basic_ok:
            profile = self.profile_sync(profile_s)
            report["sync_profile"] = profile
            report["strict_sync_ok"] = bool(profile.get("strict_sync_ok"))
            report["ok"] = bool(basic_ok and report["strict_sync_ok"])
        return report


def _quat_pose_to_mat(
    xyz: tuple[float, float, float],
    quat_xyzw: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = quat_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    rotation = np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float32,
    )
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(xyz, dtype=np.float32)
    return transform
