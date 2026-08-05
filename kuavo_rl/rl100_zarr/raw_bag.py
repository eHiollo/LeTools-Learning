"""Brain-style RL-100 raw rosbag recording and offline conversion.

The live side only records ROS messages.  State/action alignment and depth
point-cloud generation intentionally happen after the bag is closed.
"""

from __future__ import annotations

import bisect
import json
import os
import signal
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from kuavo_rl.contracts import (
    RL100_DEXHAND_JOINT_NAMES,
    compose_rl100_topic_action,
    compose_rl100_topic_state,
)
from kuavo_rl.rl100_zarr.config import RL100CollectConfig


def raw_bag_topics(config: RL100CollectConfig) -> list[str]:
    """Return the raw topics needed to rebuild one topic-native episode."""
    topics = [
        config.sensors_topic,
        config.dexhand_state_topic,
        config.arm_traj_topic,
        config.hand_command_topic,
        "/tf",
        "/tf_static",
    ]
    for camera in config.cameras:
        if camera.enabled:
            topics.extend((camera.depth_topic, camera.camera_info_topic))
    return list(dict.fromkeys(str(topic) for topic in topics if topic))


@dataclass
class RawBagHandle:
    episode_id: str
    bag_path: Path
    metadata_path: Path
    command: list[str]
    process: subprocess.Popen
    stdout_file: Any
    stderr_file: Any


class RawRosbagRecorder:
    """Start/stop one isolated rosbag process per demonstration episode."""

    def __init__(self, config: RL100CollectConfig):
        self.config = config
        self.root = config.raw_bag_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def start(self, episode_id: str) -> RawBagHandle:
        base = self.root / str(episode_id)
        bag_path = base.with_suffix(".bag")
        metadata_path = base.with_suffix(".json")
        stdout_path = base.with_suffix(".record.stdout.log")
        stderr_path = base.with_suffix(".record.stderr.log")
        command = ["rosbag", "record"]
        if self.config.raw_bag_lz4:
            command.append("--lz4")
        command.extend(["-O", str(base), *raw_bag_topics(self.config)])
        if shutil.which(command[0]) is None:
            raise RuntimeError("rosbag command not found; source /opt/ros/noetic/setup.bash first")
        stdout_file = stdout_path.open("w", encoding="utf-8")
        stderr_file = stderr_path.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            # Let rosbag subscribe before the operator moves.  This is outside
            # the sampling path and is paid once per episode.
            time.sleep(0.15)
            if process.poll() is not None:
                raise RuntimeError(
                    f"rosbag record exited immediately with code {process.returncode}; "
                    f"see {stderr_path}"
                )
        except Exception:
            stdout_file.close()
            stderr_file.close()
            raise
        return RawBagHandle(
            episode_id=str(episode_id),
            bag_path=bag_path,
            metadata_path=metadata_path,
            command=command,
            process=process,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )

    def stop(self, handle: RawBagHandle) -> int:
        process = handle.process
        if process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=float(self.config.raw_bag_stop_timeout_s))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=1.0)
        handle.stdout_file.close()
        handle.stderr_file.close()
        return int(process.returncode if process.returncode is not None else -9)

    @staticmethod
    def write_metadata(handle: RawBagHandle, payload: dict[str, Any]) -> Path:
        data = dict(payload)
        data.update(
            {
                "episode_id": handle.episode_id,
                "bag_path": str(handle.bag_path),
                "command": handle.command,
            }
        )
        handle.metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return handle.metadata_path


@dataclass(frozen=True)
class _TimedValue:
    timestamp_s: float
    header_stamp_s: float
    received_s: float
    value: np.ndarray
    extra_s: float = 0.0


@dataclass(frozen=True)
class _ReferenceFrame:
    timestamp_s: float
    header_stamp_s: float
    received_s: float


@dataclass(frozen=True)
class _CameraCloud:
    points: np.ndarray
    timestamp_s: float
    header_stamp_s: float
    received_s: float


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


def _bag_time_s(value: Any) -> float:
    try:
        result = float(value.to_sec())
    except Exception:  # noqa: BLE001
        try:
            result = float(value)
        except Exception:  # noqa: BLE001
            result = 0.0
    return result if np.isfinite(result) else 0.0


def _message_time(msg: Any, bag_time_s: float) -> tuple[float, float]:
    header = _stamp_s(msg)
    return (header if header > 0.0 else bag_time_s), header


def _positions(value: Any) -> np.ndarray:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.uint8).astype(np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _valid_value(value: Any, dim: int) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (dim,) or not np.isfinite(array).all():
        return None
    return array.copy()


def _parse_joint(msg: Any) -> tuple[np.ndarray, float] | None:
    joint_data = getattr(msg, "joint_data", None)
    value = _valid_value(getattr(joint_data, "joint_q", ()), 20)
    if value is None:
        return None
    sensor_time = getattr(msg, "sensor_time", 0.0)
    try:
        sensor_time_s = float(sensor_time.to_sec())
    except Exception:  # noqa: BLE001
        try:
            sensor_time_s = float(sensor_time)
        except Exception:  # noqa: BLE001
            sensor_time_s = 0.0
    return value, sensor_time_s if np.isfinite(sensor_time_s) else 0.0


def _parse_hand_state(msg: Any) -> tuple[np.ndarray, float] | None:
    names = tuple(str(x) for x in (getattr(msg, "name", ()) or ()))
    if names != RL100_DEXHAND_JOINT_NAMES:
        return None
    value = _valid_value(getattr(msg, "position", ()), 12)
    if value is None or np.any(value < 0.0) or np.any(value > 100.0):
        return None
    return value, 0.0


def _parse_arm_command(msg: Any) -> tuple[np.ndarray, float] | None:
    value = _valid_value(getattr(msg, "position", ()), 14)
    return (value, 0.0) if value is not None else None


def _parse_hand_command(msg: Any) -> tuple[np.ndarray, float] | None:
    left = _positions(getattr(msg, "left_hand_position", ()))
    right = _positions(getattr(msg, "right_hand_position", ()))
    if left.shape != (6,) or right.shape != (6,):
        return None
    value = np.concatenate([left, right]).astype(np.float32)
    if not np.isfinite(value).all() or np.any(value < 0.0) or np.any(value > 100.0):
        return None
    return value, 0.0


def _read_events(
    bag: Any,
    topic: str,
    parser: Callable[[Any], tuple[np.ndarray, float] | None],
) -> list[_TimedValue]:
    events: list[_TimedValue] = []
    for _, msg, bag_time in bag.read_messages(topics=[topic]):
        received = _bag_time_s(bag_time)
        timestamp, header_stamp = _message_time(msg, received)
        parsed = parser(msg)
        if parsed is None:
            continue
        value, extra_s = parsed
        events.append(_TimedValue(timestamp, header_stamp, received, value, extra_s))
    # rosbag receive order is the causal order used for state/command hold.
    events.sort(key=lambda item: item.received_s)
    return events


def _latest_before(
    events: Sequence[_TimedValue],
    timestamp_s: float,
    *,
    received: bool = False,
    index_values: Sequence[float] | None = None,
) -> _TimedValue | None:
    if not events:
        return None
    stamps = index_values or [
        event.received_s if received else event.timestamp_s for event in events
    ]
    index = bisect.bisect_right(stamps, float(timestamp_s)) - 1
    return events[index] if index >= 0 else None


def _reference_frames(bag: Any, topic: str, fps: float) -> list[_ReferenceFrame]:
    frames: list[_ReferenceFrame] = []
    min_interval = 1.0 / max(float(fps), 1.0)
    for _, msg, bag_time in bag.read_messages(topics=[topic]):
        received = _bag_time_s(bag_time)
        timestamp, header_stamp = _message_time(msg, received)
        if not np.isfinite(timestamp):
            continue
        if frames and timestamp <= frames[-1].timestamp_s:
            continue
        if not frames or timestamp - frames[-1].timestamp_s >= min_interval - 1e-6:
            frames.append(_ReferenceFrame(timestamp, header_stamp, received))
    return frames


def decimate_reference_timestamps(timestamps: Sequence[float], fps: float) -> np.ndarray:
    """Time-based reference-camera decimation, capped at the configured FPS."""
    values = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return values
    kept: list[float] = []
    interval = 1.0 / max(float(fps), 1.0)
    for value in values:
        if not np.isfinite(value):
            continue
        if not kept or (
            value > kept[-1] and value - kept[-1] >= interval - 1e-6
        ):
            kept.append(float(value))
    return np.asarray(kept, dtype=np.float64)


def _camera_info(bag: Any, topic: str) -> tuple[tuple[float, float, float, float], str] | None:
    for _, msg, _ in bag.read_messages(topics=[topic]):
        K = getattr(msg, "K", None)
        if K is None or len(K) < 6:
            continue
        intrinsics = (float(K[0]), float(K[4]), float(K[2]), float(K[5]))
        if not np.isfinite(intrinsics).all() or intrinsics[0] <= 0.0 or intrinsics[1] <= 0.0:
            continue
        frame_id = str(getattr(getattr(msg, "header", None), "frame_id", "") or "")
        return intrinsics, frame_id
    return None


def _tf_matrix(transform: Any) -> np.ndarray:
    translation = transform.transform.translation
    rotation = transform.transform.rotation
    x, y, z, w = (float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w))
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm <= 1e-12:
        raise RuntimeError("recorded TF has zero quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), float(translation.x)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), float(translation.y)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), float(translation.z)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _load_tf_buffer(bag: Any) -> Any:
    try:
        import rospy
        import tf2_ros
    except Exception as exc:  # pragma: no cover - robot-side dependency
        raise RuntimeError("offline rosbag conversion requires rospy and tf2_ros") from exc
    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3600.0))
    count = 0
    for topic in ("/tf_static", "/tf"):
        for _, msg, _ in bag.read_messages(topics=[topic]):
            for transform in getattr(msg, "transforms", ()):
                try:
                    if topic == "/tf_static" and hasattr(buffer, "set_transform_static"):
                        buffer.set_transform_static(transform, "rl100_raw_bag")
                    else:
                        buffer.set_transform(transform, "rl100_raw_bag")
                    count += 1
                except Exception:
                    continue
    if count == 0:
        raise RuntimeError("raw bag contains no /tf or /tf_static transforms")
    return buffer


def _lookup_pose(buffer: Any, base_frame: str, camera_frame: str, timestamp_s: float) -> np.ndarray:
    import rospy

    try:
        transform = buffer.lookup_transform(
            base_frame,
            camera_frame,
            rospy.Time.from_sec(float(timestamp_s)),
            rospy.Duration(0.0),
        )
    except Exception:
        # Camera extrinsics are normally static.  Falling back to the latest
        # recorded transform keeps bags usable when /tf starts after a sensor.
        transform = buffer.lookup_transform(
            base_frame,
            camera_frame,
            rospy.Time(0),
            rospy.Duration(0.0),
        )
    return _tf_matrix(transform)


def _nearest_messages(bag: Any, topic: str, targets: Sequence[float]):
    iterator = iter(bag.read_messages(topics=[topic]))
    previous: tuple[Any, float, float, float] | None = None
    current: tuple[Any, float, float, float] | None = None

    def _next() -> tuple[Any, float, float, float] | None:
        try:
            _, msg, bag_time = next(iterator)
        except StopIteration:
            return None
        received = _bag_time_s(bag_time)
        timestamp, header_stamp = _message_time(msg, received)
        return msg, timestamp, header_stamp, received

    current = _next()
    for target in targets:
        while current is not None and current[1] < float(target):
            previous = current
            current = _next()
        candidates = [item for item in (previous, current) if item is not None]
        if not candidates:
            yield None
            continue
        yield min(candidates, key=lambda item: abs(item[1] - float(target)))


def _decode_depth(msg: Any, camera: Any, bridge: Any) -> np.ndarray | None:
    from kuavo_rl.rl100_zarr.ros_depth import _decode_compressed_depth, _depth_m_from_image_msg

    kind = str(camera.depth_msg_type or "image").strip().lower()
    if kind in {"compressed_depth", "compressed", "compresseddepth"}:
        return _decode_compressed_depth(msg)
    return _depth_m_from_image_msg(bridge, msg)


def _cap_points(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    index = np.linspace(0, points.shape[0] - 1, max_points, dtype=np.int64)
    return points[index]


def _camera_clouds(
    bag: Any,
    config: RL100CollectConfig,
    camera: Any,
    targets: Sequence[float],
    tf_buffer: Any,
) -> list[_CameraCloud | None]:
    from cv_bridge import CvBridge
    from kuavo_rl.rl100_zarr.pointcloud import crop_workspace, depth_to_point_cloud

    info = _camera_info(bag, camera.camera_info_topic)
    if info is None:
        raise RuntimeError(f"missing camera info in raw bag: {camera.camera_info_topic}")
    intrinsics, info_frame = info
    bridge = CvBridge()
    output: list[_CameraCloud | None] = []
    max_camera_points = max(int(config.num_points) * 4, 4096)
    x_range, y_range, z_range = config.workspace_ranges()
    for item in _nearest_messages(bag, camera.depth_topic, targets):
        if item is None:
            output.append(None)
            continue
        msg, timestamp, header_stamp, received = item
        depth_m = _decode_depth(msg, camera, bridge)
        if depth_m is None:
            output.append(None)
            continue
        frame_id = str(getattr(getattr(msg, "header", None), "frame_id", "") or "")
        frame_id = frame_id or info_frame or str(camera.frame_id)
        if not frame_id:
            output.append(None)
            continue
        try:
            pose = _lookup_pose(tf_buffer, config.base_frame, frame_id, timestamp)
            cloud = depth_to_point_cloud(depth_m, intrinsics, pose)
            cloud = crop_workspace(
                cloud,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
            )
            cloud = _cap_points(cloud, max_camera_points)
        except Exception:
            output.append(None)
            continue
        output.append(_CameraCloud(cloud, timestamp, header_stamp, received))
    return output


def _audit_template() -> dict[str, list[Any]]:
    return {
        "sample_stamp": [],
        "sample_cutoff_received_at": [],
        "joint_state_stamp": [],
        "dexhand_state_stamp": [],
        "arm_command_stamp": [],
        "hand_command_stamp": [],
        "joint_state_received_at": [],
        "dexhand_state_received_at": [],
        "arm_command_received_at": [],
        "hand_command_received_at": [],
        "joint_sensor_time": [],
        "joint_hand_skew": [],
        "joint_age": [],
        "hand_age": [],
        "point_cloud_stamp": [],
        "point_cloud_received_at": [],
        "point_cloud_age": [],
        "point_cloud_camera_skew": [],
        "point_cloud_header_skew": [],
        "point_cloud_receive_skew": [],
        "point_cloud_reference_stamp": [],
        "point_cloud_reference_camera": [],
        "head_depth_stamp": [],
        "left_depth_stamp": [],
        "right_depth_stamp": [],
        "head_depth_received_at": [],
        "left_depth_received_at": [],
        "right_depth_received_at": [],
        "head_depth_age": [],
        "left_depth_age": [],
        "right_depth_age": [],
        "point_cloud_valid_points": [],
        "arm_command_changed": [],
        "hand_command_changed": [],
        "arm_command_seen": [],
        "hand_command_seen": [],
        "arm_command_source": [],
        "hand_command_source": [],
    }


def convert_raw_bag_to_npz(
    bag_path: str | Path,
    config: RL100CollectConfig,
    *,
    episode_dir: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Path | None:
    """Convert one labeled raw bag to topic-native episode NPZ."""
    bag_file = Path(bag_path)
    meta_file = Path(metadata_path) if metadata_path else bag_file.with_suffix(".json")
    if not meta_file.is_file():
        raise RuntimeError(f"missing raw bag metadata: {meta_file}")
    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
    if metadata.get("status") != "saved" or metadata.get("result_type") not in {"success", "failure"}:
        return None

    try:
        import rosbag
    except Exception as exc:  # pragma: no cover - robot-side dependency
        raise RuntimeError("raw rosbag conversion requires Python rosbag") from exc

    from kuavo_rl.rl100_zarr.pointcloud import build_rl100_point_cloud
    from kuavo_rl.rl100_zarr.staging import save_episode_npz

    with rosbag.Bag(str(bag_file), "r") as bag:
        references = _reference_frames(
            bag,
            next(c.depth_topic for c in config.cameras if c.name == config.camera_reference_camera),
            config.fps,
        )
        joints = _read_events(bag, config.sensors_topic, _parse_joint)
        hand_states = _read_events(bag, config.dexhand_state_topic, _parse_hand_state)
        arm_commands = _read_events(bag, config.arm_traj_topic, _parse_arm_command)
        hand_commands = _read_events(bag, config.hand_command_topic, _parse_hand_command)
        if not references or not joints or not hand_states:
            return None
        first_commands = [event.received_s for event in (*arm_commands, *hand_commands)]
        if not first_commands:
            return None
        first_command_received_s = min(first_commands)
        references = [
            frame for frame in references if frame.received_s >= first_command_received_s
        ]
        if not references:
            return None

        tf_buffer = _load_tf_buffer(bag)
        camera_clouds: dict[str, list[_CameraCloud | None]] = {}
        for camera in config.cameras:
            if camera.enabled:
                camera_clouds[camera.name] = _camera_clouds(
                    bag,
                    config,
                    camera,
                    [frame.timestamp_s for frame in references],
                    tf_buffer,
                )

        joint_received = [item.received_s for item in joints]
        hand_state_received = [item.received_s for item in hand_states]
        arm_received = [item.received_s for item in arm_commands]
        hand_received = [item.received_s for item in hand_commands]
        initial_joint = (
            _latest_before(
                joints,
                first_command_received_s,
                received=True,
                index_values=joint_received,
            )
            or joints[0]
        )
        initial_arm_deg = np.rad2deg(initial_joint.value[4:18]).astype(np.float32)
        default_hand = np.asarray(config.hand_default, dtype=np.float32)
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        point_clouds: list[np.ndarray] = []
        audit = _audit_template()
        previous_arm_time = float("-inf")
        previous_hand_time = float("-inf")
        camera_names = [camera.name for camera in config.cameras if camera.enabled]
        camera_audit_names = {
            "head_cam_h": "head",
            "wrist_cam_l": "left",
            "wrist_cam_r": "right",
        }
        x_range, y_range, z_range = config.workspace_ranges()

        for index, reference in enumerate(references):
            selected = [camera_clouds[name][index] for name in camera_names]
            if any(item is None for item in selected):
                continue
            clouds = [item.points for item in selected if item is not None]
            try:
                points = build_rl100_point_cloud(
                    clouds,
                    num_points=config.num_points,
                    x_range=x_range,
                    y_range=y_range,
                    z_range=z_range,
                    raise_on_empty=config.fail_on_empty_pointcloud,
                    min_points=config.min_workspace_points,
                )
            except Exception:
                continue

            # Bag receive time is the causal clock. Header time drives camera
            # nearest-frame selection and remains an audit field only here.
            joint = _latest_before(
                joints,
                reference.received_s,
                received=True,
                index_values=joint_received,
            )
            hand_state = _latest_before(
                hand_states,
                reference.received_s,
                received=True,
                index_values=hand_state_received,
            )
            if joint is None or hand_state is None:
                continue
            arm = _latest_before(
                arm_commands,
                reference.received_s,
                received=True,
                index_values=arm_received,
            )
            hand = _latest_before(
                hand_commands,
                reference.received_s,
                received=True,
                index_values=hand_received,
            )
            arm_value = initial_arm_deg if arm is None else arm.value
            hand_value = default_hand if hand is None else hand.value
            state = compose_rl100_topic_state(joint.value, hand_state.value)
            action = compose_rl100_topic_action(arm_value, hand_value)
            states.append(state)
            actions.append(action)
            point_clouds.append(points)

            selected_times = [item.timestamp_s for item in selected if item is not None]
            selected_headers = [item.header_stamp_s for item in selected if item is not None]
            selected_received = [item.received_s for item in selected if item is not None]
            header_reference = reference.header_stamp_s or reference.timestamp_s
            header_skew = max(abs(value - header_reference) for value in selected_times)
            if any(value > 0.0 for value in selected_headers):
                header_skew = max(
                    abs((value or item.timestamp_s) - header_reference)
                    for value, item in zip(selected_headers, selected)
                    if item is not None
                )
            receive_skew = max(abs(value - reference.received_s) for value in selected_received)
            cloud_age = max(max(reference.timestamp_s - value, 0.0) for value in selected_times)
            joint_age = max(reference.received_s - joint.received_s, 0.0)
            hand_age = max(reference.received_s - hand_state.received_s, 0.0)
            command_arm_time = arm.received_s if arm is not None else first_command_received_s
            command_hand_time = hand.received_s if hand is not None else first_command_received_s
            arm_changed = arm is not None and arm.received_s > previous_arm_time
            hand_changed = hand is not None and hand.received_s > previous_hand_time
            if arm is not None:
                previous_arm_time = arm.received_s
            if hand is not None:
                previous_hand_time = hand.received_s

            audit["sample_stamp"].append(reference.timestamp_s)
            audit["sample_cutoff_received_at"].append(reference.received_s)
            audit["joint_state_stamp"].append(joint.header_stamp_s)
            audit["dexhand_state_stamp"].append(hand_state.header_stamp_s)
            audit["arm_command_stamp"].append(arm.header_stamp_s if arm is not None else 0.0)
            audit["hand_command_stamp"].append(hand.header_stamp_s if hand is not None else 0.0)
            audit["joint_state_received_at"].append(joint.received_s)
            audit["dexhand_state_received_at"].append(hand_state.received_s)
            audit["arm_command_received_at"].append(command_arm_time)
            audit["hand_command_received_at"].append(command_hand_time)
            audit["joint_sensor_time"].append(joint.extra_s)
            audit["joint_hand_skew"].append(abs(joint.timestamp_s - hand_state.timestamp_s))
            audit["joint_age"].append(joint_age)
            audit["hand_age"].append(hand_age)
            audit["point_cloud_stamp"].append(reference.timestamp_s)
            audit["point_cloud_received_at"].append(reference.received_s)
            audit["point_cloud_age"].append(cloud_age)
            audit["point_cloud_camera_skew"].append(header_skew)
            audit["point_cloud_header_skew"].append(header_skew)
            audit["point_cloud_receive_skew"].append(receive_skew)
            audit["point_cloud_reference_stamp"].append(reference.timestamp_s)
            audit["point_cloud_reference_camera"].append(config.camera_reference_camera)
            for item, camera in zip(selected, camera_names):
                short_name = camera_audit_names.get(camera)
                if short_name is None:
                    continue
                assert item is not None
                audit[f"{short_name}_depth_stamp"].append(item.header_stamp_s or item.timestamp_s)
                audit[f"{short_name}_depth_received_at"].append(item.received_s)
                audit[f"{short_name}_depth_age"].append(max(reference.timestamp_s - item.timestamp_s, 0.0))
            audit["point_cloud_valid_points"].append(int(points.shape[0]))
            audit["arm_command_changed"].append(bool(arm_changed))
            audit["hand_command_changed"].append(bool(hand_changed))
            audit["arm_command_seen"].append(arm is not None)
            audit["hand_command_seen"].append(hand is not None)
            audit["arm_command_source"].append("topic" if arm is not None else "measured_hold")
            audit["hand_command_source"].append("topic" if hand is not None else "default_hold")

    if not states:
        return None
    output_dir = Path(episode_dir) if episode_dir else config.staging_episode_dir()
    output = output_dir / f"{metadata['episode_id']}_{metadata['result_type']}.npz"
    meta = {
        "task": config.task,
        "contract": config.contract,
        "state_dim": config.state_dim,
        "action_dim": config.action_dim,
        "result_type": metadata["result_type"],
        "stop_reason": metadata.get("stop_reason", "unknown"),
        "action_source": "raw_rosbag_topic_alignment",
        "raw_bag": str(bag_file),
        "raw_bag_metadata": str(meta_file),
        "alignment": {
            "reference_camera": config.camera_reference_camera,
            "target_fps": config.fps,
            "command_rule": "start_on_any_command + hold_last + default_hand_before_first_hand_command",
            "point_cloud_generation": "offline",
        },
    }
    return save_episode_npz(
        output,
        states=states,
        actions=actions,
        point_clouds=point_clouds,
        result_type=str(metadata["result_type"]),
        meta=meta,
        audit={key: np.asarray(value) for key, value in audit.items()},
    )


def build_zarr_from_raw_bags(
    config: RL100CollectConfig,
    *,
    raw_bag_dir: str | Path | None = None,
    episode_dir: str | Path | None = None,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert saved raw bags, then reuse the existing NPZ→Zarr writer."""
    from kuavo_rl.rl100_zarr.staging import build_zarr_from_episode_dir

    bag_root = Path(raw_bag_dir) if raw_bag_dir else config.raw_bag_dir()
    output_dir = Path(episode_dir) if episode_dir else config.output_dir() / "episodes"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    for bag_path in sorted(bag_root.glob("*.bag")):
        try:
            output = convert_raw_bag_to_npz(bag_path, config, episode_dir=output_dir)
        except Exception as exc:  # keep other episodes convertible
            skipped += 1
            print(f"[rl100-raw] skip {bag_path.name}: {exc}", flush=True)
            continue
        if output is None:
            skipped += 1
        else:
            converted += 1
    report = build_zarr_from_episode_dir(
        output_dir,
        config.zarr_path(),
        only_success=config.only_success,
        overwrite=config.overwrite,
        lambda_penalty=config.lambda_penalty,
        smooth_penalty=config.smooth_penalty,
        max_episode_len=config.max_episode_len,
        attrs=attrs,
    )
    report.update({"raw_bags": str(bag_root), "raw_bags_converted": converted, "raw_bags_skipped": skipped})
    return report
