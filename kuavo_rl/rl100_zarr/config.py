"""Config loading for RL-100 zarr collection (isolated from HIL topic profiles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import yaml

from kuavo_rl.rl100_zarr.schema import ACTION_DIM, NUM_POINTS, STATE_DIM
from kuavo_rl.contracts import (
    RL100_ACTION_DIM,
    RL100_ARM_COMMAND_DIM,
    RL100_DEXHAND_STATE_DIM,
    RL100_HAND_COMMAND_DIM,
    RL100_HAND_DEFAULT,
    RL100_RAW_JOINT_DIM,
    RL100_STATE_DIM,
    RL100_TOPIC_NATIVE_CONTRACT,
)

# Align with HIL real collect: B ends episodes; these are soft safety ceilings only.
LIVE_SAFETY_MAX_STEPS = 100_000
LIVE_SAFETY_MAX_DURATION_S = 86_400.0

# Legacy defaults for callers that do not provide an explicit camera YAML.
# Formal real-robot RL-100 configs select raw depth explicitly below.
_REAL_DEPTH = {
    "head": "/cam_h/depth/image_raw/compressedDepth",
    "wrist_l": "/cam_l/depth/image_rect_raw/compressedDepth",
    "wrist_r": "/cam_r/depth/image_rect_raw/compressedDepth",
}


@dataclass
class CameraPCConfig:
    name: str
    depth_topic: str
    camera_info_topic: str
    frame_id: str = ""
    enabled: bool = True
    # image = sensor_msgs/Image; compressed_depth = sensor_msgs/CompressedImage (PNG)
    depth_msg_type: str = "compressed_depth"
    # Optional RGB compressed topic for replay/annotation (not consumed by RL-100 training).
    color_topic: str = ""
    # Optional RGB calibration topic for offline RGB/depth consumers.
    color_camera_info_topic: str = ""


def default_cameras() -> list[CameraPCConfig]:
    """Legacy three-cam defaults; production YAMLs must select raw/image explicitly."""
    return [
        CameraPCConfig(
            name="head_cam_h",
            depth_topic=_REAL_DEPTH["head"],
            camera_info_topic="/cam_h/depth/camera_info",
            frame_id="cam_h_depth_optical_frame",
            enabled=True,
            depth_msg_type="compressed_depth",
        ),
        CameraPCConfig(
            name="wrist_cam_l",
            depth_topic=_REAL_DEPTH["wrist_l"],
            camera_info_topic="/cam_l/depth/camera_info",
            frame_id="cam_l_depth_optical_frame",
            enabled=True,
            depth_msg_type="compressed_depth",
        ),
        CameraPCConfig(
            name="wrist_cam_r",
            depth_topic=_REAL_DEPTH["wrist_r"],
            camera_info_topic="/cam_r/depth/camera_info",
            frame_id="cam_r_depth_optical_frame",
            enabled=True,
            depth_msg_type="compressed_depth",
        ),
    ]


@dataclass
class RL100CollectConfig:
    contract: str = RL100_TOPIC_NATIVE_CONTRACT
    task: str = "box_to_chest_v1"
    output_root: str = "data/rl100"
    zarr_name: str = "demo.zarr"
    overwrite: bool = False
    only_success: bool = False
    fps: float = 10.0
    num_points: int = NUM_POINTS
    # Brain-style collection: record raw ROS messages first, then align and
    # build point clouds offline. Keep online_pointcloud only as an explicit
    # fallback for old experiments.
    collection_mode: str = "raw_rosbag"
    raw_bag_root: str | None = None
    raw_bag_lz4: bool = True
    raw_bag_stop_timeout_s: float = 15.0
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    arm_command_dim: int = RL100_ARM_COMMAND_DIM
    hand_command_dim: int = RL100_HAND_COMMAND_DIM
    lambda_penalty: float = 0.05
    smooth_penalty: float = 0.0
    max_episode_len: int = 2000
    base_frame: str = "base_link"
    workspace: dict[str, list[float]] = field(default_factory=dict)
    cameras: list[CameraPCConfig] = field(default_factory=default_cameras)
    joystick_topic: str = "/quest_joystick_data"
    arm_traj_topic: str = "/kuavo_arm_traj"
    sensors_topic: str = "/sensors_data_raw"
    dexhand_state_topic: str = "/dexhand/state"
    hand_command_topic: str = "/control_robot_hand_position"
    raw_joint_dim: int = RL100_RAW_JOINT_DIM
    dexhand_state_dim: int = RL100_DEXHAND_STATE_DIM
    hand_default: list[float] = field(default_factory=lambda: RL100_HAND_DEFAULT.tolist())
    hand_default_tolerance: float = 2.0
    start_on_any_command: bool = True
    command_hold_last: bool = True
    command_timeout_s: float | None = None
    joint_state_max_age_s: float = 0.15
    dexhand_state_max_age_s: float = 0.15
    state_max_skew_s: float = 0.10
    depth_max_age_s: float = 0.15
    camera_max_skew_s: float = 0.10
    # Buffered, causal cross-camera synchronization. ``camera_max_skew_s`` is
    # retained as a legacy compatibility field; the new nested values are used
    # when camera_sync_mode is buffered_header.
    camera_sync_mode: str = "buffered_header"
    camera_reference_camera: str = "head_cam_h"
    camera_buffer_size: int = 32
    camera_max_header_skew_s: float = 0.10
    camera_warn_header_skew_s: float = 0.05
    camera_max_received_age_s: float = 0.20
    camera_max_receive_skew_s: float = 0.20
    camera_require_monotonic_header: bool = True
    camera_require_same_time_epoch: bool = True
    camera_tf_at_image_stamp: bool = True
    camera_tf_timeout_s: float = 0.05
    camera_max_consecutive_sync_failures: int = 10
    max_consecutive_source_failures: int = 3
    require_hand_motion: bool = False
    min_hand_action_range: float = 5.0
    confirm_live: bool = False
    shadow_mode: bool = True
    # Real-robot HIL collect wiring (same as hil_collection_real_v001).
    deploy_config: str = "configs/deploy/total/deploy_total.yaml"
    env_config: str = "configs/rl/kuavo_hilserl_real_mvp.yaml"
    require_all_cameras: bool = True
    fail_on_empty_pointcloud: bool = True
    min_workspace_points: int = 32
    # Soft ceilings; episode ends on B (success/fail) like HIL VR collect.
    live_max_steps: int = LIVE_SAFETY_MAX_STEPS
    live_max_duration_s: float = LIVE_SAFETY_MAX_DURATION_S
    # Align with main HIL: quest_y_button + B short/long labels.
    episode_control: str = "quest_y_button"
    chord_long_press_s: float = 0.8
    # "button" (B short=success / long=failure) or "right_stick_ud"
    # (stick down=success / up=failure) for success/failure labeling.
    reward_gesture: str = "button"
    reward_stick_threshold: float = 0.80
    reward_stick_rearm_threshold: float = 0.20
    reward_stick_debounce_s: float = 0.25
    config_path: str | None = None

    def output_dir(self) -> Path:
        return Path(self.output_root) / self.task

    def zarr_path(self) -> Path:
        return self.output_dir() / self.zarr_name

    def raw_bag_dir(self) -> Path:
        return Path(self.raw_bag_root) if self.raw_bag_root else self.output_dir() / "raw_bags"

    def staging_episode_dir(self) -> Path:
        # Keep raw-bag conversions separate from legacy online NPZs.
        name = "raw_episodes" if self.collection_mode == "raw_rosbag" else "episodes"
        return self.output_dir() / name

    def validate_contract(self) -> None:
        if self.contract != RL100_TOPIC_NATIVE_CONTRACT:
            raise ValueError(f"RL-100 collection requires contract={RL100_TOPIC_NATIVE_CONTRACT}")
        if self.state_dim != STATE_DIM or self.action_dim != ACTION_DIM:
            raise ValueError(f"RL-100 topic-native collection requires state/action {STATE_DIM}/{ACTION_DIM}")
        if self.collection_mode not in {"raw_rosbag", "online_pointcloud"}:
            raise ValueError("collection_mode must be raw_rosbag or online_pointcloud")
        if self.raw_bag_stop_timeout_s <= 0.0:
            raise ValueError("raw_bag_stop_timeout_s must be positive")
        if self.state_dim != RL100_STATE_DIM or self.action_dim != RL100_ACTION_DIM:
            raise ValueError("RL-100 topic-native collection requires state/action 32/26")
        if self.arm_command_dim != RL100_ARM_COMMAND_DIM or self.hand_command_dim != RL100_HAND_COMMAND_DIM:
            raise ValueError("RL-100 command dimensions must be arm14/hand12")
        if self.raw_joint_dim != RL100_RAW_JOINT_DIM or self.dexhand_state_dim != RL100_DEXHAND_STATE_DIM:
            raise ValueError("RL-100 raw joint/hand state dimensions must be 20/12")
        if len(self.hand_default) != RL100_HAND_COMMAND_DIM:
            raise ValueError("hand_default must contain 12 values")
        hand_default = np.asarray(self.hand_default, dtype=np.float32)
        if not np.isfinite(hand_default).all():
            raise ValueError("hand_default contains NaN/Inf")
        if np.any(hand_default < 0) or np.any(hand_default > 100):
            raise ValueError("hand_default must be in [0,100]")
        if not np.array_equal(hand_default, RL100_HAND_DEFAULT):
            raise ValueError(
                "hand_default must remain the fixed topic-native pre-hand-command hold "
                f"{RL100_HAND_DEFAULT.tolist()}"
            )
        if not self.start_on_any_command:
            raise ValueError("topic-native collection requires start_on_any_command=true")
        if not self.command_hold_last:
            raise ValueError("topic-native collection requires command_hold_last=true")
        if self.command_timeout_s is not None:
            raise ValueError("topic-native command_timeout_s must remain null (commands hold indefinitely)")
        for name in (
            "hand_default_tolerance",
            "joint_state_max_age_s",
            "dexhand_state_max_age_s",
            "state_max_skew_s",
            "depth_max_age_s",
            "camera_max_skew_s",
            "camera_max_header_skew_s",
            "camera_warn_header_skew_s",
            "camera_max_received_age_s",
            "camera_max_receive_skew_s",
            "camera_tf_timeout_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.camera_sync_mode not in {"buffered_header", "latest_legacy"}:
            raise ValueError("camera_sync_mode must be buffered_header or latest_legacy")
        if self.camera_sync_mode == "buffered_header" and not self.camera_tf_at_image_stamp:
            raise ValueError("buffered_header requires tf_at_image_stamp=true")
        if self.camera_buffer_size < 2:
            raise ValueError("camera_buffer_size must be >= 2")
        if self.camera_warn_header_skew_s > self.camera_max_header_skew_s:
            raise ValueError("camera_warn_header_skew_s must be <= camera_max_header_skew_s")
        if self.camera_reference_camera not in {c.name for c in self.cameras if c.enabled}:
            raise ValueError(
                f"camera_reference_camera={self.camera_reference_camera!r} is not an enabled camera"
            )
        if self.camera_max_consecutive_sync_failures < 1:
            raise ValueError("camera_max_consecutive_sync_failures must be >= 1")
        if int(self.max_consecutive_source_failures) < 1:
            raise ValueError("max_consecutive_source_failures must be >= 1")

    def workspace_ranges(
        self,
    ) -> tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]:
        ws = self.workspace or {
            "x_range": [0.0, 1.2],
            "y_range": [-0.6, 0.6],
            "z_range": [0.0, 1.4],
        }

        def _pair(key: str) -> tuple[float, float] | None:
            v = ws.get(key)
            if not v or len(v) != 2:
                return None
            return (float(v[0]), float(v[1]))

        return _pair("x_range"), _pair("y_range"), _pair("z_range")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RL100CollectConfig":
        camera_sync_raw = raw.get("camera_sync") or {}
        legacy_skew = raw.get("camera_max_skew_s")
        nested_skew = camera_sync_raw.get("max_header_skew_s")
        if legacy_skew is not None and nested_skew is not None:
            try:
                differs = not np.isclose(float(legacy_skew), float(nested_skew))
            except (TypeError, ValueError):
                differs = True
            if differs:
                warnings.warn(
                    "camera_sync.max_header_skew_s overrides legacy camera_max_skew_s; "
                    f"legacy={legacy_skew!r}, active={nested_skew!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        cams_raw = raw.get("cameras")
        if cams_raw is None:
            cameras = default_cameras()
        else:
            cameras = [
                CameraPCConfig(
                    name=str(c["name"]),
                    depth_topic=str(c["depth_topic"]),
                    camera_info_topic=str(c["camera_info_topic"]),
                    frame_id=str(c.get("frame_id", "")),
                    enabled=bool(c.get("enabled", True)),
                    depth_msg_type=str(c.get("depth_msg_type", "compressed_depth")),
                    color_topic=str(c.get("color_topic", "")),
                    color_camera_info_topic=str(c.get("color_camera_info_topic", "")),
                )
                for c in cams_raw
            ]
        config = cls(
            contract=str(raw.get("contract", RL100_TOPIC_NATIVE_CONTRACT)),
            task=str(raw.get("task", "box_to_chest_v1")),
            output_root=str(raw.get("output_root", "data/rl100")),
            zarr_name=str(raw.get("zarr_name", "demo.zarr")),
            overwrite=bool(raw.get("overwrite", False)),
            only_success=bool(raw.get("only_success", False)),
            fps=float(raw.get("fps", 10.0)),
            num_points=int(raw.get("num_points", NUM_POINTS)),
            collection_mode=str(raw.get("collection_mode", "raw_rosbag")),
            raw_bag_root=(
                None if raw.get("raw_bag_root") is None else str(raw.get("raw_bag_root"))
            ),
            raw_bag_lz4=bool(raw.get("raw_bag_lz4", True)),
            raw_bag_stop_timeout_s=float(raw.get("raw_bag_stop_timeout_s", 15.0)),
            state_dim=int(raw.get("state_dim", STATE_DIM)),
            action_dim=int(raw.get("action_dim", ACTION_DIM)),
            arm_command_dim=int(raw.get("arm_command_dim", RL100_ARM_COMMAND_DIM)),
            hand_command_dim=int(raw.get("hand_command_dim", RL100_HAND_COMMAND_DIM)),
            lambda_penalty=float(raw.get("lambda_penalty", 0.05)),
            smooth_penalty=float(raw.get("smooth_penalty", 0.0)),
            max_episode_len=int(raw.get("max_episode_len", 2000)),
            base_frame=str(raw.get("base_frame", "base_link")),
            workspace=dict(raw.get("workspace") or {}),
            cameras=cameras,
            joystick_topic=str(raw.get("joystick_topic", "/quest_joystick_data")),
            arm_traj_topic=str(raw.get("arm_traj_topic", "/kuavo_arm_traj")),
            sensors_topic=str(raw.get("sensors_topic", "/sensors_data_raw")),
            dexhand_state_topic=str(raw.get("dexhand_state_topic", "/dexhand/state")),
            hand_command_topic=str(
                raw.get("hand_command_topic", "/control_robot_hand_position")
            ),
            raw_joint_dim=int(raw.get("raw_joint_dim", RL100_RAW_JOINT_DIM)),
            dexhand_state_dim=int(raw.get("dexhand_state_dim", RL100_DEXHAND_STATE_DIM)),
            hand_default=list(raw.get("hand_default", RL100_HAND_DEFAULT.tolist())),
            hand_default_tolerance=float(raw.get("hand_default_tolerance", 2.0)),
            start_on_any_command=bool(raw.get("start_on_any_command", True)),
            command_hold_last=bool(raw.get("command_hold_last", True)),
            command_timeout_s=(
                None if raw.get("command_timeout_s", None) is None
                else float(raw.get("command_timeout_s"))
            ),
            joint_state_max_age_s=float(raw.get("joint_state_max_age_s", 0.15)),
            dexhand_state_max_age_s=float(raw.get("dexhand_state_max_age_s", 0.15)),
            state_max_skew_s=float(raw.get("state_max_skew_s", 0.10)),
            depth_max_age_s=float(raw.get("depth_max_age_s", 0.15)),
            camera_max_skew_s=float(raw.get("camera_max_skew_s", 0.10)),
            camera_sync_mode=str(
                camera_sync_raw.get(
                    "mode", raw.get("camera_sync_mode", "buffered_header")
                )
            ),
            camera_reference_camera=str(
                camera_sync_raw.get(
                    "reference_camera", raw.get("camera_reference_camera", "head_cam_h")
                )
            ),
            camera_buffer_size=int(
                camera_sync_raw.get(
                    "buffer_size", raw.get("camera_buffer_size", 32)
                )
            ),
            camera_max_header_skew_s=float(
                camera_sync_raw.get(
                    "max_header_skew_s", raw.get("camera_max_header_skew_s", 0.10)
                )
            ),
            camera_warn_header_skew_s=float(
                camera_sync_raw.get(
                    "warn_header_skew_s", raw.get("camera_warn_header_skew_s", 0.05)
                )
            ),
            camera_max_received_age_s=float(
                camera_sync_raw.get(
                    "max_received_age_s", raw.get("camera_max_received_age_s", 0.20)
                )
            ),
            camera_max_receive_skew_s=float(
                camera_sync_raw.get(
                    "max_receive_skew_s", raw.get("camera_max_receive_skew_s", 0.20)
                )
            ),
            camera_require_monotonic_header=bool(
                camera_sync_raw.get(
                    "require_monotonic_header",
                    raw.get("camera_require_monotonic_header", True),
                )
            ),
            camera_require_same_time_epoch=bool(
                camera_sync_raw.get(
                    "require_same_time_epoch",
                    raw.get("camera_require_same_time_epoch", True),
                )
            ),
            camera_tf_at_image_stamp=bool(
                camera_sync_raw.get(
                    "tf_at_image_stamp", raw.get("camera_tf_at_image_stamp", True)
                )
            ),
            camera_tf_timeout_s=float(
                camera_sync_raw.get(
                    "tf_timeout_s", raw.get("camera_tf_timeout_s", 0.05)
                )
            ),
            camera_max_consecutive_sync_failures=int(
                camera_sync_raw.get(
                    "max_consecutive_sync_failures",
                    raw.get("camera_max_consecutive_sync_failures", 10),
                )
            ),
            max_consecutive_source_failures=int(raw.get("max_consecutive_source_failures", 3)),
            require_hand_motion=bool(raw.get("require_hand_motion", raw.get("require_gripper_motion", False))),
            min_hand_action_range=float(raw.get("min_hand_action_range", raw.get("min_gripper_action_range", 5.0))),
            confirm_live=bool(raw.get("confirm_live", False)),
            shadow_mode=bool(raw.get("shadow_mode", True)),
            deploy_config=str(
                raw.get("deploy_config", "configs/deploy/total/deploy_total.yaml")
            ),
            env_config=str(
                raw.get("env_config", "configs/rl/kuavo_hilserl_real_mvp.yaml")
            ),
            require_all_cameras=bool(raw.get("require_all_cameras", True)),
            fail_on_empty_pointcloud=bool(raw.get("fail_on_empty_pointcloud", True)),
            min_workspace_points=int(raw.get("min_workspace_points", 32)),
            live_max_steps=int(raw.get("live_max_steps", LIVE_SAFETY_MAX_STEPS)),
            live_max_duration_s=float(
                raw.get("live_max_duration_s", LIVE_SAFETY_MAX_DURATION_S)
            ),
            episode_control=str(raw.get("episode_control", "quest_y_button")),
            chord_long_press_s=float(raw.get("chord_long_press_s", 0.8)),
            reward_gesture=str(raw.get("reward_gesture", "button")),
            reward_stick_threshold=float(raw.get("reward_stick_threshold", 0.80)),
            reward_stick_rearm_threshold=float(raw.get("reward_stick_rearm_threshold", 0.20)),
            reward_stick_debounce_s=float(raw.get("reward_stick_debounce_s", 0.25)),
        )
        config.validate_contract()
        return config


def load_rl100_collect_config(path: str | Path | None = None) -> RL100CollectConfig:
    if path is None:
        return RL100CollectConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RL100CollectConfig.from_dict(raw)
