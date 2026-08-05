"""Config loading for RL-100 zarr collection (isolated from HIL topic profiles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

# Same topics as kuavo_deploy obs_buffer / 模仿学习真机深度。
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


def default_cameras() -> list[CameraPCConfig]:
    """Three-cam depth topics aligned with Kuavo real / kuavo_deploy."""
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

    def validate_contract(self) -> None:
        if self.contract != RL100_TOPIC_NATIVE_CONTRACT:
            raise ValueError(f"RL-100 collection requires contract={RL100_TOPIC_NATIVE_CONTRACT}")
        if self.state_dim != STATE_DIM or self.action_dim != ACTION_DIM:
            raise ValueError(f"RL-100 topic-native collection requires state/action {STATE_DIM}/{ACTION_DIM}")
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
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
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
