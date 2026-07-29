"""Config loading for RL-100 zarr collection (isolated from HIL topic profiles)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from kuavo_rl.rl100_zarr.schema import ACTION_DIM, NUM_POINTS, STATE_DIM

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
    task: str = "box_to_chest_v1"
    output_root: str = "data/rl100"
    zarr_name: str = "demo.zarr"
    overwrite: bool = False
    only_success: bool = False
    fps: float = 10.0
    num_points: int = NUM_POINTS
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    lambda_penalty: float = 0.05
    smooth_penalty: float = 0.01
    max_episode_len: int = 2000
    base_frame: str = "base_link"
    workspace: dict[str, list[float]] = field(default_factory=dict)
    cameras: list[CameraPCConfig] = field(default_factory=default_cameras)
    joystick_topic: str = "/quest_joystick_data"
    arm_traj_topic: str = "/kuavo_arm_traj"
    sensors_topic: str = "/sensors_data_raw"
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

    def output_dir(self) -> Path:
        return Path(self.output_root) / self.task

    def zarr_path(self) -> Path:
        return self.output_dir() / self.zarr_name

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
        return cls(
            task=str(raw.get("task", "box_to_chest_v1")),
            output_root=str(raw.get("output_root", "data/rl100")),
            zarr_name=str(raw.get("zarr_name", "demo.zarr")),
            overwrite=bool(raw.get("overwrite", False)),
            only_success=bool(raw.get("only_success", False)),
            fps=float(raw.get("fps", 10.0)),
            num_points=int(raw.get("num_points", NUM_POINTS)),
            state_dim=int(raw.get("state_dim", STATE_DIM)),
            action_dim=int(raw.get("action_dim", ACTION_DIM)),
            lambda_penalty=float(raw.get("lambda_penalty", 0.05)),
            smooth_penalty=float(raw.get("smooth_penalty", 0.01)),
            max_episode_len=int(raw.get("max_episode_len", 2000)),
            base_frame=str(raw.get("base_frame", "base_link")),
            workspace=dict(raw.get("workspace") or {}),
            cameras=cameras,
            joystick_topic=str(raw.get("joystick_topic", "/quest_joystick_data")),
            arm_traj_topic=str(raw.get("arm_traj_topic", "/kuavo_arm_traj")),
            sensors_topic=str(raw.get("sensors_topic", "/sensors_data_raw")),
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
        )


def load_rl100_collect_config(path: str | Path | None = None) -> RL100CollectConfig:
    if path is None:
        return RL100CollectConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return RL100CollectConfig.from_dict(raw)
