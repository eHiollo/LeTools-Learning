"""Configuration dataclasses with hard validation for Kuavo HIL-SERL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kuavo_rl.contracts import (
    ACCELERATION_LIMIT_ONE_ARM,
    ACTION_DIM,
    DEFAULT_TASK_TEXT,
    IMAGE_KEYS,
    IMAGE_SHAPE_CHW,
    STATE_DIM,
    VELOCITY_LIMIT_ONE_ARM,
)


@dataclass
class SafetyConfig:
    joint_position_low: np.ndarray
    joint_position_high: np.ndarray
    max_delta_rad: np.ndarray
    velocity_limit_one_arm: np.ndarray = field(
        default_factory=lambda: VELOCITY_LIMIT_ONE_ARM.copy()
    )
    acceleration_limit_one_arm: np.ndarray = field(
        default_factory=lambda: ACCELERATION_LIMIT_ONE_ARM.copy()
    )
    claw_low: float = 0.0
    claw_high: float = 1.0
    observation_max_age_s: float = 0.15
    max_cross_topic_skew_s: float = 0.10
    max_consecutive_clips: int = 3
    control_dt_s: float = 0.1  # 10 Hz
    require_deadman_for_teleop: bool = True

    def __post_init__(self) -> None:
        self.joint_position_low = np.asarray(self.joint_position_low, dtype=np.float32)
        self.joint_position_high = np.asarray(self.joint_position_high, dtype=np.float32)
        self.max_delta_rad = np.asarray(self.max_delta_rad, dtype=np.float32)
        self.velocity_limit_one_arm = np.asarray(
            self.velocity_limit_one_arm, dtype=np.float32
        )
        self.acceleration_limit_one_arm = np.asarray(
            self.acceleration_limit_one_arm, dtype=np.float32
        )
        if self.joint_position_low.shape != (ACTION_DIM,):
            raise ValueError("joint_position_low must be 16-D")
        if self.joint_position_high.shape != (ACTION_DIM,):
            raise ValueError("joint_position_high must be 16-D")
        if self.max_delta_rad.shape != (ACTION_DIM,):
            raise ValueError("max_delta_rad must be 16-D")
        if np.any(self.joint_position_low >= self.joint_position_high):
            raise ValueError("joint_position_low must be < joint_position_high")
        if self.control_dt_s <= 0:
            raise ValueError("control_dt_s must be > 0")


@dataclass
class EpisodeConfig:
    max_steps: int = 75
    max_duration_s: float = 15.0
    success_reward: float = 1.0
    failure_reward: float = 0.0
    safety_penalty: float = -1.0
    pause_timeout_s: float = 5.0


@dataclass
class RewardConfig:
    task_text: str = DEFAULT_TASK_TEXT
    use_robometer: bool = True
    robometer_mode: str = "episode_end"  # episode_end | offline | disabled
    robometer_model_id: str = "lerobot/Robometer-4B"
    async_timeout_s: float = 30.0
    allow_dense_progress: bool = False
    dense_progress_hz: float = 1.0


@dataclass
class ActRunnerConfig:
    chunk_size: int = 10
    execute_steps: int = 1
    fps: int = 10

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if self.execute_steps != 1:
            raise ValueError(
                "HIL-SERL stage-A contract requires execute_steps=1 "
                "(predict chunk, execute first step only)"
            )


@dataclass
class EnvConfig:
    fps: int = 10
    task: str = "box_to_chest_mvp"
    shadow_mode: bool = False  # predict only, never publish
    image_keys: tuple[str, ...] = IMAGE_KEYS
    image_shape_chw: tuple[int, int, int] = IMAGE_SHAPE_CHW
    state_dim: int = STATE_DIM
    action_dim: int = ACTION_DIM
    arm_publish_unit: str = "rad_to_sdk"  # rad_to_sdk | rad_to_deg_topic
    claw_command_scale: float = 100.0
    safety: SafetyConfig | None = None
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.action_dim != ACTION_DIM or self.state_dim != STATE_DIM:
            raise ValueError("MVP locks action/state to 16-D")
        if self.fps not in (5, 10):
            raise ValueError("fps must be 5 or 10 for MVP")
        if self.safety is None:
            self.safety = default_safety_config(control_dt_s=1.0 / self.fps)


def default_safety_config(control_dt_s: float = 0.1) -> SafetyConfig:
    """Conservative defaults; freeze after field preflight."""
    joint_low = np.full(ACTION_DIM, -3.14, dtype=np.float32)
    joint_high = np.full(ACTION_DIM, 3.14, dtype=np.float32)
    joint_low[7] = joint_low[15] = 0.0
    joint_high[7] = joint_high[15] = 1.0
    # Per-step delta from 0.8x velocity * dt
    max_delta = np.zeros(ACTION_DIM, dtype=np.float32)
    max_delta[0:7] = VELOCITY_LIMIT_ONE_ARM * control_dt_s
    max_delta[8:15] = VELOCITY_LIMIT_ONE_ARM * control_dt_s
    max_delta[7] = max_delta[15] = 1.0  # claw can change fully per step
    return SafetyConfig(
        joint_position_low=joint_low,
        joint_position_high=joint_high,
        max_delta_rad=max_delta,
        control_dt_s=control_dt_s,
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be mapping: {path}")
    return data


def build_env_config_from_dict(raw: dict[str, Any]) -> EnvConfig:
    env_raw = dict(raw.get("env", raw))
    safety_raw = env_raw.pop("safety", None)
    episode_raw = env_raw.pop("episode", {})
    reward_raw = env_raw.pop("reward", {})
    safety = None
    if safety_raw:
        safety = SafetyConfig(
            joint_position_low=np.asarray(safety_raw["joint_position_low"], dtype=np.float32),
            joint_position_high=np.asarray(safety_raw["joint_position_high"], dtype=np.float32),
            max_delta_rad=np.asarray(safety_raw["max_delta_rad"], dtype=np.float32),
            observation_max_age_s=float(safety_raw.get("observation_max_age_s", 0.15)),
            max_consecutive_clips=int(safety_raw.get("max_consecutive_clips", 3)),
            control_dt_s=float(safety_raw.get("control_dt_s", 0.1)),
        )
    return EnvConfig(
        fps=int(env_raw.get("fps", 10)),
        task=str(env_raw.get("task", "box_to_chest_mvp")),
        shadow_mode=bool(env_raw.get("shadow_mode", False)),
        arm_publish_unit=str(env_raw.get("arm_publish_unit", "rad_to_sdk")),
        image_shape_chw=tuple(env_raw.get("image_shape_chw", IMAGE_SHAPE_CHW)),  # type: ignore[arg-type]
        safety=safety,
        episode=EpisodeConfig(**{k: episode_raw[k] for k in episode_raw if k in EpisodeConfig.__dataclass_fields__}),
        reward=RewardConfig(**{k: reward_raw[k] for k in reward_raw if k in RewardConfig.__dataclass_fields__}),
    )
