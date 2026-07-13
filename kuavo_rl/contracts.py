"""Frozen observation/action contracts for Kuavo 5W v62 HIL-SERL."""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np

ACTION_DIM = 16
STATE_DIM = 16

ACTION_NAMES: tuple[str, ...] = (
    "zarm_l1_link",
    "zarm_l2_link",
    "zarm_l3_link",
    "zarm_l4_link",
    "zarm_l5_link",
    "zarm_l6_link",
    "zarm_l7_link",
    "left_claw",
    "zarm_r1_link",
    "zarm_r2_link",
    "zarm_r3_link",
    "zarm_r4_link",
    "zarm_r5_link",
    "zarm_r6_link",
    "zarm_r7_link",
    "right_claw",
)

# Canonical layout: [L7, left_claw, R7, right_claw]
ARM_LEFT_IDX = slice(0, 7)
CLAW_LEFT_IDX = 7
ARM_RIGHT_IDX = slice(8, 15)
CLAW_RIGHT_IDX = 15
ARM_JOINT_IDX_IN_ACTION = (0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14)
CLAW_IDX_IN_ACTION = (7, 15)

# v62 raw /sensors_data_raw is 28-D; arms at [12:26]
RAW_STATE_DIM_V62 = 28
ARM_SLICE_BY_RAW_DIM = {
    28: slice(12, 26),
    20: slice(4, 18),
    14: slice(0, 14),
}

# 0.8 x TOPP NORMAL, one arm (rad/s, rad/s^2)
VELOCITY_LIMIT_ONE_ARM = np.array(
    [6.64, 2.56, 4.24, 2.56, 4.24, 4.24, 4.24], dtype=np.float32
)
ACCELERATION_LIMIT_ONE_ARM = np.array(
    [20.0, 20.0, 20.0, 20.0, 40.0, 40.0, 40.0], dtype=np.float32
)

IMAGE_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
IMAGE_SHAPE_CHW = (3, 480, 848)

DEFAULT_TASK_TEXT = "将物料框搬运到胸前的目标位置"


class FaultCode(str, Enum):
    NONE = "NONE"
    STOP_SIGNAL = "STOP_SIGNAL"
    ROS_SHUTDOWN = "ROS_SHUTDOWN"
    STALE_OBSERVATION = "STALE_OBSERVATION"
    ACTION_NAN = "ACTION_NAN"
    ACTION_SHAPE = "ACTION_SHAPE"
    ACTION_LIMIT = "ACTION_LIMIT"
    VELOCITY_LIMIT = "VELOCITY_LIMIT"
    SDK_EXCEPTION = "SDK_EXCEPTION"
    RESET_TIMEOUT = "RESET_TIMEOUT"
    HUMAN_ABORT = "HUMAN_ABORT"
    EPISODE_TIMEOUT = "EPISODE_TIMEOUT"
    REWARD_MODEL_ERROR = "REWARD_MODEL_ERROR"
    PAUSE_TIMEOUT = "PAUSE_TIMEOUT"


def split_action(action: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Split 16-D action into left arm, left claw, right arm, right claw."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"expected action dim {ACTION_DIM}, got {a.shape[0]}")
    return (
        a[ARM_LEFT_IDX].copy(),
        float(a[CLAW_LEFT_IDX]),
        a[ARM_RIGHT_IDX].copy(),
        float(a[CLAW_RIGHT_IDX]),
    )


def compose_arm14(action: np.ndarray) -> np.ndarray:
    """Extract 14-D arm joints (rad) from 16-D action."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    return np.concatenate([a[ARM_LEFT_IDX], a[ARM_RIGHT_IDX]], axis=0)


def compose_claws(action: np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    return np.array([a[CLAW_LEFT_IDX], a[CLAW_RIGHT_IDX]], dtype=np.float32)


def validate_action_shape(action: Sequence[float] | np.ndarray) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"ACTION_SHAPE: expected {ACTION_DIM}, got {a.shape[0]}")
    return a
