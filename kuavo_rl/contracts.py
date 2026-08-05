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

# RL-100 topic-native contract.  Keep this separate from the legacy 16-D
# HIL/ACT contract above: existing policies and generic environments depend on
# ACTION_DIM/STATE_DIM remaining 16.
RL100_TOPIC_NATIVE_CONTRACT = "rl100_topic_native_v1"
RL100_STATE_DIM = 32
RL100_ACTION_DIM = 26
RL100_RAW_JOINT_DIM = 20
RL100_DEXHAND_STATE_DIM = 12
RL100_ARM_COMMAND_DIM = 14
RL100_HAND_COMMAND_DIM = 12
RL100_ARM_SLICE_RAW20 = slice(4, 18)
RL100_ARM_JOINT_NAMES: tuple[str, ...] = tuple(f"arm_joint_{i}" for i in range(14))
RL100_DEXHAND_JOINT_NAMES: tuple[str, ...] = (
    "l_thumb",
    "l_thumb_aux",
    "l_index",
    "l_middle",
    "l_ring",
    "l_pinky",
    "r_thumb",
    "r_thumb_aux",
    "r_index",
    "r_middle",
    "r_ring",
    "r_pinky",
)
RL100_HAND_DEFAULT = np.array(
    [0, 99, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0], dtype=np.float32
)


def compose_rl100_topic_state(
    raw_joint_q20: Sequence[float] | np.ndarray,
    dexhand_position12: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compose state32 without slicing/reordering either source topic."""
    joints = np.asarray(raw_joint_q20, dtype=np.float32).reshape(-1)
    hand = np.asarray(dexhand_position12, dtype=np.float32).reshape(-1)
    if joints.shape != (RL100_RAW_JOINT_DIM,):
        raise ValueError(f"raw joint_q expected {RL100_RAW_JOINT_DIM}-D, got {joints.shape}")
    if hand.shape != (RL100_DEXHAND_STATE_DIM,):
        raise ValueError(f"dexhand position expected {RL100_DEXHAND_STATE_DIM}-D, got {hand.shape}")
    if not np.isfinite(joints).all() or not np.isfinite(hand).all():
        raise ValueError("topic-native state contains NaN/Inf")
    return np.concatenate([joints, hand]).astype(np.float32, copy=False)


def compose_rl100_topic_action(
    arm14_deg: Sequence[float] | np.ndarray,
    hand12_raw: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Compose action26 in the exact command-topic field order."""
    arm = np.asarray(arm14_deg, dtype=np.float32).reshape(-1)
    hand = np.asarray(hand12_raw, dtype=np.float32).reshape(-1)
    if arm.shape != (RL100_ARM_COMMAND_DIM,):
        raise ValueError(f"arm command expected {RL100_ARM_COMMAND_DIM}-D, got {arm.shape}")
    if hand.shape != (RL100_HAND_COMMAND_DIM,):
        raise ValueError(f"hand command expected {RL100_HAND_COMMAND_DIM}-D, got {hand.shape}")
    if not np.isfinite(arm).all() or not np.isfinite(hand).all():
        raise ValueError("topic-native action contains NaN/Inf")
    if np.any(hand < 0.0) or np.any(hand > 100.0):
        raise ValueError("hand command must be within [0, 100]")
    return np.concatenate([arm, hand]).astype(np.float32, copy=False)


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
    INFERENCE_TIMEOUT = "INFERENCE_TIMEOUT"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"


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
