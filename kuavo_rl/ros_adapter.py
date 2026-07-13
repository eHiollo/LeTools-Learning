"""ROS boundary conversions and state slicing (no rospy import at module level)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from kuavo_rl.contracts import (
    ACTION_DIM,
    ARM_SLICE_BY_RAW_DIM,
    CLAW_IDX_IN_ACTION,
    RAW_STATE_DIM_V62,
    STATE_DIM,
    compose_arm14,
    compose_claws,
    split_action,
    validate_action_shape,
)


@dataclass
class PublishedCommand:
    arm14_rad: np.ndarray
    arm14_deg: np.ndarray
    claws_norm: np.ndarray
    claws_command: np.ndarray
    raw_action: np.ndarray
    clipped_action: np.ndarray


def arm_slice_for_raw_dim(raw_dim: int) -> slice:
    if raw_dim not in ARM_SLICE_BY_RAW_DIM:
        raise ValueError(
            f"unsupported raw state dim {raw_dim}; "
            f"expected one of {sorted(ARM_SLICE_BY_RAW_DIM)}"
        )
    return ARM_SLICE_BY_RAW_DIM[raw_dim]


def slice_arm_state(raw_joint_q: np.ndarray, raw_dim: int | None = None) -> np.ndarray:
    """Extract 14-D arm joints (rad) from raw joint_q using QC layout rule."""
    q = np.asarray(raw_joint_q, dtype=np.float32).reshape(-1)
    dim = int(raw_dim if raw_dim is not None else q.shape[0])
    if q.shape[0] != dim:
        raise ValueError(f"raw joint_q length {q.shape[0]} != declared dim {dim}")
    sl = arm_slice_for_raw_dim(dim)
    arm = q[sl]
    if arm.shape[0] != 14:
        raise ValueError(f"arm slice produced {arm.shape[0]} joints, expected 14")
    return arm.astype(np.float32)


def compose_state16(arm14_rad: np.ndarray, claws_norm: np.ndarray) -> np.ndarray:
    """Compose dataset/env 16-D state: L7, left_claw, R7, right_claw."""
    arm = np.asarray(arm14_rad, dtype=np.float32).reshape(14)
    claws = np.asarray(claws_norm, dtype=np.float32).reshape(2)
    state = np.concatenate([arm[:7], claws[:1], arm[7:], claws[1:]], axis=0)
    if state.shape[0] != STATE_DIM:
        raise ValueError("composed state must be 16-D")
    return state.astype(np.float32)


def arm_rad_to_traj_deg(arm14_rad: np.ndarray) -> np.ndarray:
    """Convert 14-D arm rad -> deg for direct /kuavo_arm_traj publish."""
    return np.rad2deg(np.asarray(arm14_rad, dtype=np.float32)).astype(np.float32)


def claws_norm_to_command(claws_norm: np.ndarray, scale: float = 100.0) -> np.ndarray:
    claws = np.clip(np.asarray(claws_norm, dtype=np.float32).reshape(2), 0.0, 1.0)
    return (claws * scale).astype(np.float32)


def build_published_command(
    raw_action: np.ndarray,
    clipped_action: np.ndarray,
    *,
    claw_scale: float = 100.0,
) -> PublishedCommand:
    clipped = validate_action_shape(clipped_action)
    arm14 = compose_arm14(clipped)
    claws = compose_claws(clipped)
    return PublishedCommand(
        arm14_rad=arm14,
        arm14_deg=arm_rad_to_traj_deg(arm14),
        claws_norm=claws,
        claws_command=claws_norm_to_command(claws, scale=claw_scale),
        raw_action=validate_action_shape(raw_action),
        clipped_action=clipped,
    )


def default_v62_joint_map() -> dict:
    return {
        "robot": "Kuavo 5W v62",
        "raw_state_dim": RAW_STATE_DIM_V62,
        "arm_slice": [12, 26],
        "layout_rule": "28->12:26, 20->4:18, 14->0:14",
        "action_dim": ACTION_DIM,
        "action_order": "L7,left_claw,R7,right_claw",
        "state_unit": "rad",
        "claw_norm": "[0,1]",
        "claw_command_scale": 100.0,
        "note": "Do not use platform_config 5w=[4:18] for 28-D v62 state.",
    }


def observation_contract_check(obs: Mapping) -> list[str]:
    """Return list of contract violations (empty if OK)."""
    errors: list[str] = []
    if "observation.state" not in obs:
        errors.append("missing observation.state")
    else:
        state = np.asarray(obs["observation.state"])
        if state.shape != (STATE_DIM,):
            errors.append(f"observation.state shape {state.shape} != ({STATE_DIM},)")
        if state.dtype != np.float32:
            errors.append(f"observation.state dtype {state.dtype} != float32")
    for key in (
        "observation.images.head_cam_h",
        "observation.images.wrist_cam_l",
        "observation.images.wrist_cam_r",
    ):
        if key not in obs:
            errors.append(f"missing {key}")
            continue
        img = np.asarray(obs[key])
        if img.ndim != 3:
            errors.append(f"{key} ndim {img.ndim} != 3")
    return errors


def action_to_audit_dict(cmd: PublishedCommand) -> dict:
    left, lc, right, rc = split_action(cmd.clipped_action)
    return {
        "raw_action": cmd.raw_action.tolist(),
        "clipped_action": cmd.clipped_action.tolist(),
        "arm14_rad": cmd.arm14_rad.tolist(),
        "arm14_deg": cmd.arm14_deg.tolist(),
        "claws_norm": cmd.claws_norm.tolist(),
        "claws_command": cmd.claws_command.tolist(),
        "left_arm": left.tolist(),
        "left_claw": lc,
        "right_arm": right.tolist(),
        "right_claw": rc,
        "claw_indices": list(CLAW_IDX_IN_ACTION),
    }
