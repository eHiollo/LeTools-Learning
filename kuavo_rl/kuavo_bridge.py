"""Adapt existing Kuavo Gym env observations/actions to HIL-SERL contracts."""

from __future__ import annotations

from typing import Any

import numpy as np

from kuavo_rl.contracts import ACTION_DIM, IMAGE_KEYS, IMAGE_SHAPE_CHW, STATE_DIM
from kuavo_rl.ros_adapter import PublishedCommand


def _to_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    # KuavoBaseRosEnv often returns batched tensors: (1, D) / (1,C,H,W)
    if arr.ndim >= 1 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def normalize_kuavo_obs(raw_obs: dict) -> dict:
    """
    Convert Kuavo deploy obs dict into HIL-SERL contract keys.

    Accepts either already-prefixed keys or short camera names from ObsBuffer.
    """
    out: dict = {}
    state = raw_obs.get("observation.state", raw_obs.get("agent_pos"))
    if state is None:
        raise KeyError("Kuavo obs missing observation.state")
    state = _to_numpy(state).astype(np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(
            f"Kuavo observation.state dim {state.shape[0]} != {STATE_DIM}; "
            "check which_arm=both and arm_state_keys"
        )
    out["observation.state"] = state

    for key in IMAGE_KEYS:
        short = key.rsplit(".", 1)[-1]
        img = raw_obs.get(key)
        if img is None:
            img = raw_obs.get(f"observation.images.{short}")
        if img is None:
            img = raw_obs.get(short)
        if img is None:
            c, h, w = IMAGE_SHAPE_CHW
            out[key] = np.zeros((c, h, w), dtype=np.uint8)
            continue
        img = _to_numpy(img)
        if img.ndim == 3 and img.shape[-1] == 3:
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
        c, h, w = IMAGE_SHAPE_CHW
        if img.shape != (c, h, w):
            # Match ACT / dataset contract (3, 480, 848); sim RGB may publish 640x480.
            import cv2

            hwc = np.transpose(img, (1, 2, 0))
            if hwc.dtype != np.uint8:
                if float(np.max(hwc)) <= 1.0:
                    hwc = (hwc * 255.0).clip(0, 255)
                hwc = hwc.astype(np.uint8)
            hwc = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_AREA)
            img = np.transpose(hwc, (2, 0, 1))
        if img.dtype != np.uint8:
            # deploy may return float tensor in [0,1] or [0,255]
            if img.max() <= 1.0:
                img = (img * 255.0).clip(0, 255)
            img = img.astype(np.uint8)
        out[key] = img

    # optional timing metadata
    for meta in ("observation_age_s", "cross_topic_skew_s", "raw_joint_dim"):
        if meta in raw_obs:
            out[meta] = raw_obs[meta]
    return out


def _unwrap_kuavo_env(kuavo_env: Any) -> Any:
    """gym.make() wraps with TimeLimit/OrderEnforcing; Kuavo APIs live on .unwrapped."""
    env = kuavo_env
    seen: set[int] = set()
    while True:
        if hasattr(env, "get_obs") and hasattr(env, "exec_action"):
            return env
        nxt = getattr(env, "unwrapped", None)
        if nxt is None or nxt is env or id(nxt) in seen:
            return env
        seen.add(id(env))
        env = nxt


class KuavoGymBridge:
    """
    Thin wrapper around an existing Kuavo Gym env instance.

    Does not import rospy at module import time; pass an already-constructed env.
    """

    def __init__(self, kuavo_env: Any):
        self.env = _unwrap_kuavo_env(kuavo_env)
        self.wrapped_env = kuavo_env

    def reset(self, *, seed: int | None = None) -> dict:
        # Prefer outer wrapper reset (respects TimeLimit); fall back to unwrapped.
        target = self.wrapped_env if hasattr(self.wrapped_env, "reset") else self.env
        try:
            obs, info = target.reset(seed=seed)
        except TypeError:
            obs, info = target.reset()
        return normalize_kuavo_obs(obs)

    def get_obs(self) -> dict:
        if hasattr(self.env, "get_obs"):
            return normalize_kuavo_obs(self.env.get_obs())
        raise AttributeError("kuavo env missing get_obs()")

    def exec_action16(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != ACTION_DIM:
            raise ValueError(f"expected {ACTION_DIM}-D action, got {a.shape[0]}")
        if hasattr(self.env, "exec_action"):
            self.env.exec_action(a)
        else:
            raise AttributeError("kuavo env missing exec_action()")

    def publish_command(self, command: PublishedCommand) -> None:
        self.exec_action16(command.clipped_action)

    def is_stop(self) -> bool:
        ctrl = getattr(self.env, "control_signal_manager", None)
        return bool(ctrl is not None and ctrl.stop_flag.is_set())

    def is_pause(self) -> bool:
        ctrl = getattr(self.env, "control_signal_manager", None)
        return bool(ctrl is not None and ctrl.pause_flag.is_set())
