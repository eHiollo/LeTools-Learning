"""Sparse success reward + length/smooth penalties (RL-100 data_prepare style)."""

from __future__ import annotations

import numpy as np


def assign_episode_rewards(
    actions: np.ndarray,
    *,
    is_success: bool,
    lambda_penalty: float = 0.05,
    smooth_penalty: float = 0.01,
    max_episode_len: int = 2000,
) -> np.ndarray:
    """Build per-step rewards for one episode.

    - Intermediate steps: 0 (minus optional smoothness penalty)
    - Terminal step: ``+1`` on success (then length penalty), else ``0``
    - Failure episodes keep terminal reward 0 so Offline RL can down-weight them
    """
    acts = np.asarray(actions, dtype=np.float32)
    if acts.ndim != 2:
        raise ValueError(f"actions expected (T, Da), got {acts.shape}")
    t = acts.shape[0]
    if t == 0:
        return np.zeros((0,), dtype=np.float32)

    rewards = np.zeros((t,), dtype=np.float32)
    if is_success:
        terminal = 1.0 - float(lambda_penalty) * (t / max(max_episode_len, 1))
        rewards[-1] = float(terminal)

    if smooth_penalty > 0.0 and t > 1:
        deltas = np.linalg.norm(acts[1:] - acts[:-1], axis=-1)
        rewards[1:] -= float(smooth_penalty) * deltas.astype(np.float32)

    return rewards


def should_keep_episode(result_type: str, *, only_success: bool) -> bool:
    """Filter policy for zarr inclusion.

    ``abort`` / unlabeled episodes are dropped. Failures are kept unless
    ``only_success`` is set.
    """
    rt = (result_type or "").strip().lower()
    if rt in {"success", "success_candidate", "success_button"}:
        return True
    if rt in {"failure", "failure_candidate", "failure_button"}:
        return not only_success
    return False


def is_success_result(result_type: str) -> bool:
    rt = (result_type or "").strip().lower()
    return rt in {"success", "success_candidate", "success_button"}
