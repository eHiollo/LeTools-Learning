"""Build zarr buffers from labeled episode step lists (no ROS)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from kuavo_rl.rl100_zarr.reward import (
    assign_episode_rewards,
    is_success_result,
    should_keep_episode,
)
from kuavo_rl.rl100_zarr.schema import ZarrEpisodeBuffers


def append_labeled_episode(
    buffers: ZarrEpisodeBuffers,
    *,
    states: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    point_clouds: Sequence[np.ndarray],
    result_type: str,
    lambda_penalty: float = 0.05,
    smooth_penalty: float = 0.01,
    max_episode_len: int = 2000,
    only_success: bool = False,
    source: dict[str, Any] | None = None,
) -> bool:
    """Append one episode if it passes the keep filter. Returns whether kept."""
    if not should_keep_episode(result_type, only_success=only_success):
        return False

    n = len(actions)
    if not (len(states) == n == len(point_clouds)):
        raise ValueError(
            f"length mismatch states={len(states)} actions={n} pcs={len(point_clouds)}"
        )
    if n == 0:
        return False

    act_arr = np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0)
    rewards = assign_episode_rewards(
        act_arr,
        is_success=is_success_result(result_type),
        lambda_penalty=lambda_penalty,
        smooth_penalty=smooth_penalty,
        max_episode_len=max_episode_len,
    )

    for i in range(n):
        last = i == n - 1
        nxt = i if last else i + 1
        buffers.append_transition(
            point_cloud=point_clouds[i],
            state=states[i],
            action=actions[i],
            reward=float(rewards[i]),
            done=last,
            timeout=last,
            next_point_cloud=point_clouds[nxt],
            next_state=states[nxt],
            next_action=actions[nxt],
        )

    meta = {"result_type": result_type, "length": n}
    if source:
        meta.update(source)
    buffers.close_episode(source=meta)
    return True
