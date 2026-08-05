"""RL-100 topic-native zarr schema.

Disk layout (matches RL-100 ``data_prepare.write_zarr`` / FoldingDataset)::

    <task>.zarr/
      data/
        point_cloud, next_point_cloud   (T, 1024, 3) float32
        state, next_state               (T, 32) float32   # training maps to agent_pos
        action, next_action             (T, 26) float32
        reward, return                  (T, 1) float32
        done, timeout                   (T, 1) bool
      meta/
        episode_ends                    (E,) int64
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from kuavo_rl.contracts import RL100_ACTION_DIM, RL100_STATE_DIM

STATE_DIM = RL100_STATE_DIM
ACTION_DIM = RL100_ACTION_DIM

NUM_POINTS = 1024
POINT_DIM = 3
RETURN_GAMMA = 0.99

RL100_DATA_KEYS = (
    "point_cloud",
    "next_point_cloud",
    "state",
    "next_state",
    "action",
    "next_action",
    "reward",
    "return",
    "done",
    "timeout",
)


@dataclass
class ZarrEpisodeBuffers:
    """Append-only transition buffers for one or more episodes."""

    point_cloud: list[np.ndarray] = field(default_factory=list)
    next_point_cloud: list[np.ndarray] = field(default_factory=list)
    state: list[np.ndarray] = field(default_factory=list)
    next_state: list[np.ndarray] = field(default_factory=list)
    action: list[np.ndarray] = field(default_factory=list)
    next_action: list[np.ndarray] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    done: list[bool] = field(default_factory=list)
    timeout: list[bool] = field(default_factory=list)
    episode_ends: list[int] = field(default_factory=list)
    source_manifest: list[dict[str, Any]] = field(default_factory=list)
    audit: dict[str, list[Any]] = field(default_factory=dict)
    _audit_keys: tuple[str, ...] | None = None
    _total_count: int = 0

    def __len__(self) -> int:
        return len(self.action)

    def append_transition(
        self,
        *,
        point_cloud: np.ndarray,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        timeout: bool,
        next_point_cloud: np.ndarray | None = None,
        next_state: np.ndarray | None = None,
        next_action: np.ndarray | None = None,
        audit: Mapping[str, Any] | None = None,
    ) -> None:
        pc = _as_pc(point_cloud)
        st = _as_vec(state, STATE_DIM, "state")
        act = _as_vec(action, ACTION_DIM, "action")
        npc = pc if next_point_cloud is None else _as_pc(next_point_cloud)
        nst = st if next_state is None else _as_vec(next_state, STATE_DIM, "next_state")
        nact = act if next_action is None else _as_vec(next_action, ACTION_DIM, "next_action")
        for name, value in (("state", st), ("next_state", nst)):
            if np.any(value[20:] < 0.0) or np.any(value[20:] > 100.0):
                raise ValueError(f"{name} dexhand values must be within [0,100]")
        for name, value in (("action", act), ("next_action", nact)):
            if np.any(value[14:] < 0.0) or np.any(value[14:] > 100.0):
                raise ValueError(f"{name} hand command values must be within [0,100]")

        audit_keys: tuple[str, ...] | None = None
        if audit is not None:
            audit_keys = tuple(sorted(str(k) for k in audit))
            if self._audit_keys is None:
                if self._total_count > 0:
                    raise ValueError("audit metadata must be present for every transition")
            elif audit_keys != self._audit_keys:
                raise ValueError(
                    f"audit keys changed within zarr buffer: {audit_keys} != {self._audit_keys}"
                )
        elif self._audit_keys is not None:
            raise ValueError("audit metadata missing for a transition after audit started")

        self.point_cloud.append(pc)
        self.next_point_cloud.append(npc)
        self.state.append(st)
        self.next_state.append(nst)
        self.action.append(act)
        self.next_action.append(nact)
        self.reward.append(float(reward))
        self.done.append(bool(done))
        self.timeout.append(bool(timeout))
        if audit is not None:
            if self._audit_keys is None:
                self._audit_keys = audit_keys
                self.audit = {key: [] for key in audit_keys or ()}
            for key in self._audit_keys:
                value = audit[key]
                if isinstance(value, np.ndarray):
                    value = value.copy()
                self.audit[key].append(value)
        elif self._audit_keys is not None:
            raise ValueError("audit metadata missing for a transition after audit started")
        self._total_count += 1

    def close_episode(self, *, source: dict[str, Any] | None = None) -> None:
        if self._total_count <= 0:
            raise RuntimeError("close_episode called with no transitions")
        if self.episode_ends and self.episode_ends[-1] == self._total_count:
            raise RuntimeError("episode already closed")
        self.episode_ends.append(int(self._total_count))
        if source is not None:
            self.source_manifest.append(dict(source))

    def n_episodes(self) -> int:
        return len(self.episode_ends)


def _as_vec(x: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.shape[0] != dim:
        raise ValueError(f"{name} expected dim {dim}, got {arr.shape[0]}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr.copy()


def _as_pc(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"point_cloud expected (N, >=3), got {arr.shape}")
    if arr.shape[0] != NUM_POINTS:
        raise ValueError(f"point_cloud expected N={NUM_POINTS}, got {arr.shape[0]}")
    out = arr[:, :POINT_DIM].astype(np.float32, copy=True)
    if not np.all(np.isfinite(out)):
        raise ValueError("point_cloud contains non-finite values")
    return out


def compute_return(reward: np.ndarray, not_done: np.ndarray, gamma: float = RETURN_GAMMA) -> np.ndarray:
    """Backward discounted return; ``not_done`` is 0 at terminal/timeout steps."""
    r = np.asarray(reward, dtype=np.float32).reshape(-1)
    nd = np.asarray(not_done, dtype=np.float32).reshape(-1)
    if r.shape != nd.shape:
        raise ValueError("reward and not_done length mismatch")
    out = np.zeros((r.shape[0], 1), dtype=np.float32)
    running = 0.0
    for i in range(r.shape[0] - 1, -1, -1):
        running = float(r[i]) + float(gamma) * running * float(nd[i])
        out[i, 0] = running
    return out
