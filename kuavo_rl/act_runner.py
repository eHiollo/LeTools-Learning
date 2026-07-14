"""Stage-A ACT runner: predict chunk_size actions, execute only the first step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from kuavo_rl.config import ActRunnerConfig
from kuavo_rl.contracts import ACTION_DIM


class ActionChunkPolicy(Protocol):
    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        """Return (chunk_size, action_dim) or (action_dim,) array."""
        ...


@dataclass
class ActStepResult:
    action: np.ndarray
    chunk: np.ndarray
    executed_index: int
    discarded_tail: np.ndarray


class ActExecuteFirstRunner:
    """
    Enforces handbook stage-A contract:
    - model may predict chunk_size>=1
    - only index 0 is executed via env.step
    - remaining chunk is discarded (no queue backlog)
    """

    def __init__(self, policy: ActionChunkPolicy, config: ActRunnerConfig | None = None):
        self.policy = policy
        self.config = config or ActRunnerConfig()
        if self.config.execute_steps != 1:
            raise ValueError("execute_steps must be 1")
        self.last_chunk: np.ndarray | None = None
        self.pending_queue: list[np.ndarray] = []  # always kept empty by design

    def clear_queue(self) -> None:
        """Emergency stop / pause must clear any residual queue."""
        self.pending_queue.clear()
        self.last_chunk = None

    def select_action(self, obs: dict) -> ActStepResult:
        chunk = np.asarray(self.policy.predict_action_chunk(obs), dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if chunk.ndim != 2:
            raise ValueError(f"action chunk must be 2-D, got shape {chunk.shape}")
        if chunk.shape[1] != ACTION_DIM:
            raise ValueError(
                f"action dim {chunk.shape[1]} != {ACTION_DIM}; refusing silent reshape"
            )
        if chunk.shape[0] < 1:
            raise ValueError("empty action chunk")
        # Optionally warn if policy chunk != configured chunk_size, but still take first.
        self.last_chunk = chunk
        action = chunk[0].copy()
        discarded = chunk[1:].copy()
        # Critical: do NOT enqueue discarded steps
        self.pending_queue.clear()
        return ActStepResult(
            action=action,
            chunk=chunk,
            executed_index=0,
            discarded_tail=discarded,
        )

    def run_episode(
        self,
        env: Any,
        *,
        max_steps: int | None = None,
        on_step: Callable[[dict], None] | None = None,
    ) -> dict:
        obs, info = env.reset()
        history = []
        steps = max_steps or getattr(getattr(env, "config", None), "episode", None)
        limit = max_steps
        if limit is None:
            limit = getattr(getattr(env, "config", None), "episode", None)
            limit = getattr(limit, "max_steps", 50) if limit is not None else 50

        for step_id in range(int(limit)):
            result = self.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(result.action)
            record = {
                # policy_action is always the ACT proposal.  executed_action is
                # resolved by the env and can instead be the VR action.
                "action": result.action,
                "step_id": step_id,
                "policy_action": result.action.copy(),
                "executed_action": info.get("action_audit", {}).get("raw_action"),
                "chunk_len": int(result.chunk.shape[0]),
                "discarded": int(result.discarded_tail.shape[0]),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
            if on_step:
                callback_record = dict(record)
                callback_record["observation"] = obs
                callback_record["next_observation"] = next_obs
                on_step(callback_record)
            history.append(record)
            obs = next_obs
            if terminated or truncated:
                break
            # Re-observe and re-predict every step — no backlog
            assert len(self.pending_queue) == 0
        return {"steps": history, "n": len(history)}


class ConstantChunkPolicy:
    """Test policy that returns a fixed chunk."""

    def __init__(self, chunk: np.ndarray):
        self.chunk = np.asarray(chunk, dtype=np.float32)

    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        return self.chunk.copy()
