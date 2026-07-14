"""Teleop / VR intervention interface (VR wiring is future work)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from kuavo_rl.contracts import ACTION_DIM


@dataclass
class TeleopEvent:
    action: np.ndarray | None = None
    is_intervention: bool = False
    # 16-D boolean mask: dimensions actively controlled by the human.
    intervention_mask: np.ndarray | None = None
    success: bool = False
    failure: bool = False
    abort: bool = False
    pause: bool = False
    stop: bool = False
    deadman: bool = False
    source: str = "none"
    age_s: float | None = None


class TeleopAdapter:
    """
    Unified teleop event source.

    Default implementation is a software panel: call push_* from a UI/thread.
    VR devices should implement the same poll() contract.
    """

    def __init__(self, poll_fn: Callable[[], TeleopEvent] | None = None):
        self._poll_fn = poll_fn
        self._pending = TeleopEvent()

    def reset(self) -> None:
        self._pending = TeleopEvent()

    def poll(self) -> TeleopEvent:
        if self._poll_fn is not None:
            return self._poll_fn()
        # edge-triggered events are cleared after read
        ev = self._pending
        self._pending = TeleopEvent(
            action=ev.action,
            is_intervention=ev.is_intervention,
            intervention_mask=ev.intervention_mask,
            deadman=ev.deadman,
            pause=ev.pause,
            stop=ev.stop,
        )
        return ev

    def push_intervention(
        self, action: np.ndarray, *, deadman: bool = True, intervention_mask: np.ndarray | None = None
    ) -> None:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != ACTION_DIM:
            raise ValueError(f"teleop action must be {ACTION_DIM}-D")
        if intervention_mask is not None:
            mask = np.asarray(intervention_mask, dtype=bool).reshape(-1)
            if mask.shape[0] != ACTION_DIM:
                raise ValueError(f"intervention mask must be {ACTION_DIM}-D")
            self._pending.intervention_mask = mask
        else:
            self._pending.intervention_mask = np.ones(ACTION_DIM, dtype=bool)
        self._pending.action = a
        self._pending.is_intervention = True
        self._pending.deadman = deadman

    def release_intervention(self) -> None:
        self._pending.is_intervention = False
        self._pending.action = None
        self._pending.deadman = False

    def push_success(self) -> None:
        self._pending.success = True

    def push_failure(self) -> None:
        self._pending.failure = True

    def push_abort(self) -> None:
        self._pending.abort = True

    def set_pause(self, value: bool) -> None:
        self._pending.pause = value

    def set_stop(self, value: bool = True) -> None:
        self._pending.stop = value
