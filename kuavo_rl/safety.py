"""SafetyGate: pure numpy checks before any robot command is published."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kuavo_rl.config import SafetyConfig
from kuavo_rl.contracts import ACTION_DIM, FaultCode, validate_action_shape


@dataclass
class SafetyResult:
    ok: bool
    action: np.ndarray
    clipped: bool
    fault_code: FaultCode
    reason: str = ""


class SafetyGate:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self._last_action: np.ndarray | None = None
        self.consecutive_clips = 0

    def reset(self, initial_action: np.ndarray | None = None) -> None:
        self._last_action = (
            None if initial_action is None else np.asarray(initial_action, dtype=np.float32).copy()
        )
        self.consecutive_clips = 0

    def check(
        self,
        action: np.ndarray,
        *,
        stop: bool = False,
        ros_shutdown: bool = False,
        observation_age_s: float | None = None,
        cross_topic_skew_s: float | None = None,
    ) -> SafetyResult:
        if stop:
            return SafetyResult(False, self._hold(), False, FaultCode.STOP_SIGNAL, "stop")
        if ros_shutdown:
            return SafetyResult(False, self._hold(), False, FaultCode.ROS_SHUTDOWN, "ros shutdown")

        try:
            raw = validate_action_shape(action)
        except ValueError as exc:
            return SafetyResult(False, self._hold(), False, FaultCode.ACTION_SHAPE, str(exc))

        if not np.isfinite(raw).all():
            return SafetyResult(False, self._hold(), False, FaultCode.ACTION_NAN, "NaN/Inf in action")

        if observation_age_s is not None and observation_age_s > self.config.observation_max_age_s:
            return SafetyResult(
                False,
                self._hold(),
                False,
                FaultCode.STALE_OBSERVATION,
                f"obs age {observation_age_s:.3f}s",
            )
        if (
            cross_topic_skew_s is not None
            and cross_topic_skew_s > self.config.max_cross_topic_skew_s
        ):
            return SafetyResult(
                False,
                self._hold(),
                False,
                FaultCode.STALE_OBSERVATION,
                f"topic skew {cross_topic_skew_s:.3f}s",
            )

        clipped = raw.copy()
        was_clipped = False

        # Position bounds
        bounded = np.clip(clipped, self.config.joint_position_low, self.config.joint_position_high)
        if not np.allclose(bounded, clipped):
            was_clipped = True
            clipped = bounded

        # Adjacent-command delta / velocity proxy: always soft-clip.
        # Hard VELOCITY_LIMIT is reserved for callers that disable soft-clip (not default).
        if self._last_action is not None:
            delta = clipped - self._last_action
            max_delta = self.config.max_delta_rad
            limited = np.clip(delta, -max_delta, max_delta)
            if not np.allclose(limited, delta):
                was_clipped = True
                clipped = self._last_action + limited

        if was_clipped:
            self.consecutive_clips += 1
        else:
            self.consecutive_clips = 0

        self._last_action = clipped.copy()
        return SafetyResult(True, clipped, was_clipped, FaultCode.NONE, "")

    def clips_exceeded(self) -> bool:
        return self.consecutive_clips >= self.config.max_consecutive_clips

    def _hold(self) -> np.ndarray:
        if self._last_action is not None:
            return self._last_action.copy()
        return np.zeros(ACTION_DIM, dtype=np.float32)
