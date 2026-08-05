"""Pure-Python pieces of the RL-100 topic-native contract.

The command cache deliberately has no ROS dependency.  This keeps the most
important episode-start, hold-last and causality rules unit-testable on the
training machine.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from kuavo_rl.contracts import (
    RL100_ACTION_DIM,
    RL100_ARM_COMMAND_DIM,
    RL100_HAND_COMMAND_DIM,
    RL100_HAND_DEFAULT,
    compose_rl100_topic_action,
)


def _header_stamp_s(msg: Any) -> float:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    try:
        value = float(stamp.to_sec()) if stamp is not None else 0.0
    except Exception:  # noqa: BLE001
        try:
            value = float(stamp)
        except Exception:  # noqa: BLE001
            value = 0.0
    return value if np.isfinite(value) and value > 0.0 else 0.0


def _positions(value: Any) -> np.ndarray:
    # ROS uint8[] may be exposed as bytes by some Python message generators.
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(value, dtype=np.uint8).astype(np.float32)
    return np.asarray(value, dtype=np.float32).reshape(-1)


@dataclass(frozen=True)
class TopicCommandSnapshot:
    arm14_deg: np.ndarray
    hand12_raw: np.ndarray
    arm_header_stamp_s: float
    hand_header_stamp_s: float
    arm_received_at_s: float
    hand_received_at_s: float
    arm_changed: bool
    hand_changed: bool
    arm_seen: bool
    hand_seen: bool
    arm_source: str
    hand_source: str

    @property
    def action26(self) -> np.ndarray:
        return compose_rl100_topic_action(self.arm14_deg, self.hand12_raw)


@dataclass(frozen=True)
class _CommandEntry:
    value: np.ndarray
    header_stamp_s: float
    received_at_s: float
    source: str


class TopicCommandCache:
    """Thread-safe, timestamp-aware hold-last cache for the two command topics.

    ``received_at_s`` is a monotonic process timestamp and is the causal clock.
    ROS header stamps are retained for diagnostics only.  A short history is
    kept so a callback that runs after a sampler's cutoff cannot leak into the
    earlier sample.
    """

    def __init__(self, *, hand_default: Sequence[float] | np.ndarray = RL100_HAND_DEFAULT):
        default = np.asarray(hand_default, dtype=np.float32).reshape(-1)
        if default.shape != (RL100_HAND_COMMAND_DIM,):
            raise ValueError("hand_default must be 12-D")
        if not np.isfinite(default).all() or np.any(default < 0.0) or np.any(default > 100.0):
            raise ValueError("hand_default must be finite and within [0, 100]")
        self.hand_default = default.copy()
        self._lock = threading.RLock()
        self._arm_history: deque[_CommandEntry] = deque(maxlen=256)
        self._hand_history: deque[_CommandEntry] = deque(maxlen=256)
        self._active = False
        self._generation = 0
        self._record_start_received_at = 0.0
        self._last_snapshot_cutoff = float("-inf")
        self.invalid_counts: dict[str, int] = {}
        self._last_header_stamp = {"arm": 0.0, "hand": 0.0}

    def reset(
        self,
        measured_arm14_rad: Sequence[float] | np.ndarray,
        *,
        generation: int | None = None,
        record_start_received_at: float | None = None,
    ) -> None:
        arm_rad = np.asarray(measured_arm14_rad, dtype=np.float32).reshape(-1)
        if arm_rad.shape != (RL100_ARM_COMMAND_DIM,):
            raise ValueError("measured_arm14_rad must be 14-D")
        if not np.isfinite(arm_rad).all():
            raise ValueError("measured_arm14_rad contains NaN/Inf")
        now = time.monotonic() if record_start_received_at is None else float(record_start_received_at)
        if not np.isfinite(now):
            raise ValueError("record_start_received_at must be finite")
        with self._lock:
            self._generation = int(self._generation + 1 if generation is None else generation)
            self._record_start_received_at = now
            self._last_snapshot_cutoff = now
            self._arm_history.clear()
            self._hand_history.clear()
            self._last_header_stamp = {"arm": 0.0, "hand": 0.0}
            self._arm_history.append(
                _CommandEntry(np.rad2deg(arm_rad).astype(np.float32), 0.0, now, "measured_hold")
            )
            self._hand_history.append(
                _CommandEntry(self.hand_default.copy(), 0.0, now, "default_hold")
            )
            self._active = True

    begin_episode = reset

    def end_episode(self) -> None:
        with self._lock:
            self._active = False

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def record_start_received_at(self) -> float:
        with self._lock:
            return self._record_start_received_at

    def update_arm(
        self,
        joint_state_msg: Any,
        *,
        received_at_s: float | None = None,
        generation: int | None = None,
    ) -> bool:
        position = np.asarray(getattr(joint_state_msg, "position", ()), dtype=np.float32).reshape(-1)
        if position.shape != (RL100_ARM_COMMAND_DIM,) or not np.isfinite(position).all():
            return False
        return self._update(
            self._arm_history,
            position,
            _header_stamp_s(joint_state_msg),
            received_at_s,
            generation,
            "topic",
            kind="arm",
        )

    def update_hand(
        self,
        robot_hand_position_msg: Any,
        *,
        received_at_s: float | None = None,
        generation: int | None = None,
    ) -> bool:
        left = _positions(getattr(robot_hand_position_msg, "left_hand_position", ()))
        right = _positions(getattr(robot_hand_position_msg, "right_hand_position", ()))
        if left.shape != (6,) or right.shape != (6,):
            return False
        hand = np.concatenate([left, right]).astype(np.float32, copy=False)
        if not np.isfinite(hand).all() or np.any(hand < 0.0) or np.any(hand > 100.0):
            return False
        return self._update(
            self._hand_history,
            hand,
            _header_stamp_s(robot_hand_position_msg),
            received_at_s,
            generation,
            "topic",
            kind="hand",
        )

    def _update(
        self,
        history: deque[_CommandEntry],
        value: np.ndarray,
        header_stamp_s: float,
        received_at_s: float | None,
        generation: int | None,
        source: str,
        *,
        kind: str,
    ) -> bool:
        received = time.monotonic() if received_at_s is None else float(received_at_s)
        with self._lock:
            if not self._active or (generation is not None and int(generation) != self._generation):
                return False
            if not np.isfinite(received) or received < self._record_start_received_at:
                self.invalid_counts[f"{kind}_received_at"] = self.invalid_counts.get(
                    f"{kind}_received_at", 0
                ) + 1
                return False
            if history and received < history[-1].received_at_s:
                self.invalid_counts[f"{kind}_received_at_backwards"] = self.invalid_counts.get(
                    f"{kind}_received_at_backwards", 0
                ) + 1
                return False
            if header_stamp_s <= 0.0:
                self.invalid_counts[f"{kind}_header_zero"] = self.invalid_counts.get(
                    f"{kind}_header_zero", 0
                ) + 1
            else:
                previous_header = self._last_header_stamp[kind]
                if previous_header > 0.0 and header_stamp_s < previous_header - 1e-6:
                    self.invalid_counts[f"{kind}_header_backwards"] = self.invalid_counts.get(
                        f"{kind}_header_backwards", 0
                    ) + 1
                    return False
                self._last_header_stamp[kind] = float(header_stamp_s)
            history.append(
                _CommandEntry(np.asarray(value, dtype=np.float32).copy(), header_stamp_s, received, source)
            )
            return True

    @staticmethod
    def _latest_before(history: deque[_CommandEntry], cutoff: float) -> _CommandEntry:
        selected: _CommandEntry | None = None
        for entry in history:
            if entry.received_at_s <= cutoff:
                selected = entry
            else:
                break
        if selected is None:
            raise RuntimeError("command cache has no entry at requested cutoff")
        return selected

    def has_any_topic_command(self, cutoff_received_at: float | None = None) -> bool:
        cutoff = time.monotonic() if cutoff_received_at is None else float(cutoff_received_at)
        with self._lock:
            return any(
                entry.source == "topic"
                and entry.received_at_s <= cutoff
                and entry.received_at_s >= self._record_start_received_at
                for entry in (*self._arm_history, *self._hand_history)
            )

    def snapshot_and_clear_changed(self, sample_cutoff_received_at: float | None = None) -> TopicCommandSnapshot:
        cutoff = time.monotonic() if sample_cutoff_received_at is None else float(sample_cutoff_received_at)
        with self._lock:
            if not self._active:
                raise RuntimeError("command cache is not active")
            if cutoff < self._last_snapshot_cutoff:
                raise ValueError("sample cutoff moved backwards")
            arm = self._latest_before(self._arm_history, cutoff)
            hand = self._latest_before(self._hand_history, cutoff)
            previous = self._last_snapshot_cutoff
            self._last_snapshot_cutoff = cutoff
            return TopicCommandSnapshot(
                arm14_deg=arm.value.copy(),
                hand12_raw=hand.value.copy(),
                arm_header_stamp_s=float(arm.header_stamp_s),
                hand_header_stamp_s=float(hand.header_stamp_s),
                arm_received_at_s=float(arm.received_at_s),
                hand_received_at_s=float(hand.received_at_s),
                arm_changed=arm.source == "topic" and arm.received_at_s > previous,
                hand_changed=hand.source == "topic" and hand.received_at_s > previous,
                arm_seen=any(e.source == "topic" and e.received_at_s <= cutoff for e in self._arm_history),
                hand_seen=any(e.source == "topic" and e.received_at_s <= cutoff for e in self._hand_history),
                arm_source=arm.source,
                hand_source=hand.source,
            )

    def snapshot(self, sample_cutoff_received_at: float | None = None) -> TopicCommandSnapshot:
        return self.snapshot_and_clear_changed(sample_cutoff_received_at)


def validate_topic_native_action(action: Sequence[float] | np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (RL100_ACTION_DIM,):
        raise ValueError(f"topic-native action expected {RL100_ACTION_DIM}-D, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("topic-native action contains NaN/Inf")
    if np.any(value[RL100_ARM_COMMAND_DIM:] < 0.0) or np.any(value[RL100_ARM_COMMAND_DIM:] > 100.0):
        raise ValueError("topic-native hand action outside [0, 100]")
    return value


def validate_topic_native_prediction(action: Sequence[float] | np.ndarray) -> np.ndarray:
    """Validate a model prediction before safety clipping.

    Unlike a dataset/topic action, a raw model output may temporarily leave the
    hand range; the safety gate is responsible for clipping it before publish.
    """
    value = np.asarray(action, dtype=np.float32).reshape(-1)
    if value.shape != (RL100_ACTION_DIM,):
        raise ValueError(f"topic-native prediction expected {RL100_ACTION_DIM}-D, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("topic-native prediction contains NaN/Inf")
    return value
