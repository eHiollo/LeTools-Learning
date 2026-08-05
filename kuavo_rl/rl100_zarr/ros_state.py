"""Raw ROS state hub for the RL-100 topic-native contract."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from kuavo_rl.contracts import (
    RL100_DEXHAND_JOINT_NAMES,
    RL100_DEXHAND_STATE_DIM,
    RL100_RAW_JOINT_DIM,
    compose_rl100_topic_state,
)


def _stamp_s(msg: Any) -> float:
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


def _time_field_s(value: Any) -> float:
    try:
        result = float(value.to_sec())
    except Exception:  # noqa: BLE001
        try:
            result = float(value)
        except Exception:  # noqa: BLE001
            result = 0.0
    return result if np.isfinite(result) and result > 0.0 else 0.0


@dataclass(frozen=True)
class TopicStateSample:
    state32: np.ndarray
    raw_joint_q20: np.ndarray
    dexhand_position12: np.ndarray
    joint_stamp_s: float
    hand_stamp_s: float
    joint_received_at_s: float
    hand_received_at_s: float
    joint_sensor_time_s: float
    joint_hand_skew_s: float
    joint_age_s: float
    hand_age_s: float


@dataclass
class _JointSample:
    value: np.ndarray
    stamp_s: float
    received_at_s: float
    sensor_time_s: float


@dataclass
class _HandSample:
    value: np.ndarray
    names: tuple[str, ...]
    stamp_s: float
    received_at_s: float


class TopicStateHub:
    """Subscribe to raw 20-D joint state and 12-D dexhand feedback.

    The class also exposes ``update_*`` methods for deterministic unit tests,
    so shape/order/unit checks do not require a ROS installation.
    """

    def __init__(
        self,
        *,
        sensors_topic: str = "/sensors_data_raw",
        dexhand_state_topic: str = "/dexhand/state",
    ) -> None:
        self.sensors_topic = str(sensors_topic)
        self.dexhand_state_topic = str(dexhand_state_topic)
        self._lock = threading.RLock()
        self._joint: _JointSample | None = None
        self._hand: _HandSample | None = None
        self._subs: list[Any] = []
        self._ros: Any | None = None
        self._joint_intervals: deque[float] = deque(maxlen=4096)
        self._hand_intervals: deque[float] = deque(maxlen=4096)
        self._last_joint_received: float | None = None
        self._last_hand_received: float | None = None
        self.invalid_counts = defaultdict(int)

    def start(self) -> None:
        if self._ros is not None:
            return
        import rospy
        from kuavo_msgs.msg import sensorsData
        from sensor_msgs.msg import JointState

        self._ros = rospy
        self._subs = [
            rospy.Subscriber(self.sensors_topic, sensorsData, self._on_sensors, queue_size=1),
            rospy.Subscriber(self.dexhand_state_topic, JointState, self._on_hand, queue_size=1),
        ]

    def close(self) -> None:
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:  # noqa: BLE001
                pass
        self._subs = []
        self._ros = None

    def _on_sensors(self, msg: Any) -> None:
        joint_data = getattr(msg, "joint_data", None)
        values = np.asarray(getattr(joint_data, "joint_q", ()), dtype=np.float32).reshape(-1)
        if values.shape != (RL100_RAW_JOINT_DIM,) or not np.isfinite(values).all():
            self.invalid_counts["joint_shape_or_finite"] += 1
            return
        received = time.time()
        with self._lock:
            self._record_interval("joint", received)
            self._joint = _JointSample(
                values.copy(),
                _stamp_s(msg) or received,
                received,
                _time_field_s(getattr(msg, "sensor_time", 0.0)),
            )

    def _on_hand(self, msg: Any) -> None:
        self.update_hand(
            getattr(msg, "position", ()),
            names=getattr(msg, "name", ()),
            stamp_s=_stamp_s(msg),
            received_at_s=time.time(),
        )

    def _record_interval(self, kind: str, received: float) -> None:
        if kind == "joint":
            if self._last_joint_received is not None:
                self._joint_intervals.append(received - self._last_joint_received)
            self._last_joint_received = received
        else:
            if self._last_hand_received is not None:
                self._hand_intervals.append(received - self._last_hand_received)
            self._last_hand_received = received

    def update_joint(
        self,
        values: Any,
        *,
        stamp_s: float = 0.0,
        received_at_s: float | None = None,
        sensor_time_s: float = 0.0,
    ) -> bool:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if array.shape != (RL100_RAW_JOINT_DIM,) or not np.isfinite(array).all():
            self.invalid_counts["joint_shape_or_finite"] += 1
            return False
        received = time.time() if received_at_s is None else float(received_at_s)
        if not np.isfinite(received):
            return False
        with self._lock:
            self._record_interval("joint", received)
            self._joint = _JointSample(
                array.copy(),
                float(stamp_s) if float(stamp_s) > 0.0 else received,
                received,
                float(sensor_time_s) if float(sensor_time_s) > 0.0 else 0.0,
            )
        return True

    def update_hand(
        self,
        values: Any,
        *,
        names: Any = RL100_DEXHAND_JOINT_NAMES,
        stamp_s: float = 0.0,
        received_at_s: float | None = None,
    ) -> bool:
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        raw_names = () if names is None else names
        normalized_names = tuple(str(x) for x in raw_names)
        if array.shape != (RL100_DEXHAND_STATE_DIM,):
            self.invalid_counts["hand_shape"] += 1
            return False
        if normalized_names != RL100_DEXHAND_JOINT_NAMES:
            self.invalid_counts["hand_names"] += 1
            return False
        if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 100.0):
            self.invalid_counts["hand_range_or_finite"] += 1
            return False
        received = time.time() if received_at_s is None else float(received_at_s)
        if not np.isfinite(received):
            return False
        with self._lock:
            self._record_interval("hand", received)
            self._hand = _HandSample(
                array.copy(),
                normalized_names,
                float(stamp_s) if float(stamp_s) > 0.0 else received,
                received,
            )
        return True

    def _snapshot(self) -> tuple[_JointSample, _HandSample]:
        with self._lock:
            if self._joint is None or self._hand is None:
                missing = []
                if self._joint is None:
                    missing.append("/sensors_data_raw")
                if self._hand is None:
                    missing.append("/dexhand/state")
                raise RuntimeError(f"missing raw state topics: {missing}")
            return (
                _JointSample(
                    self._joint.value.copy(),
                    self._joint.stamp_s,
                    self._joint.received_at_s,
                    self._joint.sensor_time_s,
                ),
                _HandSample(
                    self._hand.value.copy(),
                    self._hand.names,
                    self._hand.stamp_s,
                    self._hand.received_at_s,
                ),
            )

    def snapshot(
        self,
        joint_max_age_s: float,
        hand_max_age_s: float,
        max_skew_s: float,
    ) -> TopicStateSample:
        joint, hand = self._snapshot()
        now = time.time()
        joint_age = max(0.0, now - joint.received_at_s)
        hand_age = max(0.0, now - hand.received_at_s)
        skew = abs(joint.stamp_s - hand.stamp_s)
        if joint_age > float(joint_max_age_s):
            raise RuntimeError(f"raw joint state stale: {joint_age:.3f}s")
        if hand_age > float(hand_max_age_s):
            raise RuntimeError(f"dexhand state stale: {hand_age:.3f}s")
        if skew > float(max_skew_s):
            raise RuntimeError(f"raw joint/dexhand skew {skew:.3f}s")
        state = compose_rl100_topic_state(joint.value, hand.value)
        return TopicStateSample(
            state32=state,
            raw_joint_q20=joint.value,
            dexhand_position12=hand.value,
            joint_stamp_s=joint.stamp_s,
            hand_stamp_s=hand.stamp_s,
            joint_received_at_s=joint.received_at_s,
            hand_received_at_s=hand.received_at_s,
            joint_sensor_time_s=joint.sensor_time_s,
            joint_hand_skew_s=skew,
            joint_age_s=joint_age,
            hand_age_s=hand_age,
        )

    def _rate_report(self, intervals: deque[float]) -> dict[str, float | int | None]:
        values = np.asarray(intervals, dtype=np.float64)
        if values.size == 0:
            return {"count": 0, "hz": None, "interval_p50_s": None, "interval_p95_s": None, "interval_p99_s": None}
        return {
            "count": int(values.size),
            "hz": float(1.0 / np.mean(values)) if np.mean(values) > 0 else None,
            "interval_p50_s": float(np.quantile(values, 0.50)),
            "interval_p95_s": float(np.quantile(values, 0.95)),
            "interval_p99_s": float(np.quantile(values, 0.99)),
        }

    def preflight(self, timeout_s: float = 5.0) -> dict[str, Any]:
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            with self._lock:
                ready = self._joint is not None and self._hand is not None
            if ready:
                break
            time.sleep(0.02)
        with self._lock:
            report = {
                "ok": self._joint is not None and self._hand is not None,
                "sensors_topic": self.sensors_topic,
                "dexhand_state_topic": self.dexhand_state_topic,
                "raw_joint_dim": int(self._joint.value.size) if self._joint is not None else None,
                "dexhand_dim": int(self._hand.value.size) if self._hand is not None else None,
                "dexhand_names": list(self._hand.names) if self._hand is not None else None,
                "expected_dexhand_names": list(RL100_DEXHAND_JOINT_NAMES),
                "joint_rate": self._rate_report(self._joint_intervals),
                "dexhand_rate": self._rate_report(self._hand_intervals),
                "invalid_counts": dict(self.invalid_counts),
            }
        return report
