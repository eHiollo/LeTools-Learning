"""ROS / monotonic / wall dual timebase sampling."""

from __future__ import annotations

import time
from typing import Callable

from kuavo_rl.hil_recording.models import TimeStamps

_RosNowNs = Callable[[], int]


def _default_ros_time_ns() -> int:
    """Best-effort ROS time; falls back to wall ns when rospy unavailable."""
    try:
        import rospy  # type: ignore

        if rospy.core.is_initialized():
            t = rospy.Time.now()
            return int(t.to_nsec())
    except Exception:
        pass
    # Not ROS time — marked only via absence of rospy; callers must not treat
    # this as authoritative ROS time for alignment when dry_run=False.
    return int(time.time_ns())


_ros_now_ns: _RosNowNs = _default_ros_time_ns


def set_ros_time_provider(provider: _RosNowNs | None) -> None:
    """Inject ROS clock for tests / simulation."""
    global _ros_now_ns
    _ros_now_ns = provider or _default_ros_time_ns


def now_stamps(*, source_header_stamp_ns: int | None = None) -> TimeStamps:
    """Sample ros / monotonic / wall at one logical instant."""
    mono = time.monotonic_ns()
    wall = time.time_ns()
    ros = int(_ros_now_ns())
    return TimeStamps(
        ros_time_ns=ros,
        monotonic_time_ns=mono,
        wall_time_ns=wall,
        source_header_stamp_ns=source_header_stamp_ns,
    )


def header_stamp_to_ns(stamp: object) -> int | None:
    """Convert rospy.Time / genpy.Time / (sec, nsec) to ns."""
    if stamp is None:
        return None
    if isinstance(stamp, (int, float)):
        # Heuristic: values < 1e12 are seconds
        v = float(stamp)
        if v < 1e12:
            return int(v * 1e9)
        return int(v)
    secs = getattr(stamp, "secs", None)
    nsecs = getattr(stamp, "nsecs", None)
    if secs is not None and nsecs is not None:
        return int(secs) * 1_000_000_000 + int(nsecs)
    to_nsec = getattr(stamp, "to_nsec", None)
    if callable(to_nsec):
        return int(to_nsec())
    return None


def sleep_monotonic(duration_s: float) -> None:
    """Duration waits must use monotonic (via time.sleep is OS-backed)."""
    if duration_s > 0:
        time.sleep(duration_s)
