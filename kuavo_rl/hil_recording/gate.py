"""Pre-record gate: ROS / topic profile / disk / session mutex (ACT is not a conflict)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

from kuavo_rl.hil_recording.config import RecordingConfig
from kuavo_rl.hil_recording.database import HILDatabase
from kuavo_rl.hil_recording.models import GateReport, ProducerInfo
from kuavo_rl.hil_recording.timebase import now_stamps
from kuavo_rl.hil_recording.topics import ResolvedTopics


TopicPresenceFn = Callable[[str], bool]
TopicRateFn = Callable[[str], float | None]
LatchedSeenFn = Callable[[str], bool]
RosMasterOkFn = Callable[[], bool]


class RecordGate:
    def __init__(
        self,
        config: RecordingConfig,
        db: HILDatabase,
        *,
        topic_present: TopicPresenceFn | None = None,
        topic_rate_hz: TopicRateFn | None = None,
        latched_seen: LatchedSeenFn | None = None,
        ros_master_ok: RosMasterOkFn | None = None,
    ):
        self.config = config
        self.db = db
        self._topic_present = topic_present or self._default_topic_present
        self._topic_rate_hz = topic_rate_hz or self._default_topic_rate
        self._latched_seen = latched_seen or self._default_latched_seen
        self._ros_master_ok = ros_master_ok or self._default_ros_master_ok

    def evaluate(
        self,
        *,
        resolved: ResolvedTopics,
        session_dir: Path,
        producers: list[ProducerInfo],
        episode_id: str | None = None,
        skip_ros: bool | None = None,
    ) -> GateReport:
        stamps = now_stamps()
        reasons: list[str] = []
        missing: list[str] = []
        low_rate: list[str] = []
        stale: list[str] = []
        latched_missing: list[str] = []

        skip = self.config.skip_gate_ros if skip_ros is None else skip_ros

        # Disk
        usage = _disk_usage_percent(session_dir)
        if usage >= self.config.start_block_disk_percent:
            reasons.append(
                f"disk_usage={usage:.1f}% >= start_block={self.config.start_block_disk_percent}"
            )

        # Writable
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
            probe = session_dir / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            reasons.append(f"session_dir_not_writable: {exc}")

        # Active session / .active mutex (ACT producer is NOT a conflict)
        for row in self.db.list_active_sessions():
            if episode_id and row["episode_id"] == episode_id:
                continue
            reasons.append(
                f"active_session_conflict: {row['episode_id']} state={row['session_state']}"
            )

        active_marker = self.config.root_dir / ".active"
        if active_marker.exists():
            try:
                content = active_marker.read_text(encoding="utf-8").strip()
            except OSError:
                content = "?"
            if not episode_id or content != episode_id:
                reasons.append(f"unrecovered_.active file present ({content})")

        # Second recorder: any other session with record_pid > 0 in active states
        for row in self.db.list_active_sessions():
            if episode_id and row["episode_id"] == episode_id:
                continue
            if int(row.get("record_pid") or 0) > 0:
                reasons.append(
                    f"second_recorder_conflict: episode={row['episode_id']} pid={row['record_pid']}"
                )

        if not skip:
            if not self._ros_master_ok():
                reasons.append("ros_master_unavailable")
            for spec in resolved.for_start():
                if spec.mode == "latched":
                    if not self._latched_seen(spec.name):
                        latched_missing.append(spec.name)
                        reasons.append(f"latched_missing:{spec.name}")
                    continue
                # streaming
                if not self._topic_present(spec.name):
                    missing.append(spec.name)
                    reasons.append(f"missing_topic:{spec.name}")
                    continue
                if spec.min_hz is not None:
                    rate = self._topic_rate_hz(spec.name)
                    # None = no heavy probe injected; presence already passed.
                    if rate is not None and rate < float(spec.min_hz):
                        low_rate.append(spec.name)
                        reasons.append(
                            f"low_rate:{spec.name} rate={rate} min={spec.min_hz}"
                        )

        producer_dicts = [{"name": p.name, "pid": p.pid, "kind": p.kind} for p in producers]
        status = "Pass"
        if reasons:
            # Degraded only for non-blocking soft issues when explicitly allowed later;
            # gate hard-blocks on any reason for start safety.
            status = "Block"

        report = GateReport(
            status=status,
            checked_at_ros_ns=stamps.ros_time_ns,
            checked_at_mono_ns=stamps.monotonic_time_ns,
            missing_topics=missing,
            low_rate_topics=low_rate,
            stale_topics=stale,
            latched_missing=latched_missing,
            disk_usage_percent=usage,
            producers=producer_dicts,
            reasons=reasons,
        )
        gate_path = session_dir / "gate.json"
        gate_path.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _default_ros_master_ok() -> bool:
        try:
            import rosgraph  # type: ignore

            return bool(rosgraph.is_master_online())
        except Exception:
            return False

    @staticmethod
    def _default_topic_present(name: str) -> bool:
        try:
            import rospy  # type: ignore

            topics = {t[0] for t in rospy.get_published_topics()}
            return name in topics
        except Exception:
            return False

    @staticmethod
    def _default_topic_rate(_name: str) -> float | None:
        # Heavy probe intentionally NOT done here every call in production;
        # inject a cached rate provider from session/watchdog.
        return None

    @staticmethod
    def _default_latched_seen(name: str) -> bool:
        # Best-effort: presence on master is enough for latched at gate time.
        return RecordGate._default_topic_present(name)


def _disk_usage_percent(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return 100.0 * (1.0 - usage.free / max(usage.total, 1))
