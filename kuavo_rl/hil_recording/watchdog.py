"""Recorder watchdog: read cached state only; emit StopRequest, never control robot."""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from kuavo_rl.hil_recording.config import RecordingConfig
from kuavo_rl.hil_recording.models import StopRequest
from kuavo_rl.hil_recording.rosbag_recorder import RecorderHandle, _pid_alive
from kuavo_rl.hil_recording.timebase import now_stamps


@dataclass
class TopicFreshnessCache:
    """Updated by external subscription callbacks; watchdog only reads."""

    last_msg_ros_ns: dict[str, int] = field(default_factory=dict)
    last_msg_mono_ns: dict[str, int] = field(default_factory=dict)
    latched_seen: dict[str, bool] = field(default_factory=dict)

    def note(self, topic: str, *, ros_ns: int, mono_ns: int | None = None, latched: bool = False) -> None:
        self.last_msg_ros_ns[topic] = int(ros_ns)
        self.last_msg_mono_ns[topic] = int(mono_ns if mono_ns is not None else time.monotonic_ns())
        if latched:
            self.latched_seen[topic] = True


class RecorderWatchdog:
    def __init__(
        self,
        config: RecordingConfig,
        stop_queue: Queue[StopRequest],
        *,
        freshness: TopicFreshnessCache | None = None,
        streaming_freshness_s: dict[str, float] | None = None,
    ):
        self.config = config
        self.stop_queue = stop_queue
        self.freshness = freshness or TopicFreshnessCache()
        self.streaming_freshness_s = streaming_freshness_s or {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._handle: RecorderHandle | None = None
        self._episode_id: str | None = None
        self._events_path: Path | None = None
        self._report: dict[str, Any] = {}
        self._last_bag_size = 0
        self._last_bag_growth_mono = 0
        self._last_bag_stat_mono = 0
        self._last_disk_check_mono = 0
        self._stop_requested = False

    def start(self, handle: RecorderHandle, episode_id: str) -> None:
        self._handle = handle
        self._episode_id = episode_id
        self._stop.clear()
        self._stop_requested = False
        session_dir = self.config.session_dir(episode_id)
        self._events_path = session_dir / "watchdog.events.jsonl"
        self._last_bag_size = 0
        self._last_bag_growth_mono = time.monotonic_ns()
        self._last_bag_stat_mono = 0
        self._last_disk_check_mono = 0
        self._thread = threading.Thread(target=self._loop, name="hil-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        report = dict(self._report)
        if self._episode_id:
            path = self.config.session_dir(self._episode_id) / "watchdog.report.json"
            path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return report

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                self._log_event({"type": "watchdog_exception", "error": str(exc)})
            self._stop.wait(self.config.watchdog_loop_s)

    def _tick(self) -> None:
        """500ms loop: only read cached / cheap OS state — no heavy topic probes."""
        handle = self._handle
        episode_id = self._episode_id
        if handle is None or episode_id is None:
            return
        now_mono = time.monotonic_ns()
        stamps = now_stamps()

        alive = _pid_alive(handle.pid)
        if handle.process is not None and handle.process.poll() is not None:
            alive = False

        # bag size stat at 1s
        bag_size = self._last_bag_size
        if (now_mono - self._last_bag_stat_mono) / 1e9 >= self.config.bag_stat_interval_s:
            self._last_bag_stat_mono = now_mono
            if handle.bag_path.exists():
                bag_size = handle.bag_path.stat().st_size
            else:
                # rosbag may write without .bag suffix until stop; also check stem
                alt = handle.bag_path.with_suffix(".bag.active")
                if alt.exists():
                    bag_size = alt.stat().st_size
            if bag_size > self._last_bag_size:
                self._last_bag_size = bag_size
                self._last_bag_growth_mono = now_mono
            else:
                self._last_bag_size = bag_size

        stalled = (
            alive
            and (now_mono - self._last_bag_growth_mono) / 1e9 >= self.config.bag_stall_timeout_s
        )

        # disk at 5s
        disk_pct = None
        if (now_mono - self._last_disk_check_mono) / 1e9 >= self.config.disk_check_interval_s:
            self._last_disk_check_mono = now_mono
            usage = shutil.disk_usage(self.config.root_dir)
            disk_pct = 100.0 * (1.0 - usage.free / max(usage.total, 1))

        stale_topics: list[str] = []
        for topic, freshness_s in self.streaming_freshness_s.items():
            last = self.freshness.last_msg_mono_ns.get(topic)
            if last is None:
                continue
            age_s = (now_mono - last) / 1e9
            if age_s > freshness_s:
                stale_topics.append(topic)

        self._report = {
            "episode_id": episode_id,
            "pid": handle.pid,
            "alive": alive,
            "bag_size": bag_size,
            "bag_stalled": stalled,
            "disk_usage_percent": disk_pct,
            "stale_topics": stale_topics,
            "checked_at_ros_ns": stamps.ros_time_ns,
            "checked_at_mono_ns": stamps.monotonic_time_ns,
        }

        if self._stop_requested:
            return

        if not alive:
            self._emit_stop(episode_id, "record_error:recorder_exit", stamps)
            return
        if stalled:
            self._emit_stop(episode_id, "record_error:bag_stalled", stamps)
            return
        if disk_pct is not None and disk_pct >= self.config.hard_stop_disk_percent:
            self._emit_stop(episode_id, "record_error:disk_hard_stop", stamps)
            return
        if stale_topics:
            self._emit_stop(
                episode_id,
                f"record_error:stale_topics:{','.join(stale_topics)}",
                stamps,
            )

    def _emit_stop(self, episode_id: str, reason: str, stamps) -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        self._log_event({"type": "stop_request", "reason": reason})
        self.stop_queue.put(
            StopRequest(
                episode_id=episode_id,
                reason=reason,
                source="watchdog",
                stamps=stamps,
            )
        )

    def _log_event(self, payload: dict[str, Any]) -> None:
        if self._events_path is None:
            return
        row = {"mono_ns": time.monotonic_ns(), **payload}
        with self._events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def drain_stop_queue(queue: Queue[StopRequest]) -> StopRequest | None:
    latest: StopRequest | None = None
    while True:
        try:
            latest = queue.get_nowait()
        except Empty:
            break
    return latest
