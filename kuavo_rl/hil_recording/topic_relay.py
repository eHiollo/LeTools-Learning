"""Relay live ROS bus topics onto canonical bag names (no lower-machine changes)."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TopicRelayHandle:
    pairs: list[tuple[str, str]]
    processes: list[subprocess.Popen] = field(default_factory=list)
    log_path: Path | None = None

    @property
    def pids(self) -> list[int]:
        return [int(p.pid) for p in self.processes if p.pid]


class TopicRelayManager:
    """Start ``topic_tools/relay`` (source → canonical) for rosbag-compatible names.

    Gate still probes ``source`` on the robot bus; bags record ``name``.
    """

    def start(
        self,
        pairs: list[tuple[str, str]],
        *,
        log_path: Path | None = None,
        ready_timeout_s: float = 3.0,
    ) -> TopicRelayHandle:
        if not pairs:
            return TopicRelayHandle(pairs=[])

        log_fp = None
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fp = open(log_path, "a", encoding="utf-8")

        procs: list[subprocess.Popen] = []
        try:
            for src, dst in pairs:
                if src == dst:
                    continue
                cmd = ["rosrun", "topic_tools", "relay", src, dst]
                proc = subprocess.Popen(
                    cmd,
                    stdout=log_fp or subprocess.DEVNULL,
                    stderr=subprocess.STDOUT if log_fp else subprocess.DEVNULL,
                    start_new_session=True,
                )
                procs.append(proc)
        except Exception:
            self._stop_procs(procs)
            if log_fp is not None:
                log_fp.close()
            raise

        handle = TopicRelayHandle(pairs=list(pairs), processes=procs, log_path=log_path)
        if ready_timeout_s > 0:
            self._wait_ready(pairs, timeout_s=ready_timeout_s)
        return handle

    def stop(self, handle: TopicRelayHandle | None) -> None:
        if handle is None:
            return
        self._stop_procs(handle.processes)
        handle.processes.clear()

    @staticmethod
    def _stop_procs(procs: list[subprocess.Popen]) -> None:
        for proc in procs:
            if proc.poll() is not None:
                continue
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.send_signal(signal.SIGINT)
                except ProcessLookupError:
                    continue
        deadline = time.monotonic() + 2.0
        for proc in procs:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining if remaining > 0 else 0.05)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass

    @staticmethod
    def _wait_ready(pairs: list[tuple[str, str]], *, timeout_s: float) -> None:
        """Best-effort: wait until remapped topics appear (skip if ROS unavailable)."""
        try:
            import rosgraph  # type: ignore
            import rospy  # type: ignore

            if not rosgraph.is_master_online():
                return
            if not rospy.core.is_initialized():
                # Do not init a node here; topic list query is enough.
                pass
        except Exception:
            return

        deadline = time.monotonic() + timeout_s
        targets = {dst for _, dst in pairs}
        while time.monotonic() < deadline:
            try:
                import rosgraph  # type: ignore

                master = rosgraph.Master("/hil_topic_relay_probe")
                published = {t for t, _ in master.getPublishedTopics("")}
                if targets.issubset(published):
                    return
            except Exception:
                return
            time.sleep(0.05)
