"""rosbag record subprocess management with .active marker."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from kuavo_rl.hil_recording.config import RecordingConfig


@dataclass
class RecorderHandle:
    episode_id: str
    pid: int
    command: list[str]
    bag_path: Path
    stdout_path: Path
    stderr_path: Path
    active_path: Path
    process: subprocess.Popen | None
    dry_run: bool = False


class RosbagRecorder:
    def __init__(self, config: RecordingConfig):
        self.config = config

    def start(
        self,
        episode_id: str,
        topics: list[str],
        *,
        dry_run: bool | None = None,
    ) -> RecorderHandle:
        dry = self.config.dry_run_recorder if dry_run is None else dry_run
        bags = self.config.bags_dir(episode_id)
        bag_path = bags / "original.bag"
        stdout_path = self.config.session_dir(episode_id) / "record.stdout.log"
        stderr_path = self.config.session_dir(episode_id) / "record.stderr.log"
        active_path = self.config.root_dir / ".active"

        if dry:
            # Fake recorder: write growing bag file via a short Python child.
            cmd = [
                "python3",
                "-c",
                (
                    "import time,sys;"
                    f"p={str(bag_path)!r};"
                    "open(p,'ab').write(b'BAG');"
                    "[(open(p,'ab').write(b'x'*1024), time.sleep(0.2)) for _ in range(1000)]"
                ),
            ]
        else:
            # rosbag record -O <path_without_ext> <topics...>
            out_base = str(bag_path.with_suffix(""))
            cmd = list(self.config.rosbag_cmd) + [out_base] + list(topics)

        stdout_fp = open(stdout_path, "a", encoding="utf-8")
        stderr_fp = open(stderr_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=stdout_fp,
            stderr=stderr_fp,
            start_new_session=True,
        )
        active_path.write_text(episode_id, encoding="utf-8")
        return RecorderHandle(
            episode_id=episode_id,
            pid=int(proc.pid),
            command=cmd,
            bag_path=bag_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            active_path=active_path,
            process=proc,
            dry_run=dry,
        )

    def is_alive(self, handle: RecorderHandle) -> bool:
        if handle.process is not None:
            return handle.process.poll() is None
        return _pid_alive(handle.pid)

    def stop(self, handle: RecorderHandle, *, timeout_s: float | None = None) -> int:
        timeout = (
            self.config.recorder_stop_timeout_s if timeout_s is None else timeout_s
        )
        proc = handle.process
        if proc is None:
            if _pid_alive(handle.pid):
                os.kill(handle.pid, signal.SIGINT)
                return _wait_pid(handle.pid, timeout)
            self._clear_active(handle)
            return 0

        if proc.poll() is not None:
            self._clear_active(handle)
            return int(proc.returncode or 0)

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        except ProcessLookupError:
            pass

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._clear_active(handle)
                return int(proc.returncode or 0)
            time.sleep(0.05)

        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=1.0)
        self._clear_active(handle)
        return int(proc.returncode if proc.returncode is not None else -9)

    def kill_orphan(self, pid: int, command_hint: str | None = None) -> bool:
        if pid <= 0 or not _pid_alive(pid):
            return False
        cmdline = _read_cmdline(pid)
        if command_hint and command_hint not in cmdline and "rosbag" not in cmdline and "python" not in cmdline:
            return False
        try:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.2)
            if _pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return False

    def _clear_active(self, handle: RecorderHandle) -> None:
        try:
            if handle.active_path.exists():
                content = handle.active_path.read_text(encoding="utf-8").strip()
                if content == handle.episode_id:
                    handle.active_path.unlink(missing_ok=True)
        except OSError:
            pass


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_pid(pid: int, timeout_s: float) -> int:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return 0
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return -9


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""
