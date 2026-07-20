"""HIL recording configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kuavo_rl.hil_recording.models import REPLAY_SCHEMA_VERSION, SCHEMA_VERSION, TOPICS_VERSION


@dataclass
class RecordingConfig:
    root_dir: Path
    db_name: str = "hil_recording.db"
    topics_version: str = TOPICS_VERSION
    schema_version: str = SCHEMA_VERSION
    replay_schema_version: str = REPLAY_SCHEMA_VERSION
    start_block_disk_percent: float = 90.0
    hard_stop_disk_percent: float = 95.0
    post_roll_s: float = 0.5
    bag_stall_timeout_s: float = 5.0
    watchdog_loop_s: float = 0.5
    bag_stat_interval_s: float = 1.0
    disk_check_interval_s: float = 5.0
    recorder_stop_timeout_s: float = 10.0
    finalize_timeout_s: float = 60.0
    allow_degraded_export: bool = False
    # When True, gate skips live ROS probes (unit tests / mock).
    skip_gate_ros: bool = False
    # When True, use fake recorder subprocess (sleep) instead of rosbag.
    dry_run_recorder: bool = False
    rosbag_cmd: list[str] = field(default_factory=lambda: ["rosbag", "record", "-O"])
    control_topics_for_external_check: list[str] = field(
        default_factory=lambda: ["/joint_cmd", "/kuavo_arm_traj"]
    )

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def db_path(self) -> Path:
        return self.root_dir / self.db_name

    @property
    def sessions_dir(self) -> Path:
        d = self.root_dir / "sessions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def accepted_replay_dir(self) -> Path:
        d = self.root_dir / "accepted_replay"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def quarantine_dir(self) -> Path:
        d = self.root_dir / "quarantine"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def pending_review_dir(self) -> Path:
        d = self.root_dir / "pending_review"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def session_dir(self, episode_id: str) -> Path:
        d = self.sessions_dir / episode_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def staging_dir(self, episode_id: str) -> Path:
        d = self.session_dir(episode_id) / "staging"
        d.mkdir(parents=True, exist_ok=True)
        (d / "frames").mkdir(exist_ok=True)
        return d

    def bags_dir(self, episode_id: str) -> Path:
        d = self.session_dir(episode_id) / "bags"
        d.mkdir(parents=True, exist_ok=True)
        return d
