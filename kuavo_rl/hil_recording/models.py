"""Session / bag / gate / watchdog dataclasses and state constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = "hil-db-v002"
SCHEMA_VERSION_V001 = "hil-db-v001"
TOPICS_VERSION = "hil-v002"
REPLAY_SCHEMA_VERSION = "hil-replay-v002"

REVIEW_READY_MARKER = "REVIEW_READY"
TRAIN_READY_MARKER = "TRAIN_READY"


# --- session_state ---
STATE_PREPARING = "Preparing"
STATE_RECORDING = "Recording"
STATE_STOPPING = "Stopping"
STATE_FINALIZING = "Finalizing"
STATE_FINALIZED_HEALTHY = "Finalized(Healthy)"
STATE_FINALIZED_DEGRADED = "Finalized(ManuallyAcceptedDegraded)"
STATE_FAILED_RECORD = "Failed(record_error)"
STATE_FAILED_SELF_CHECK = "Failed(self_check)"
STATE_CANCELED = "Canceled"
STATE_DELETED = "Deleted"

SESSION_STATES = frozenset(
    {
        STATE_PREPARING,
        STATE_RECORDING,
        STATE_STOPPING,
        STATE_FINALIZING,
        STATE_FINALIZED_HEALTHY,
        STATE_FINALIZED_DEGRADED,
        STATE_FAILED_RECORD,
        STATE_FAILED_SELF_CHECK,
        STATE_CANCELED,
        STATE_DELETED,
    }
)

# Whitelist: from_state -> allowed next states
SESSION_TRANSITIONS: dict[str, frozenset[str]] = {
    STATE_PREPARING: frozenset({STATE_RECORDING, STATE_CANCELED, STATE_FAILED_RECORD}),
    STATE_RECORDING: frozenset({STATE_STOPPING, STATE_FAILED_RECORD}),
    STATE_STOPPING: frozenset({STATE_FINALIZING, STATE_FAILED_RECORD}),
    STATE_FINALIZING: frozenset(
        {
            STATE_FINALIZED_HEALTHY,
            STATE_FINALIZED_DEGRADED,
            STATE_FAILED_SELF_CHECK,
            STATE_FAILED_RECORD,
        }
    ),
    STATE_FINALIZED_HEALTHY: frozenset({STATE_DELETED}),
    STATE_FINALIZED_DEGRADED: frozenset({STATE_DELETED}),
    STATE_FAILED_RECORD: frozenset({STATE_DELETED}),
    STATE_FAILED_SELF_CHECK: frozenset({STATE_FINALIZED_DEGRADED, STATE_DELETED}),
    STATE_CANCELED: frozenset({STATE_DELETED}),
    STATE_DELETED: frozenset(),
}

# --- replay_export_status ---
EXPORT_NOT_STARTED = "NotStarted"
EXPORT_STAGED = "Staged"
EXPORT_PENDING_REVIEW = "PendingReview"
EXPORT_PUBLISHED = "Published"
EXPORT_QUARANTINED = "Quarantined"

EXPORT_STATES = frozenset(
    {
        EXPORT_NOT_STARTED,
        EXPORT_STAGED,
        EXPORT_PENDING_REVIEW,
        EXPORT_PUBLISHED,
        EXPORT_QUARANTINED,
    }
)

EXPORT_TRANSITIONS: dict[str, frozenset[str]] = {
    EXPORT_NOT_STARTED: frozenset(
        {EXPORT_STAGED, EXPORT_PENDING_REVIEW, EXPORT_QUARANTINED}
    ),
    EXPORT_STAGED: frozenset({EXPORT_PENDING_REVIEW, EXPORT_QUARANTINED}),
    EXPORT_PENDING_REVIEW: frozenset({EXPORT_PUBLISHED, EXPORT_QUARANTINED}),
    EXPORT_PUBLISHED: frozenset(),
    EXPORT_QUARANTINED: frozenset(),
}

FINALIZED_OK = frozenset({STATE_FINALIZED_HEALTHY, STATE_FINALIZED_DEGRADED})
ACTIVE_SESSION_STATES = frozenset(
    {STATE_RECORDING, STATE_STOPPING, STATE_FINALIZING}
)


@dataclass(frozen=True)
class TimeStamps:
    ros_time_ns: int
    monotonic_time_ns: int
    wall_time_ns: int
    source_header_stamp_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def align_key_ns(self) -> int:
        """ROS time used for cross-stream alignment."""
        if self.source_header_stamp_ns is not None:
            return int(self.source_header_stamp_ns)
        return int(self.ros_time_ns)


@dataclass
class ProducerInfo:
    name: str
    pid: int
    kind: str  # policy | teleop | other


@dataclass
class StopRequest:
    episode_id: str
    reason: str
    source: str = "controller"  # controller | watchdog | user
    stamps: TimeStamps | None = None


@dataclass(frozen=True)
class EpisodeControlEvent:
    """Collection lifecycle control — not a ResultEvent task label."""

    event_type: str
    # right_stick_left/right/down | timeout | success_candidate | failure_candidate
    source: str
    stamps: TimeStamps
    episode_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Collection outer FSM phases (operator UX); independent of session_state.
PHASE_RESETTING = "RESETTING"
PHASE_RECORDING = "RECORDING"
PHASE_FINALIZING = "FINALIZING"
PHASE_PENDING_REVIEW = "PENDING_REVIEW"
PHASE_COLLECTION_ENDED = "COLLECTION_ENDED"

EPISODE_CONTROL_EVENTS = frozenset(
    {
        "right_stick_left",
        "right_stick_right",
        "right_stick_down",
        "timeout",
        "success_candidate",
        "failure_candidate",
        "ctrl_c_abort",
    }
)

STOP_REASONS_CONTROL = frozenset(
    {"early_end", "collection_complete", "rerecord", "timeout"}
)


@dataclass
class ResultEvent:
    episode_id: str
    event_type: str  # success | failure | abort | estop | fault
    source: str
    stamps: TimeStamps
    payload: dict[str, Any] = field(default_factory=dict)


RESULT_PRIORITY = {
    "estop": 100,
    "fault": 90,
    "abort": 50,
    "failure": 30,
    "success": 10,
}


@dataclass
class RecordRequest:
    episode_id: str
    task_id: str
    control_profile: str = "act_vr"  # act | act_vr | vr_only
    robot_type: str = "Kuavo"
    eef_type: str = "leju_claw"
    robot_version: str | None = None
    lower_commit: str | None = None
    dry_run: bool = False  # skip real rosbag / ROS probes
    skip_gate_ros: bool = False  # unit tests / mock
    allow_degraded: bool = False
    post_roll_s: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionSnapshot:
    episode_id: str
    task_id: str
    session_state: str
    replay_export_status: str
    session_dir: str
    gate_status: str = "Pending"
    watchdog_status: str = "Disabled"
    quality_status: str = "Pending"
    result_type: str | None = None
    control_profile: str = "act_vr"
    record_pid: int = 0
    error_message: str | None = None
    quality_report_json: str | None = None
    resolved_topics_json: str = "[]"
    producers: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    stop_reason: str | None = None
    operator_label_hint: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeLabel:
    episode_id: str
    label_status: str = "pending"  # pending | labeled | reviewed | rejected
    operator_label_hint: str = "unknown"
    stop_reason: str | None = None
    final_label: str | None = None  # success | failure | abort | unsafe | invalid
    failure_reason: str | None = None
    labeler: str | None = None
    label_version: str | None = None
    labeled_at_wall_ns: int | None = None
    reviewer: str | None = None
    reviewed_at_wall_ns: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GateReport:
    status: str  # Pass | Block | Degraded
    checked_at_ros_ns: int
    checked_at_mono_ns: int
    missing_topics: list[str] = field(default_factory=list)
    low_rate_topics: list[str] = field(default_factory=list)
    stale_topics: list[str] = field(default_factory=list)
    latched_missing: list[str] = field(default_factory=list)
    disk_usage_percent: float = 0.0
    producers: list[dict[str, Any]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QualityReport:
    status: str  # Healthy | Degraded | Failed
    bag_readable: bool
    sidecar_step_count: int = 0
    bag_audit_step_count: int | None = None
    sidecar_bag_match: bool = True
    topic_issues: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExportReport:
    episode_id: str
    status: str  # PendingReview | Published | Quarantined | Rejected
    path: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecoveryReport:
    recovered: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    killed_pids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IllegalStateTransition(ValueError):
    pass
