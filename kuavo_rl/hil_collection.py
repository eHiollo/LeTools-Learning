"""Local HIL dataset collection orchestration (C0–C3 skeleton + C1 Quest gate)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kuavo_rl.hil_recording.config import RecordingConfig
from kuavo_rl.hil_recording.database import HILDatabase
from kuavo_rl.hil_recording.gate import RecordGate
from kuavo_rl.hil_recording.models import (
    EPISODE_CONTROL_EVENTS,
    EXPORT_PENDING_REVIEW,
    PHASE_COLLECTION_ENDED,
    PHASE_FINALIZING,
    PHASE_RECORDING,
    PHASE_RESETTING,
    REVIEW_READY_MARKER,
    TRAIN_READY_MARKER,
    EpisodeControlEvent,
    EpisodeLabel,
    ProducerInfo,
    RecordRequest,
    RecoveryReport,
    ResultEvent,
    STATE_FINALIZED_HEALTHY,
)
from kuavo_rl.hil_recording.publish_replay import (
    publish_accepted,
    publish_pending_review,
    quarantine_episode,
)
from kuavo_rl.hil_recording.session import HILRecordingSession
from kuavo_rl.hil_recording.timebase import now_stamps
from kuavo_rl.hil_recording.topics import resolve_topics
from kuavo_rl.quest_episode_control import (
    DEFAULT_OVERRIDE_PATH,
    StickAxisCalibration,
    StickEdgeDetector,
    load_stick_calibration,
    save_stick_calibration,
    verify_right_stick_exclusive,
)

# Printed in preflight; keep in sync with ModifierStickDetector.
EPISODE_CONTROL_OPERATOR_CARD = {
    "mode": "quest_y_stick",
    "mapping": {
        "Y": "left_second_button_pressed (hold = collection modifier)",
        "right_stick": "only while Y held",
        "B": "right_second_button_pressed (labels only)",
    },
    "actions": [
        {
            "chord": "Hold Y + stick →",
            "RESETTING": "start recording",
            "RECORDING": "ignored — must end with B so every episode has a label",
        },
        {
            "chord": "Hold Y + stick ←",
            "RECORDING": "rerecord → quarantine (keep evidence) → reset",
        },
        {
            "chord": "Hold Y + stick ↓",
            "RESETTING": "end collection session",
            "RECORDING": "ignored — press B to finish episode first",
        },
        {
            "chord": "Y released",
            "any": "stick returns to waist/chassis teleop (not consumed here)",
        },
        {
            "chord": "left dual buttons (existing)",
            "any": "estop (unchanged)",
        },
        {
            "chord": "B click / double / long",
            "RECORDING": (
                "only way to keep an episode (auto_accept): "
                "click=success→accepted_replay; double=failure→accepted_replay; "
                "hold≥1.2s=abort→quarantine; no step/time timeout"
            ),
        },
    ],
    "conflicts_avoided": [
        "X + A teleop activate",
        "X + stick waist master (different modifier)",
        "stick without Y → waist/chassis unchanged",
        "triggers / grips / B labels unchanged",
    ],
    "note": "Solo 默认 auto_accept：B 即标签+可训练。录制无 step 截止。",
}


def _sha256_path(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd or Path.cwd()),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def make_episode_id(task_id: str, *, when: datetime | None = None) -> str:
    ts = (when or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%S")
    safe_task = "".join(c if c.isalnum() or c in "-_" else "_" for c in task_id)[:48]
    return f"{ts}_{safe_task}_{uuid.uuid4().hex[:8]}"


def load_collection_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML required") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"invalid collection config: {path}")
    return data


@dataclass
class CollectionConfig:
    task_id: str = "unnamed_task"
    task_text: str = ""
    root: Path = field(default_factory=lambda: Path("data/rl_runs/hilserl_episodes/hilserl_vr"))
    mode: str = "act_vr"
    default_max_steps: int = 300
    default_max_duration_s: float = 90.0
    reset_time_s: float = 60.0
    batch_stop_on_quarantine: bool = True
    # Default: hold Y + right stick. Alt: quest_y_chord / quest_right_stick.
    episode_control: str = "quest_y_stick"
    right_stick_trigger_threshold: float = 0.80
    right_stick_rearm_neutral_threshold: float = 0.20
    right_stick_debounce_s: float = 0.25
    right_stick_exclusive: bool = False
    require_collection_mode_ack: bool = False
    chord_long_press_s: float = 1.0
    auto_start_after_reset: bool = False
    # Solo / local collect: B end → auto label+review → accepted_replay/TRAIN_READY
    auto_accept: bool = True
    # After accept: run KuavoBrain CvtRosbag2Lerobot (needs live bag, not dry_run)
    auto_export_lerobot: bool = True
    lerobot_topic_profile: str = "sim"  # sim | brain
    robot_type: str = "Kuavo"
    eef_type: str = "leju_claw"
    robot_version: str = "unknown"
    lower_commit: str = "unknown"
    scene_id: str = "unset"
    task_variant: str = "default"
    topics_profile: Path = field(
        default_factory=lambda: Path("configs/rl/hil_topics_v002.yaml")
    )
    live_rosbag: bool = False
    post_roll_s: float = 0.5
    start_block_disk_percent: float = 90.0
    hard_stop_disk_percent: float = 95.0
    allow_degraded_export: bool = False
    shadow_mode: bool = False
    ros_teleop: bool = True
    skip_gate_ros: bool = True  # sim/C0 default: no ROS
    dry_run_recorder: bool = True
    stick_calibration_path: Path = field(default_factory=lambda: DEFAULT_OVERRIDE_PATH)
    require_stick_calibration_for_live: bool = True
    checkpoint: str | None = None
    deploy_config: Path = field(
        default_factory=lambda: Path("configs/deploy/total/deploy_sim_mujoco_native_cams.yaml")
    )
    env_config: Path = field(
        default_factory=lambda: Path("configs/rl/kuavo_hilserl_sim.yaml")
    )
    config_path: Path | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, config_path: Path | None = None) -> "CollectionConfig":
        col = raw.get("collection") or {}
        meta = raw.get("metadata") or {}
        rec = raw.get("recording") or {}
        runner = raw.get("runner") or {}
        stick_path = col.get("stick_calibration_path") or str(DEFAULT_OVERRIDE_PATH)
        return cls(
            task_id=str(col.get("task_id", "unnamed_task")),
            task_text=str(col.get("task_text", "")),
            root=Path(col.get("root", "data/rl_runs/hilserl_episodes/hilserl_vr")),
            mode=str(col.get("mode", "act_vr")),
            default_max_steps=int(col.get("default_max_steps", 300)),
            default_max_duration_s=float(col.get("default_max_duration_s", 90)),
            reset_time_s=float(col.get("reset_time_s", 60)),
            batch_stop_on_quarantine=bool(col.get("batch_stop_on_quarantine", True)),
            episode_control=str(col.get("episode_control", "quest_y_stick")),
            right_stick_trigger_threshold=float(
                col.get("right_stick_trigger_threshold", 0.80)
            ),
            right_stick_rearm_neutral_threshold=float(
                col.get("right_stick_rearm_neutral_threshold", 0.20)
            ),
            right_stick_debounce_s=float(col.get("right_stick_debounce_s", 0.25)),
            right_stick_exclusive=bool(col.get("right_stick_exclusive", False)),
            require_collection_mode_ack=bool(col.get("require_collection_mode_ack", False)),
            chord_long_press_s=float(col.get("chord_long_press_s", 1.0)),
            auto_start_after_reset=bool(col.get("auto_start_after_reset", False)),
            auto_accept=bool(col.get("auto_accept", True)),
            auto_export_lerobot=bool(col.get("auto_export_lerobot", True)),
            lerobot_topic_profile=str(col.get("lerobot_topic_profile", "sim")),
            robot_type=str(meta.get("robot_type", "Kuavo")),
            eef_type=str(meta.get("eef_type", "leju_claw")),
            robot_version=str(meta.get("robot_version", "unknown")),
            lower_commit=str(meta.get("lower_commit", "unknown")),
            scene_id=str(meta.get("scene_id", "unset")),
            task_variant=str(meta.get("task_variant", "default")),
            topics_profile=Path(rec.get("topics_profile", "configs/rl/hil_topics_v002.yaml")),
            live_rosbag=bool(rec.get("live_rosbag", False)),
            post_roll_s=float(rec.get("post_roll_s", 0.5)),
            start_block_disk_percent=float(rec.get("start_block_disk_percent", 90)),
            hard_stop_disk_percent=float(rec.get("hard_stop_disk_percent", 95)),
            allow_degraded_export=bool(rec.get("allow_degraded_export", False)),
            shadow_mode=bool(runner.get("shadow_mode", False)),
            ros_teleop=bool(runner.get("ros_teleop", True)),
            skip_gate_ros=bool(rec.get("skip_gate_ros", True)),
            dry_run_recorder=bool(rec.get("dry_run_recorder", not rec.get("live_rosbag", False))),
            stick_calibration_path=Path(stick_path),
            require_stick_calibration_for_live=bool(
                col.get("require_stick_calibration_for_live", True)
            ),
            checkpoint=str(runner["checkpoint"]) if runner.get("checkpoint") else None,
            deploy_config=Path(
                runner.get(
                    "deploy_config",
                    "configs/deploy/total/deploy_sim_mujoco_native_cams.yaml",
                )
            ),
            env_config=Path(runner.get("env_config", "configs/rl/kuavo_hilserl_sim.yaml")),
            config_path=config_path,
        )

    def to_recording_config(self) -> RecordingConfig:
        return RecordingConfig(
            root_dir=self.root,
            post_roll_s=self.post_roll_s,
            start_block_disk_percent=self.start_block_disk_percent,
            hard_stop_disk_percent=self.hard_stop_disk_percent,
            allow_degraded_export=self.allow_degraded_export,
            skip_gate_ros=self.skip_gate_ros,
            dry_run_recorder=self.dry_run_recorder,
            bag_stall_timeout_s=30.0 if self.live_rosbag else 5.0,
        )


@dataclass
class CollectionPhaseState:
    phase: str = PHASE_RESETTING
    episode_id: str | None = None
    last_event: str | None = None

    def apply(self, event: EpisodeControlEvent) -> str:
        """Apply a mockable episode-control event; return new phase."""
        if event.event_type not in EPISODE_CONTROL_EVENTS:
            raise ValueError(f"unknown EpisodeControlEvent: {event.event_type}")
        self.last_event = event.event_type
        if event.event_type == "ctrl_c_abort":
            if self.phase == PHASE_RECORDING:
                self.phase = PHASE_FINALIZING
            else:
                self.phase = PHASE_COLLECTION_ENDED
            return self.phase

        if self.phase == PHASE_RESETTING:
            if event.event_type == "right_stick_right":
                self.phase = PHASE_RECORDING
            elif event.event_type == "right_stick_down":
                self.phase = PHASE_COLLECTION_ENDED
            return self.phase

        if self.phase == PHASE_RECORDING:
            if event.event_type == "right_stick_left":
                self.phase = PHASE_FINALIZING  # → quarantine rerecord → RESETTING
            elif event.event_type == "right_stick_right":
                self.phase = PHASE_FINALIZING  # early_end → pending_review → RESETTING
            elif event.event_type == "right_stick_down":
                self.phase = PHASE_FINALIZING  # collection_complete → end
            elif event.event_type == "timeout":
                self.phase = PHASE_FINALIZING
            return self.phase

        return self.phase


class CollectionIndex:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "collection_manifest.json"
        self.events_path = self.root / "collection_events.jsonl"
        self.index_path = self.root / "collection_index.json"

    def start_manifest(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self.append_event("collection_started", payload)
        return payload

    def append_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        row = {
            "event_type": event_type,
            "wall_time_ns": now_stamps().wall_time_ns,
            "payload": payload or {},
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rebuild_index(self, db: HILDatabase) -> dict[str, Any]:
        episodes = []
        for row in db.list_sessions():
            eid = row["episode_id"]
            label = db.get_label(eid)
            episodes.append(
                {
                    "episode_id": eid,
                    "task_id": row["task_id"],
                    "session_state": row["session_state"],
                    "replay_export_status": row["replay_export_status"],
                    "quality_status": row["quality_status"],
                    "result_type": row.get("result_type"),
                    "stop_reason": row.get("stop_reason"),
                    "operator_label_hint": row.get("operator_label_hint"),
                    "label": label.to_dict() if label else None,
                    "session_dir": row["session_dir"],
                    "pending_review": str(self.root / "pending_review" / eid),
                    "accepted_replay": str(self.root / "accepted_replay" / eid),
                    "quarantine": str(self.root / "quarantine" / eid),
                }
            )
        index = {
            "root": str(self.root),
            "rebuilt_at_wall_ns": now_stamps().wall_time_ns,
            "episodes": episodes,
        }
        self.index_path.write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return index


class HILCollectionOrchestrator:
    """Orchestrator: preflight / recover / inspect / label / dry-run collect."""

    def __init__(self, config: CollectionConfig):
        self.config = config
        self.recording = config.to_recording_config()
        self.db = HILDatabase(self.recording.db_path)
        self.session = HILRecordingSession(
            self.recording,
            db=self.db,
            profile_path=config.topics_profile if config.topics_profile.exists() else None,
        )
        self.gate = RecordGate(self.recording, self.db)
        self.index = CollectionIndex(config.root)
        self.phase = CollectionPhaseState()
        self.stick_calibration = load_stick_calibration(config.stick_calibration_path)
        self.stick = StickEdgeDetector(
            trigger_threshold=config.right_stick_trigger_threshold,
            rearm_neutral_threshold=config.right_stick_rearm_neutral_threshold,
            debounce_s=config.right_stick_debounce_s,
            calibration=self.stick_calibration,
        )

    def close(self) -> None:
        self.session.close()

    def recover(self) -> RecoveryReport:
        report = self.session.recover_interrupted()
        self.index.append_event("recover", report.to_dict())
        self.index.rebuild_index(self.db)
        return report

    def preflight(
        self,
        *,
        task_id: str | None = None,
        producers: list[ProducerInfo] | None = None,
        for_live_collect: bool = False,
        exclusive_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """No ACT load, no rosbag, no control publish."""
        self.recover()
        task = task_id or self.config.task_id
        resolved = resolve_topics(
            control_profile=self.config.mode if self.config.mode != "shadow" else "act",
            robot_type=self.config.robot_type,
            eef_type=self.config.eef_type,
            profile_path=self.config.topics_profile
            if self.config.topics_profile.exists()
            else None,
        )
        # Temp dir for gate write probe (not a real episode)
        probe_dir = self.recording.sessions_dir / "_preflight_probe"
        probe_dir.mkdir(parents=True, exist_ok=True)
        report = self.gate.evaluate(
            resolved=resolved,
            session_dir=probe_dir,
            producers=producers or [],
            episode_id=None,
            skip_ros=self.config.skip_gate_ros,
        )
        # Reload calibration each preflight (operator may have just written override).
        self.stick_calibration = load_stick_calibration(self.config.stick_calibration_path)
        self.stick.calibration = self.stick_calibration
        # Only bare stick mode needs exclusive ack; Y+stick / Y-chord gate stick.
        uses_bare_stick = self.config.episode_control == "quest_right_stick"
        exclusive = None
        if uses_bare_stick and (self.config.right_stick_exclusive or for_live_collect):
            excl_kwargs: dict[str, Any] = {
                "calibration": self.stick_calibration,
                "require_ack": bool(
                    self.config.require_collection_mode_ack and for_live_collect
                ),
                "require_calibration": bool(
                    self.config.require_stick_calibration_for_live and for_live_collect
                ),
                "for_live_collect": for_live_collect,
            }
            if exclusive_overrides:
                excl_kwargs.update(exclusive_overrides)
            exclusive = verify_right_stick_exclusive(**excl_kwargs)

        status = report.status
        if for_live_collect and exclusive is not None and exclusive.status == "Block":
            status = "Block"

        card = dict(EPISODE_CONTROL_OPERATOR_CARD)
        card["mode"] = self.config.episode_control
        stick_skip_reason = self.config.episode_control
        out = {
            "status": status,
            "task_id": task,
            "mode": self.config.mode,
            "root": str(self.config.root),
            "topics_profile": str(self.config.topics_profile),
            "resolved_topics": json.loads(resolved.to_json()),
            "gate": report.to_dict(),
            "episode_control": self.config.episode_control,
            "episode_control_card": card,
            "stick_exclusive": exclusive.to_dict()
            if exclusive
            else {"status": "Skipped", "reason": stick_skip_reason},
            "stick_calibration": self.stick_calibration.to_dict(),
            "stick_calibration_path": str(self.config.stick_calibration_path),
            "skip_gate_ros": self.config.skip_gate_ros,
            "live_rosbag": self.config.live_rosbag,
            "for_live_collect": for_live_collect,
            "quest_event_source_publishes_actions": False,
        }
        self.index.append_event("preflight", out)
        return out

    def ensure_collection_manifest(
        self,
        *,
        collection_id: str,
        operator: str,
        scene_id: str | None = None,
    ) -> dict[str, Any]:
        stamps = now_stamps()
        cfg_path = self.config.config_path
        payload = {
            "collection_id": collection_id,
            "task_id": self.config.task_id,
            "task_text": self.config.task_text,
            "operator": operator,
            "scene_id": scene_id or self.config.scene_id,
            "config_path": str(cfg_path) if cfg_path else None,
            "config_sha256": _sha256_path(cfg_path) if cfg_path else None,
            "topics_profile_sha256": _sha256_path(self.config.topics_profile),
            "git_head": _git_head(),
            "started_at_wall_ns": stamps.wall_time_ns,
            "mode": self.config.mode,
        }
        return self.index.start_manifest(payload)

    def inspect(
        self,
        *,
        episode_id: str | None = None,
        pending_review: bool = False,
    ) -> dict[str, Any]:
        if episode_id:
            snap = self.db.snapshot(episode_id)
            label = self.db.get_label(episode_id)
            return {
                "session": snap.to_dict(),
                "label": label.to_dict() if label else None,
                "label_events": self.db.list_label_events(episode_id),
                "paths": {
                    "session": snap.session_dir,
                    "pending_review": str(self.recording.pending_review_dir / episode_id),
                    "accepted": str(self.recording.accepted_replay_dir / episode_id),
                    "quarantine": str(self.recording.quarantine_dir / episode_id),
                },
            }
        if pending_review:
            rows = self.db.list_by_export_status(EXPORT_PENDING_REVIEW)
            items = []
            for row in rows:
                eid = row["episode_id"]
                ready = (self.recording.pending_review_dir / eid / REVIEW_READY_MARKER).exists()
                items.append(
                    {
                        "episode_id": eid,
                        "task_id": row["task_id"],
                        "quality_status": row["quality_status"],
                        "review_ready": ready,
                        "label": (
                            self.db.get_label(eid).to_dict()
                            if self.db.get_label(eid)
                            else None
                        ),
                    }
                )
            return {"pending_review": items}
        return self.index.rebuild_index(self.db)

    def set_operator_hint(
        self,
        episode_id: str,
        hint: str,
        *,
        stop_reason: str | None = None,
        actor: str = "collector",
    ) -> EpisodeLabel:
        label = self.db.get_label(episode_id) or EpisodeLabel(episode_id=episode_id)
        label.operator_label_hint = hint
        if stop_reason is not None:
            label.stop_reason = stop_reason
        self.db.upsert_label(label, actor=actor, event_type="hint_updated")
        self.index.append_event(
            "label_hint",
            {"episode_id": episode_id, "hint": hint, "stop_reason": stop_reason},
        )
        return label

    def label_episode(
        self,
        episode_id: str,
        *,
        final_label: str,
        reason: str | None = None,
        labeler: str = "anonymous",
        label_version: str = "v1",
    ) -> EpisodeLabel:
        label = self.db.get_label(episode_id) or EpisodeLabel(episode_id=episode_id)
        label.label_status = "labeled"
        label.final_label = final_label
        label.failure_reason = reason
        label.labeler = labeler
        label.label_version = label_version
        label.labeled_at_wall_ns = now_stamps().wall_time_ns
        self.db.upsert_label(label, actor=labeler, event_type="label_created")
        self.index.append_event(
            "label_created",
            {"episode_id": episode_id, "final_label": final_label, "reason": reason},
        )
        return label

    def review_episode(
        self,
        episode_id: str,
        *,
        approve: bool,
        reviewer: str = "anonymous",
        reason: str | None = None,
        require_distinct_reviewer: bool = False,
    ) -> EpisodeLabel:
        label = self.db.get_label(episode_id)
        if label is None or label.label_status not in {"labeled", "reviewed"}:
            raise RuntimeError("episode must be labeled before review")
        if require_distinct_reviewer and label.labeler and label.labeler == reviewer:
            raise RuntimeError("reviewer must differ from labeler for this dataset")
        if approve:
            label.label_status = "reviewed"
            label.reviewer = reviewer
            label.reviewed_at_wall_ns = now_stamps().wall_time_ns
            self.db.upsert_label(label, actor=reviewer, event_type="label_reviewed")
            self.index.append_event(
                "label_reviewed", {"episode_id": episode_id, "reviewer": reviewer}
            )
        else:
            label.label_status = "rejected"
            label.reviewer = reviewer
            label.reviewed_at_wall_ns = now_stamps().wall_time_ns
            label.failure_reason = reason or label.failure_reason
            self.db.upsert_label(label, actor=reviewer, event_type="label_rejected")
            quarantine_episode(
                self.recording, self.db, episode_id, reason=reason or "review_rejected"
            )
            self.index.append_event(
                "label_rejected",
                {"episode_id": episode_id, "reason": reason, "reviewer": reviewer},
            )
        self.index.rebuild_index(self.db)
        return label

    def publish_pending(self, episode_id: str) -> dict[str, Any]:
        report = publish_pending_review(self.recording, self.db, episode_id)
        self.index.append_event("export_transition", report.to_dict())
        self.index.rebuild_index(self.db)
        return report.to_dict()

    def publish_train_ready(self, episode_id: str) -> dict[str, Any]:
        report = publish_accepted(self.recording, self.db, episode_id)
        self.index.append_event("export_transition", report.to_dict())
        self.index.rebuild_index(self.db)
        return report.to_dict()

    def accept_after_collect(
        self,
        episode_id: str,
        *,
        operator: str,
        stop_reason: str,
    ) -> dict[str, Any]:
        """Solo path: B label → pending → reviewed → accepted_replay/TRAIN_READY.

        ``abort`` / ``unsafe`` / ``invalid`` still land in quarantine (not train-ready).
        """
        final_by_stop = {
            "success_button": "success",
            "failure_button": "failure",
            "abort": "abort",
            "estop": "abort",
        }
        final = final_by_stop.get(stop_reason)
        if final is None:
            q = quarantine_episode(
                self.recording,
                self.db,
                episode_id,
                reason=f"auto_accept_missing_label:{stop_reason}",
            )
            self.index.append_event("export_transition", q.to_dict())
            self.index.rebuild_index(self.db)
            return {"status": "quarantined", **q.to_dict()}

        self.label_episode(
            episode_id,
            final_label=final,
            reason=f"auto_accept:{stop_reason}",
            labeler=operator,
        )
        pending = self.publish_pending(episode_id)
        if pending.get("status") == "Quarantined":
            return pending

        self.review_episode(
            episode_id,
            approve=True,
            reviewer=operator,
            require_distinct_reviewer=False,
        )
        if final in {"abort", "unsafe", "invalid"}:
            # publish_accepted would quarantine; keep explicit reason
            q = quarantine_episode(
                self.recording, self.db, episode_id, reason=f"final_label={final}"
            )
            self.index.append_event("export_transition", q.to_dict())
            self.index.rebuild_index(self.db)
            return {"status": "quarantined", "final_label": final, **q.to_dict()}

        accepted = self.publish_train_ready(episode_id)
        return {"status": accepted.get("status"), "final_label": final, **accepted}

    def apply_control_event(self, event: EpisodeControlEvent) -> CollectionPhaseState:
        self.phase.apply(event)
        self.index.append_event(
            "episode_control",
            {"phase": self.phase.phase, "event": event.to_dict()},
        )
        return self.phase

    def collection_report(self) -> dict[str, Any]:
        """C3: aggregate by final_label / failure_reason / intervention hint."""
        by_final: dict[str, int] = {}
        by_failure: dict[str, int] = {}
        by_hint: dict[str, int] = {}
        by_export: dict[str, int] = {}
        pending_n = 0
        train_ready_n = 0
        for row in self.db.list_sessions():
            eid = row["episode_id"]
            export = row.get("replay_export_status") or "NotStarted"
            by_export[export] = by_export.get(export, 0) + 1
            label = self.db.get_label(eid)
            if label is None:
                continue
            fl = label.final_label or "(unlabeled)"
            by_final[fl] = by_final.get(fl, 0) + 1
            if label.failure_reason:
                by_failure[label.failure_reason] = by_failure.get(label.failure_reason, 0) + 1
            hint = label.operator_label_hint or "unknown"
            by_hint[hint] = by_hint.get(hint, 0) + 1
            if (self.recording.pending_review_dir / eid / REVIEW_READY_MARKER).exists():
                pending_n += 1
            if (self.recording.accepted_replay_dir / eid / TRAIN_READY_MARKER).exists():
                train_ready_n += 1
        report = {
            "root": str(self.config.root),
            "by_final_label": by_final,
            "by_failure_reason": by_failure,
            "by_operator_hint": by_hint,
            "by_export_status": by_export,
            "pending_review_ready": pending_n,
            "train_ready": train_ready_n,
        }
        self.index.append_event("collection_report", report)
        return report

    def collect_episode_dry_run(
        self,
        *,
        episode_id: str | None = None,
        operator: str = "dry_run",
        scene_id: str | None = None,
        max_steps: int = 8,
        end_event: str = "right_stick_right",
        operator_label_hint: str = "unknown",
        stop_reason: str | None = None,
    ) -> dict[str, Any]:
        """C2 dry-run: session + staging transitions, no ACT/env/ROS action publish.

        Simulates RECORDING with synthetic transitions, then finalize → pending_review
        (or quarantine on rerecord). Never loads checkpoint or publishes robot cmds.
        """
        import os

        import numpy as np

        from kuavo_rl.recording import HILReplayWriter, TransitionRecord

        eid = episode_id or make_episode_id(self.config.task_id)
        stop_reason = stop_reason or {
            "right_stick_left": "rerecord",
            "right_stick_right": "early_end",
            "right_stick_down": "collection_complete",
            "ctrl_c_abort": "abort",
        }.get(end_event, "early_end")

        pf = self.preflight(for_live_collect=False)
        if pf["status"] == "Block":
            return {"status": "Block", "preflight": pf, "episode_id": eid}

        metadata = {
            "operator": operator,
            "scene_id": scene_id or self.config.scene_id,
            "task_variant": self.config.task_variant,
            "mode": self.config.mode,
            "collection": "dry_run",
            "checkpoint": self.config.checkpoint,
            "git_head": _git_head(),
        }
        profile = self.config.mode if self.config.mode != "shadow" else "act"
        req = RecordRequest(
            episode_id=eid,
            task_id=self.config.task_id,
            control_profile=profile if profile != "act_vr" else "act",
            robot_type=self.config.robot_type,
            robot_version=self.config.robot_version,
            eef_type=self.config.eef_type,
            lower_commit=self.config.lower_commit,
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=min(0.05, float(self.config.post_roll_s)),
            metadata=metadata,
        )
        self.session.create(req)
        self.session.register_producer("act_dry_run", os.getpid(), "policy")
        self.session.start(eid)
        self.index.append_event("episode_started", {"episode_id": eid, "dry_run": True})
        self.phase.phase = PHASE_RECORDING
        self.phase.episode_id = eid

        staging = self.recording.staging_dir(eid)
        writer = HILReplayWriter(self.config.root, "dry_run", staging_dir=staging)
        for i in range(max_steps):
            stamps = now_stamps()
            rec = TransitionRecord(
                experiment_id="dry_run",
                episode_id=eid,
                step_id=i,
                timestamp=float(i),
                action=[0.0] * 16,
                reward=0.0,
                reward_source="none",
                terminated=i == max_steps - 1,
                truncated=False,
                fault_code="NONE",
                is_intervention=False,
                action_clipped=False,
                extras={
                    "intervention_mask": [0] * 16,
                    "intervention_segment_step": 0,
                    "stamps": stamps.to_dict(),
                    "dry_run": True,
                },
            )
            obs = {"observation.state": np.zeros(16, dtype=np.float32)}
            writer.log_transition(rec, observation=obs, next_observation=obs)
            self.session.update_transition(
                {
                    "step_id": i,
                    "stamps": stamps,
                    "intervention_mask": [0] * 16,
                    "intervention_segment_step": 0,
                    "policy_action": [0.0] * 16,
                    "reward": 0.0,
                    "fault_code": "NONE",
                }
            )
        writer.close()

        ctrl = EpisodeControlEvent(end_event, "dry_run", now_stamps())
        self.apply_control_event(ctrl)
        self.set_operator_hint(eid, operator_label_hint, stop_reason=stop_reason)

        result_type = "abort" if end_event in {"ctrl_c_abort", "right_stick_left"} else "success"
        self.session.record_event(
            ResultEvent(
                episode_id=eid,
                event_type=result_type,
                source="dry_run",
                stamps=now_stamps(),
                payload={"end_event": end_event, "stop_reason": stop_reason},
            )
        )
        self.session.request_stop(eid, stop_reason)
        snap = self.session.wait_finalized(eid, timeout_s=30.0)

        if end_event == "right_stick_left":
            q = quarantine_episode(self.recording, self.db, eid, reason="rerecord")
            self.index.append_event("export_transition", q.to_dict())
            self.index.rebuild_index(self.db)
            return {
                "status": "quarantined_rerecord",
                "episode_id": eid,
                "session_state": snap.session_state,
                "stop_reason": stop_reason,
                "export": q.to_dict(),
                "accepted_blocked": not (self.recording.accepted_replay_dir / eid).exists(),
            }

        if snap.session_state != STATE_FINALIZED_HEALTHY and not self.config.allow_degraded_export:
            q = quarantine_episode(
                self.recording, self.db, eid, reason=f"quality:{snap.quality_status}"
            )
            self.index.append_event("export_transition", q.to_dict())
            return {
                "status": "quarantined_quality",
                "episode_id": eid,
                "session_state": snap.session_state,
                "export": q.to_dict(),
            }

        pub = self.publish_pending(eid)
        pending = self.recording.pending_review_dir / eid
        return {
            "status": "pending_review",
            "episode_id": eid,
            "session_state": snap.session_state,
            "stop_reason": stop_reason,
            "export": pub,
            "pending_review": str(pending),
            "review_ready": (pending / REVIEW_READY_MARKER).exists(),
            "accepted_blocked": not (self.recording.accepted_replay_dir / eid).exists(),
            "train_ready_blocked": not (
                self.recording.accepted_replay_dir / eid / TRAIN_READY_MARKER
            ).exists(),
        }

    def batch_dry_run(
        self,
        *,
        episodes: int,
        operator: str = "dry_run",
        scene_id: str | None = None,
        max_steps: int = 4,
        end_events: list[str] | None = None,
        collection_id: str | None = None,
    ) -> dict[str, Any]:
        """C4 dry-run batch: RESETTING → RECORDING → finalize → next (or end).

        ``end_events`` optional per-episode control events; default early_end each.
        Stops on non-rerecord quarantine when ``batch_stop_on_quarantine``.
        Rerecord quarantines do not count toward completed episodes and retry same index.
        """
        n = max(1, int(episodes))
        # Queue of end events for each *attempt* (rerecord consumes one without completing).
        event_q = list(end_events) if end_events else ["right_stick_right"] * n

        cid = collection_id or f"{datetime.now(timezone.utc).strftime('%Y%m%d')}_{self.config.task_id}"
        self.ensure_collection_manifest(
            collection_id=cid, operator=operator, scene_id=scene_id
        )
        self.phase.phase = PHASE_RESETTING
        results: list[dict[str, Any]] = []
        completed = 0
        attempts = 0
        max_attempts = max(n * 3, len(event_q) + 1)
        while completed < n and attempts < max_attempts:
            attempts += 1
            # RESETTING: operator would tip → to start; dry-run auto-starts.
            start_ev = EpisodeControlEvent("right_stick_right", "batch_dry_run", now_stamps())
            self.apply_control_event(start_ev)
            end_ev = event_q.pop(0) if event_q else "right_stick_right"
            out = self.collect_episode_dry_run(
                operator=operator,
                scene_id=scene_id,
                max_steps=max_steps,
                end_event=end_ev,
            )
            results.append(out)
            status = out.get("status")
            if status == "quarantined_rerecord":
                # Same episode index, back to RESETTING (next queue event is retry end).
                self.phase.phase = PHASE_RESETTING
                continue
            if status == "Block":
                break
            if status.startswith("quarantined") and self.config.batch_stop_on_quarantine:
                self.phase.phase = PHASE_COLLECTION_ENDED
                self.index.append_event(
                    "collection_ended",
                    {"reason": "stop_on_quarantine", "episode_id": out.get("episode_id")},
                )
                break
            completed += 1
            if end_ev == "right_stick_down":
                self.phase.phase = PHASE_COLLECTION_ENDED
                self.index.append_event(
                    "collection_ended",
                    {"reason": "collection_complete", "episode_id": out.get("episode_id")},
                )
                break
            # early_end / success path → next episode
            self.phase.phase = PHASE_RESETTING

        if self.phase.phase != PHASE_COLLECTION_ENDED:
            self.phase.phase = PHASE_COLLECTION_ENDED
            self.index.append_event(
                "collection_ended", {"reason": "batch_complete", "count": completed}
            )

        summary = {
            "status": "ok" if results and results[-1].get("status") != "Block" else "failed",
            "collection_id": cid,
            "completed_episodes": completed,
            "attempts": attempts,
            "results": results,
            "report": self.collection_report(),
        }
        self.index.rebuild_index(self.db)
        return summary

    def save_stick_calibration(self, cal: StickAxisCalibration) -> Path:
        path = save_stick_calibration(cal, self.config.stick_calibration_path)
        self.stick_calibration = cal
        self.stick.calibration = cal
        self.index.append_event(
            "stick_calibrated",
            {"path": str(path), "calibration": cal.to_dict()},
        )
        return path
