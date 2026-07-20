"""staging → pending_review → accepted_replay atomic publish / quarantine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from kuavo_rl.hil_recording.config import RecordingConfig
from kuavo_rl.hil_recording.database import HILDatabase
from kuavo_rl.hil_recording.models import (
    EXPORT_NOT_STARTED,
    EXPORT_PENDING_REVIEW,
    EXPORT_PUBLISHED,
    EXPORT_QUARANTINED,
    EXPORT_STAGED,
    FINALIZED_OK,
    REPLAY_SCHEMA_VERSION,
    REVIEW_READY_MARKER,
    TRAIN_READY_MARKER,
    ExportReport,
)
from kuavo_rl.hil_recording.result_events import should_quarantine_result
from kuavo_rl.hil_recording.timebase import now_stamps


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_move_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst)
    try:
        os.rename(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))


def mark_staged(db: HILDatabase, episode_id: str) -> None:
    snap = db.snapshot(episode_id)
    if snap.replay_export_status in (
        EXPORT_STAGED,
        EXPORT_PENDING_REVIEW,
        EXPORT_PUBLISHED,
        EXPORT_QUARANTINED,
    ):
        return
    if snap.replay_export_status == EXPORT_NOT_STARTED:
        db.migrate_export_status(episode_id, EXPORT_STAGED, now_stamps())


def _staging_has_payload(staging: Path) -> bool:
    """True only if staging has replay payload (not just an empty frames/ dir)."""
    if not staging.exists():
        return False
    if (staging / "transitions.jsonl").exists():
        return True
    frames = staging / "frames"
    if frames.is_dir() and any(frames.iterdir()):
        return True
    return False


def quarantine_episode(
    config: RecordingConfig,
    db: HILDatabase,
    episode_id: str,
    *,
    reason: str,
) -> ExportReport:
    session_dir = config.session_dir(episode_id)
    staging = session_dir / "staging"
    pending = config.pending_review_dir / episode_id
    dest = config.quarantine_dir / episode_id
    dest.mkdir(parents=True, exist_ok=True)

    # Prefer moving staging; if already in pending_review, move that instead.
    # Only move when staging has real payload (transitions/frames). An empty
    # staging/frames recreated after the first move must NOT wipe quarantine/.
    if _staging_has_payload(staging):
        target = dest / "staging"
        if target.exists():
            shutil.rmtree(target)
        _atomic_move_dir(staging, target)
        config.staging_dir(episode_id)
    elif pending.exists() and not _staging_has_payload(dest / "staging"):
        target = dest / "pending_review"
        if target.exists():
            shutil.rmtree(target)
        _atomic_move_dir(pending, target)
    # else: already quarantined or nothing to move — refresh metadata only

    for name in ("quality_report.json", "gate.json", "watchdog.report.json", "session.json"):
        src = session_dir / name
        if src.exists():
            shutil.copy2(src, dest / name)
    reason_path = dest / "quarantine_reason.txt"
    if reason_path.exists():
        prev = reason_path.read_text(encoding="utf-8").strip()
        if prev and prev != reason and reason not in prev:
            reason = f"{prev}; {reason}"
    reason_path.write_text(reason + "\n", encoding="utf-8")

    snap = db.snapshot(episode_id)
    if snap.replay_export_status in (
        EXPORT_NOT_STARTED,
        EXPORT_STAGED,
        EXPORT_PENDING_REVIEW,
    ):
        db.migrate_export_status(episode_id, EXPORT_QUARANTINED, now_stamps())
    return ExportReport(
        episode_id=episode_id,
        status=EXPORT_QUARANTINED,
        path=str(dest),
        reasons=[reason],
    )


def publish_pending_review(
    config: RecordingConfig,
    db: HILDatabase,
    episode_id: str,
) -> ExportReport:
    """Move staging → pending_review with REVIEW_READY. Does NOT create TRAIN_READY."""
    snap = db.snapshot(episode_id)
    if snap.session_state not in FINALIZED_OK:
        return ExportReport(
            episode_id=episode_id,
            status="Rejected",
            reasons=[f"session_state={snap.session_state} not finalized-ok"],
        )
    if should_quarantine_result(snap.result_type):
        return quarantine_episode(
            config, db, episode_id, reason=f"result_type={snap.result_type}"
        )

    staging = config.staging_dir(episode_id)
    if not (staging / "transitions.jsonl").exists():
        return quarantine_episode(
            config, db, episode_id, reason="staging_transitions_missing"
        )

    mark_staged(db, episode_id)
    snap = db.snapshot(episode_id)
    if snap.replay_export_status == EXPORT_PENDING_REVIEW:
        dest = config.pending_review_dir / episode_id
        return ExportReport(
            episode_id=episode_id,
            status=EXPORT_PENDING_REVIEW,
            path=str(dest),
            reasons=["already_pending_review"],
        )

    dest = config.pending_review_dir / episode_id
    quality_path = config.session_dir(episode_id) / "quality_report.json"
    stamps = now_stamps()
    manifest = {
        "replay_schema_version": REPLAY_SCHEMA_VERSION,
        "export_stage": EXPORT_PENDING_REVIEW,
        "replay_source_quality_report": {
            "path": str(quality_path),
            "sha256": _sha256_file(quality_path),
        },
        "replay_source_session": episode_id,
        "published_at_wall_ns": stamps.wall_time_ns,
        "published_at_ros_ns": stamps.ros_time_ns,
        "result_type": snap.result_type,
        "session_state": snap.session_state,
        "label_status": "pending",
    }
    (staging / "publish_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (staging / REVIEW_READY_MARKER).write_text("ok\n", encoding="utf-8")
    _atomic_move_dir(staging, dest)
    config.staging_dir(episode_id)
    db.migrate_export_status(episode_id, EXPORT_PENDING_REVIEW, stamps)
    return ExportReport(
        episode_id=episode_id,
        status=EXPORT_PENDING_REVIEW,
        path=str(dest),
        reasons=[],
    )


def publish_accepted(
    config: RecordingConfig,
    db: HILDatabase,
    episode_id: str,
) -> ExportReport:
    """Move pending_review → accepted_replay with TRAIN_READY. Requires reviewed label."""
    snap = db.snapshot(episode_id)
    if snap.session_state not in FINALIZED_OK:
        return ExportReport(
            episode_id=episode_id,
            status="Rejected",
            reasons=[f"session_state={snap.session_state} not finalized-ok"],
        )
    if snap.replay_export_status != EXPORT_PENDING_REVIEW:
        return ExportReport(
            episode_id=episode_id,
            status="Rejected",
            reasons=[
                f"replay_export_status={snap.replay_export_status} "
                f"(need {EXPORT_PENDING_REVIEW})"
            ],
        )
    if should_quarantine_result(snap.result_type):
        return quarantine_episode(
            config, db, episode_id, reason=f"result_type={snap.result_type}"
        )

    label = db.get_label(episode_id)
    if label is None or label.label_status != "reviewed":
        return ExportReport(
            episode_id=episode_id,
            status="Rejected",
            reasons=["label_status must be reviewed before TRAIN_READY"],
        )
    if label.final_label in {"abort", "unsafe", "invalid"}:
        return quarantine_episode(
            config,
            db,
            episode_id,
            reason=f"final_label={label.final_label}",
        )
    if label.final_label not in {"success", "failure"}:
        return ExportReport(
            episode_id=episode_id,
            status="Rejected",
            reasons=[f"final_label={label.final_label!r} not train-eligible"],
        )

    pending = config.pending_review_dir / episode_id
    if not pending.exists() or not (pending / REVIEW_READY_MARKER).exists():
        return ExportReport(
            episode_id=episode_id,
            status="Rejected",
            reasons=["pending_review/REVIEW_READY missing"],
        )

    dest = config.accepted_replay_dir / episode_id
    stamps = now_stamps()
    (pending / TRAIN_READY_MARKER).write_text(
        json.dumps(
            {
                "final_label": label.final_label,
                "reviewer": label.reviewer,
                "reviewed_at_wall_ns": label.reviewed_at_wall_ns,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    # Update manifest stage
    man_path = pending / "publish_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        man["export_stage"] = EXPORT_PUBLISHED
        man["accepted_at_wall_ns"] = stamps.wall_time_ns
        man["final_label"] = label.final_label
        man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    _atomic_move_dir(pending, dest)
    db.migrate_export_status(episode_id, EXPORT_PUBLISHED, stamps)
    return ExportReport(
        episode_id=episode_id,
        status=EXPORT_PUBLISHED,
        path=str(dest),
        reasons=[],
    )


def publish_replay(
    config: RecordingConfig,
    db: HILDatabase,
    episode_id: str,
) -> ExportReport:
    """Backward-compatible alias → publish_pending_review (no TRAIN_READY bypass)."""
    return publish_pending_review(config, db, episode_id)
