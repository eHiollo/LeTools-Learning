"""SQLite session store: WAL, foreign keys, whitelist state migrations."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from kuavo_rl.hil_recording.models import (
    ACTIVE_SESSION_STATES,
    EXPORT_NOT_STARTED,
    EXPORT_TRANSITIONS,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V001,
    SESSION_TRANSITIONS,
    STATE_PREPARING,
    EpisodeLabel,
    IllegalStateTransition,
    SessionSnapshot,
    TimeStamps,
)

_DDL_V002 = """
CREATE TABLE IF NOT EXISTS schema_version (
  version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hil_sessions (
  episode_id TEXT PRIMARY KEY,
  task_id TEXT NOT NULL,
  session_state TEXT NOT NULL,
  replay_export_status TEXT NOT NULL DEFAULT 'NotStarted',
  created_at_wall_ns INTEGER NOT NULL,
  started_at_ros_ns INTEGER,
  stopped_at_ros_ns INTEGER,
  robot_type TEXT,
  robot_version TEXT,
  lower_commit TEXT,
  eef_type TEXT,
  control_profile TEXT NOT NULL,
  topics_version TEXT,
  resolved_topics_json TEXT NOT NULL,
  gate_status TEXT NOT NULL DEFAULT 'Pending',
  watchdog_status TEXT NOT NULL DEFAULT 'Disabled',
  quality_status TEXT NOT NULL DEFAULT 'Pending',
  result_type TEXT,
  result_event_ros_ns INTEGER,
  result_event_mono_ns INTEGER,
  result_event_source TEXT,
  session_dir TEXT NOT NULL,
  record_pid INTEGER DEFAULT 0,
  record_command TEXT,
  stdout_path TEXT,
  stderr_path TEXT,
  watchdog_report_path TEXT,
  watchdog_log_path TEXT,
  quality_report_json TEXT,
  error_message TEXT,
  producers_json TEXT NOT NULL DEFAULT '[]',
  metadata_json TEXT NOT NULL DEFAULT '{}',
  stop_reason TEXT,
  operator_label_hint TEXT NOT NULL DEFAULT 'unknown'
);

CREATE TABLE IF NOT EXISTS hil_bags (
  bag_id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL,
  bag_type TEXT NOT NULL,
  path TEXT NOT NULL,
  state TEXT NOT NULL,
  size_bytes INTEGER DEFAULT 0,
  duration_sec REAL DEFAULT 0,
  quality_report_json TEXT,
  UNIQUE (episode_id, bag_type),
  FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
);

CREATE TABLE IF NOT EXISTS hil_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  ros_time_ns INTEGER,
  monotonic_time_ns INTEGER NOT NULL,
  wall_time_ns INTEGER NOT NULL,
  source_header_stamp_ns INTEGER,
  source TEXT NOT NULL,
  payload_json TEXT,
  FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
);

CREATE TABLE IF NOT EXISTS hil_episode_labels (
  episode_id TEXT PRIMARY KEY,
  label_status TEXT NOT NULL DEFAULT 'pending',
  operator_label_hint TEXT NOT NULL DEFAULT 'unknown',
  stop_reason TEXT,
  final_label TEXT,
  failure_reason TEXT,
  labeler TEXT,
  label_version TEXT,
  labeled_at_wall_ns INTEGER,
  reviewer TEXT,
  reviewed_at_wall_ns INTEGER,
  FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
);

CREATE TABLE IF NOT EXISTS hil_label_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  actor TEXT,
  wall_time_ns INTEGER NOT NULL,
  payload_json TEXT,
  FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
);
"""


class HILDatabase:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._init_schema()

    def _configure(self) -> None:
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")

    def _init_schema(self) -> None:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        has_schema = cur.fetchone() is not None
        if not has_schema:
            self._conn.executescript(_DDL_V002)
            self._conn.execute(
                "INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,)
            )
            self._conn.commit()
            return

        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            self._conn.executescript(_DDL_V002)
            self._conn.execute(
                "INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,)
            )
            self._conn.commit()
            return

        version = row["version"]
        if version == SCHEMA_VERSION:
            # Ensure new tables exist even if partially created.
            self._conn.executescript(_DDL_V002)
            self._conn.commit()
            return
        if version == SCHEMA_VERSION_V001:
            self._migrate_v001_to_v002()
            return
        raise RuntimeError(
            f"hil_recording.db schema mismatch: found {version!r}, "
            f"expected {SCHEMA_VERSION!r} (or migratable {SCHEMA_VERSION_V001!r})"
        )

    def _migrate_v001_to_v002(self) -> None:
        """Explicit migration: add metadata/label columns and tables."""
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(hil_sessions)").fetchall()
        }
        alters: list[str] = []
        if "metadata_json" not in cols:
            alters.append(
                "ALTER TABLE hil_sessions ADD COLUMN metadata_json TEXT NOT NULL DEFAULT '{}'"
            )
        if "stop_reason" not in cols:
            alters.append("ALTER TABLE hil_sessions ADD COLUMN stop_reason TEXT")
        if "operator_label_hint" not in cols:
            alters.append(
                "ALTER TABLE hil_sessions ADD COLUMN operator_label_hint TEXT NOT NULL DEFAULT 'unknown'"
            )
        for sql in alters:
            self._conn.execute(sql)
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS hil_episode_labels (
              episode_id TEXT PRIMARY KEY,
              label_status TEXT NOT NULL DEFAULT 'pending',
              operator_label_hint TEXT NOT NULL DEFAULT 'unknown',
              stop_reason TEXT,
              final_label TEXT,
              failure_reason TEXT,
              labeler TEXT,
              label_version TEXT,
              labeled_at_wall_ns INTEGER,
              reviewer TEXT,
              reviewed_at_wall_ns INTEGER,
              FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
            );
            CREATE TABLE IF NOT EXISTS hil_label_events (
              event_id INTEGER PRIMARY KEY AUTOINCREMENT,
              episode_id TEXT NOT NULL,
              event_type TEXT NOT NULL,
              actor TEXT,
              wall_time_ns INTEGER NOT NULL,
              payload_json TEXT,
              FOREIGN KEY (episode_id) REFERENCES hil_sessions(episode_id)
            );
            """
        )
        # Seed pending labels for existing sessions.
        for row in self._conn.execute("SELECT episode_id FROM hil_sessions").fetchall():
            eid = row["episode_id"]
            exists = self._conn.execute(
                "SELECT 1 FROM hil_episode_labels WHERE episode_id=?", (eid,)
            ).fetchone()
            if exists is None:
                self._conn.execute(
                    """
                    INSERT INTO hil_episode_labels(episode_id, label_status)
                    VALUES (?, 'pending')
                    """,
                    (eid,),
                )
        self._conn.execute("DELETE FROM schema_version")
        self._conn.execute(
            "INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def insert_session(
        self,
        *,
        episode_id: str,
        task_id: str,
        control_profile: str,
        session_dir: str,
        resolved_topics_json: str,
        created_at_wall_ns: int,
        robot_type: str | None = None,
        eef_type: str | None = None,
        topics_version: str | None = None,
        robot_version: str | None = None,
        lower_commit: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO hil_sessions(
                  episode_id, task_id, session_state, replay_export_status,
                  created_at_wall_ns, robot_type, robot_version, lower_commit,
                  eef_type, control_profile, topics_version, resolved_topics_json,
                  session_dir, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    task_id,
                    STATE_PREPARING,
                    EXPORT_NOT_STARTED,
                    created_at_wall_ns,
                    robot_type,
                    robot_version,
                    lower_commit,
                    eef_type,
                    control_profile,
                    topics_version,
                    resolved_topics_json,
                    session_dir,
                    meta_json,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO hil_episode_labels(episode_id, label_status)
                VALUES (?, 'pending')
                """,
                (episode_id,),
            )
            stamps = TimeStamps(
                ros_time_ns=0,
                monotonic_time_ns=created_at_wall_ns,
                wall_time_ns=created_at_wall_ns,
            )
            self._insert_event_unlocked(
                episode_id,
                "state_transition",
                stamps,
                "database",
                {"to": STATE_PREPARING, "from": None},
            )

    def get_session(self, episode_id: str) -> dict[str, Any] | None:
        cur = self._conn.execute(
            "SELECT * FROM hil_sessions WHERE episode_id = ?", (episode_id,)
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM hil_sessions ORDER BY created_at_wall_ns DESC"
        )
        return [dict(r) for r in cur.fetchall()]

    def list_active_sessions(self) -> list[dict[str, Any]]:
        placeholders = ",".join("?" * len(ACTIVE_SESSION_STATES))
        cur = self._conn.execute(
            f"SELECT * FROM hil_sessions WHERE session_state IN ({placeholders})",
            tuple(ACTIVE_SESSION_STATES),
        )
        return [dict(r) for r in cur.fetchall()]

    def list_by_export_status(self, status: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM hil_sessions WHERE replay_export_status=? ORDER BY created_at_wall_ns DESC",
            (status,),
        )
        return [dict(r) for r in cur.fetchall()]

    def snapshot(self, episode_id: str) -> SessionSnapshot:
        row = self.get_session(episode_id)
        if row is None:
            raise KeyError(f"unknown episode_id={episode_id}")
        producers = json.loads(row.get("producers_json") or "[]")
        metadata = json.loads(row.get("metadata_json") or "{}")
        return SessionSnapshot(
            episode_id=row["episode_id"],
            task_id=row["task_id"],
            session_state=row["session_state"],
            replay_export_status=row["replay_export_status"],
            session_dir=row["session_dir"],
            gate_status=row["gate_status"],
            watchdog_status=row["watchdog_status"],
            quality_status=row["quality_status"],
            result_type=row["result_type"],
            control_profile=row["control_profile"],
            record_pid=int(row["record_pid"] or 0),
            error_message=row["error_message"],
            quality_report_json=row["quality_report_json"],
            resolved_topics_json=row["resolved_topics_json"],
            producers=producers,
            metadata=metadata,
            stop_reason=row.get("stop_reason"),
            operator_label_hint=row.get("operator_label_hint") or "unknown",
        )

    def migrate_session_state(
        self,
        episode_id: str,
        to_state: str,
        stamps: TimeStamps,
        *,
        source: str = "session",
        error_message: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction():
            row = self.get_session(episode_id)
            if row is None:
                raise KeyError(episode_id)
            from_state = row["session_state"]
            allowed = SESSION_TRANSITIONS.get(from_state, frozenset())
            if to_state not in allowed and from_state != to_state:
                raise IllegalStateTransition(
                    f"illegal session_state {from_state!r} -> {to_state!r}"
                )
            if from_state == to_state:
                return
            fields = {"session_state": to_state}
            if error_message is not None:
                fields["error_message"] = error_message
            if extra:
                fields.update(extra)
            sets = ", ".join(f"{k}=?" for k in fields)
            self._conn.execute(
                f"UPDATE hil_sessions SET {sets} WHERE episode_id=?",
                (*fields.values(), episode_id),
            )
            self._insert_event_unlocked(
                episode_id,
                "state_transition",
                stamps,
                source,
                {"from": from_state, "to": to_state, "error": error_message},
            )

    def migrate_export_status(
        self,
        episode_id: str,
        to_status: str,
        stamps: TimeStamps,
        *,
        source: str = "publish",
    ) -> None:
        with self.transaction():
            row = self.get_session(episode_id)
            if row is None:
                raise KeyError(episode_id)
            from_status = row["replay_export_status"]
            allowed = EXPORT_TRANSITIONS.get(from_status, frozenset())
            if to_status not in allowed and from_status != to_status:
                raise IllegalStateTransition(
                    f"illegal replay_export_status {from_status!r} -> {to_status!r}"
                )
            if from_status == to_status:
                return
            self._conn.execute(
                "UPDATE hil_sessions SET replay_export_status=? WHERE episode_id=?",
                (to_status, episode_id),
            )
            self._insert_event_unlocked(
                episode_id,
                "export_transition",
                stamps,
                source,
                {"from": from_status, "to": to_status},
            )

    def update_fields(self, episode_id: str, **fields: Any) -> None:
        if not fields:
            return
        with self.transaction():
            sets = ", ".join(f"{k}=?" for k in fields)
            self._conn.execute(
                f"UPDATE hil_sessions SET {sets} WHERE episode_id=?",
                (*fields.values(), episode_id),
            )

    def set_metadata(self, episode_id: str, metadata: dict[str, Any]) -> None:
        self.update_fields(
            episode_id, metadata_json=json.dumps(metadata, ensure_ascii=False)
        )

    def set_producers(self, episode_id: str, producers: list[dict[str, Any]]) -> None:
        self.update_fields(episode_id, producers_json=json.dumps(producers))

    def insert_event(
        self,
        episode_id: str,
        event_type: str,
        stamps: TimeStamps,
        source: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction():
            self._insert_event_unlocked(episode_id, event_type, stamps, source, payload)

    def _insert_event_unlocked(
        self,
        episode_id: str,
        event_type: str,
        stamps: TimeStamps,
        source: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO hil_events(
              episode_id, event_type, ros_time_ns, monotonic_time_ns, wall_time_ns,
              source_header_stamp_ns, source, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_id,
                event_type,
                stamps.ros_time_ns,
                stamps.monotonic_time_ns,
                stamps.wall_time_ns,
                stamps.source_header_stamp_ns,
                source,
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )

    def upsert_bag(
        self,
        episode_id: str,
        bag_type: str,
        path: str,
        state: str,
        *,
        size_bytes: int = 0,
        duration_sec: float = 0.0,
        quality_report_json: str | None = None,
    ) -> None:
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO hil_bags(episode_id, bag_type, path, state, size_bytes, duration_sec, quality_report_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id, bag_type) DO UPDATE SET
                  path=excluded.path,
                  state=excluded.state,
                  size_bytes=excluded.size_bytes,
                  duration_sec=excluded.duration_sec,
                  quality_report_json=excluded.quality_report_json
                """,
                (
                    episode_id,
                    bag_type,
                    path,
                    state,
                    size_bytes,
                    duration_sec,
                    quality_report_json,
                ),
            )

    def get_label(self, episode_id: str) -> EpisodeLabel | None:
        row = self._conn.execute(
            "SELECT * FROM hil_episode_labels WHERE episode_id=?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return EpisodeLabel(**{k: row[k] for k in EpisodeLabel.__dataclass_fields__})

    def upsert_label(self, label: EpisodeLabel, *, actor: str, event_type: str) -> None:
        with self.transaction():
            self._conn.execute(
                """
                INSERT INTO hil_episode_labels(
                  episode_id, label_status, operator_label_hint, stop_reason,
                  final_label, failure_reason, labeler, label_version,
                  labeled_at_wall_ns, reviewer, reviewed_at_wall_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(episode_id) DO UPDATE SET
                  label_status=excluded.label_status,
                  operator_label_hint=excluded.operator_label_hint,
                  stop_reason=excluded.stop_reason,
                  final_label=excluded.final_label,
                  failure_reason=excluded.failure_reason,
                  labeler=excluded.labeler,
                  label_version=excluded.label_version,
                  labeled_at_wall_ns=excluded.labeled_at_wall_ns,
                  reviewer=excluded.reviewer,
                  reviewed_at_wall_ns=excluded.reviewed_at_wall_ns
                """,
                (
                    label.episode_id,
                    label.label_status,
                    label.operator_label_hint,
                    label.stop_reason,
                    label.final_label,
                    label.failure_reason,
                    label.labeler,
                    label.label_version,
                    label.labeled_at_wall_ns,
                    label.reviewer,
                    label.reviewed_at_wall_ns,
                ),
            )
            import time

            self._conn.execute(
                """
                INSERT INTO hil_label_events(episode_id, event_type, actor, wall_time_ns, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    label.episode_id,
                    event_type,
                    actor,
                    time.time_ns(),
                    json.dumps(label.to_dict(), ensure_ascii=False),
                ),
            )
            # Mirror key fields onto session row for inspect convenience.
            self._conn.execute(
                """
                UPDATE hil_sessions SET stop_reason=?, operator_label_hint=?
                WHERE episode_id=?
                """,
                (label.stop_reason, label.operator_label_hint, label.episode_id),
            )

    def list_label_events(self, episode_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT * FROM hil_label_events WHERE episode_id=? ORDER BY event_id",
            (episode_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def export_session_json(self, episode_id: str, path: Path) -> None:
        snap = self.snapshot(episode_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = snap.to_dict()
        label = self.get_label(episode_id)
        if label is not None:
            payload["label"] = label.to_dict()
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
