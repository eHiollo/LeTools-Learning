"""HILRecordingSession: create / start / request_stop / wait_finalized / publish."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from queue import Queue
from typing import Any

from kuavo_rl.hil_recording.audit_publisher import AuditPublisher
from kuavo_rl.hil_recording.config import RecordingConfig
from kuavo_rl.hil_recording.database import HILDatabase
from kuavo_rl.hil_recording.gate import RecordGate
from kuavo_rl.hil_recording.models import (
    EXPORT_QUARANTINED,
    FINALIZED_OK,
    STATE_CANCELED,
    STATE_FAILED_RECORD,
    STATE_FAILED_SELF_CHECK,
    STATE_FINALIZED_DEGRADED,
    STATE_FINALIZED_HEALTHY,
    STATE_FINALIZING,
    STATE_PREPARING,
    STATE_RECORDING,
    STATE_STOPPING,
    ExportReport,
    ProducerInfo,
    RecordRequest,
    RecoveryReport,
    ResultEvent,
    SessionSnapshot,
    StopRequest,
)
from kuavo_rl.hil_recording.publish_replay import (
    publish_accepted,
    publish_pending_review,
    publish_replay,
    quarantine_episode,
)
from kuavo_rl.hil_recording.quality import run_quality_check, write_quality_report
from kuavo_rl.hil_recording.result_events import prefer_result, validate_event
from kuavo_rl.hil_recording.rosbag_recorder import RecorderHandle, RosbagRecorder
from kuavo_rl.hil_recording.timebase import now_stamps, sleep_monotonic
from kuavo_rl.hil_recording.topic_relay import TopicRelayHandle, TopicRelayManager
from kuavo_rl.hil_recording.topics import ResolvedTopics, resolve_topics
from kuavo_rl.hil_recording.watchdog import RecorderWatchdog, TopicFreshnessCache, drain_stop_queue


class HILRecordingSession:
    """Episode recording controller. Only this object (main thread) stops the recorder."""

    def __init__(
        self,
        config: RecordingConfig,
        *,
        db: HILDatabase | None = None,
        gate: RecordGate | None = None,
        profile_path: str | Path | None = None,
    ):
        self.config = config
        self.db = db or HILDatabase(config.db_path)
        self.gate = gate or RecordGate(config, self.db)
        self.profile_path = profile_path
        self.recorder = RosbagRecorder(config)
        self.topic_relay = TopicRelayManager()
        self.stop_queue: Queue[StopRequest] = Queue()
        self.freshness = TopicFreshnessCache()

        self._producers: list[ProducerInfo] = []
        self._resolved: ResolvedTopics | None = None
        self._handle: RecorderHandle | None = None
        self._relay_handle: TopicRelayHandle | None = None
        self._watchdog: RecorderWatchdog | None = None
        self._audit: AuditPublisher | None = None
        self._current_episode: str | None = None
        self._dry_run = False
        self._post_roll_s = config.post_roll_s
        self._stop_lock = threading.Lock()
        self._stop_started = False
        self._finalize_thread: threading.Thread | None = None
        self._finalized = threading.Event()
        self._transition_count = 0
        self._last_state_ros_ns: int | None = None
        self._recovering = False

    # ------------------------------------------------------------------ API
    def create(self, request: RecordRequest) -> SessionSnapshot:
        if self._recovering:
            raise RuntimeError("recovery in progress; gate blocked")
        stamps = now_stamps()
        resolved = resolve_topics(
            control_profile=request.control_profile,
            robot_type=request.robot_type,
            eef_type=request.eef_type,
            profile_path=self.profile_path,
        )
        self._resolved = resolved
        self._dry_run = bool(request.dry_run or self.config.dry_run_recorder)
        self._post_roll_s = float(request.post_roll_s)
        session_dir = self.config.session_dir(request.episode_id)
        self.config.staging_dir(request.episode_id)
        self.config.bags_dir(request.episode_id)

        self.db.insert_session(
            episode_id=request.episode_id,
            task_id=request.task_id,
            control_profile=request.control_profile,
            session_dir=str(session_dir),
            resolved_topics_json=resolved.to_json(),
            created_at_wall_ns=stamps.wall_time_ns,
            robot_type=request.robot_type,
            eef_type=request.eef_type,
            topics_version=resolved.version,
            robot_version=request.robot_version,
            lower_commit=request.lower_commit,
            metadata=request.metadata,
        )
        self._current_episode = request.episode_id
        self._stop_started = False
        self._finalized.clear()
        self._transition_count = 0
        self._producers = []
        self._audit = AuditPublisher(
            session_dir, enable_ros=not self._dry_run and not request.skip_gate_ros
        )
        self.db.export_session_json(request.episode_id, session_dir / "session.json")
        return self.db.snapshot(request.episode_id)

    def register_producer(self, name: str, pid: int, kind: str) -> None:
        if not self._current_episode:
            raise RuntimeError("create() first")
        info = ProducerInfo(name=name, pid=pid, kind=kind)
        self._producers = [p for p in self._producers if p.name != name]
        self._producers.append(info)
        self.db.set_producers(
            self._current_episode,
            [{"name": p.name, "pid": p.pid, "kind": p.kind} for p in self._producers],
        )

    def start(self, episode_id: str) -> SessionSnapshot:
        row = self.db.get_session(episode_id)
        if row is None:
            raise KeyError(episode_id)
        if row["session_state"] != STATE_PREPARING:
            raise RuntimeError(f"cannot start from state={row['session_state']}")

        resolved = self._resolved or ResolvedTopics.from_json(row["resolved_topics_json"])
        self._resolved = resolved
        session_dir = Path(row["session_dir"])
        skip_ros = self.config.skip_gate_ros or self._dry_run

        report = self.gate.evaluate(
            resolved=resolved,
            session_dir=session_dir,
            producers=self._producers,
            episode_id=episode_id,
            skip_ros=skip_ros,
        )
        self.db.update_fields(
            episode_id,
            gate_status=report.status,
        )
        if report.status == "Block":
            stamps = now_stamps()
            self.db.migrate_session_state(
                episode_id,
                STATE_FAILED_RECORD,
                stamps,
                error_message="; ".join(report.reasons) or "gate_block",
            )
            self.db.export_session_json(episode_id, session_dir / "session.json")
            raise RuntimeError(f"gate blocked: {report.reasons}")

        # Relay bus topics → canonical bag names (e.g. /camera → /cam_h) when configured.
        # Gate already passed on ``source``; dry-run skips live relay processes.
        relay_pairs = resolved.relay_pairs()
        if relay_pairs and not self._dry_run:
            try:
                self._relay_handle = self.topic_relay.start(
                    relay_pairs,
                    log_path=session_dir / "topic_relay.log",
                )
            except Exception as exc:
                stamps = now_stamps()
                self.db.migrate_session_state(
                    episode_id,
                    STATE_FAILED_RECORD,
                    stamps,
                    error_message=f"topic_relay_failed: {exc}",
                )
                self.db.export_session_json(episode_id, session_dir / "session.json")
                raise RuntimeError(f"topic relay failed: {exc}") from exc

        # Start recorder (records canonical ``name`` topics)
        topics = resolved.record_topic_names()
        try:
            handle = self.recorder.start(episode_id, topics, dry_run=self._dry_run)
        except Exception:
            self.topic_relay.stop(self._relay_handle)
            self._relay_handle = None
            raise
        self._handle = handle
        stamps = now_stamps()
        self.db.update_fields(
            episode_id,
            record_pid=handle.pid,
            record_command=json.dumps(handle.command),
            stdout_path=str(handle.stdout_path),
            stderr_path=str(handle.stderr_path),
            started_at_ros_ns=stamps.ros_time_ns,
            watchdog_status="Running",
        )
        self.db.upsert_bag(
            episode_id, "original", str(handle.bag_path), state="recording"
        )
        self.db.migrate_session_state(episode_id, STATE_RECORDING, stamps)

        freshness_map = {
            t.bus_name: float(t.freshness_s)
            for t in resolved.topics
            if t.mode == "streaming" and t.freshness_s is not None and t.applies_to(resolved.control_profile)
        }
        self._watchdog = RecorderWatchdog(
            self.config,
            self.stop_queue,
            freshness=self.freshness,
            streaming_freshness_s=freshness_map,
        )
        self._watchdog.start(handle, episode_id)
        self._current_episode = episode_id
        self.db.export_session_json(episode_id, session_dir / "session.json")
        return self.db.snapshot(episode_id)

    def record_event(self, event: ResultEvent) -> None:
        validate_event(event)
        row = self.db.get_session(event.episode_id)
        if row is None:
            raise KeyError(event.episode_id)
        current = prefer_result(row.get("result_type"), event.event_type)
        self.db.insert_event(
            event.episode_id,
            event.event_type,
            event.stamps,
            event.source,
            event.payload,
        )
        self.db.update_fields(
            event.episode_id,
            result_type=current,
            result_event_ros_ns=event.stamps.ros_time_ns,
            result_event_mono_ns=event.stamps.monotonic_time_ns,
            result_event_source=event.source,
        )
        if self._audit is not None:
            self._audit.publish_result(event)

    def update_transition(self, info: dict[str, Any]) -> None:
        if self._current_episode is None or self._audit is None:
            return
        stamps = info.get("stamps") or now_stamps(
            source_header_stamp_ns=info.get("source_header_stamp_ns")
        )
        if hasattr(stamps, "to_dict"):
            stamp_dict = stamps.to_dict()
            ros_ns = stamps.ros_time_ns
        else:
            stamp_dict = stamps
            ros_ns = int(stamps.get("ros_time_ns", 0))
        self._last_state_ros_ns = ros_ns
        payload = {
            "episode_id": self._current_episode,
            "step_id": info.get("step_id"),
            "policy_action": info.get("policy_action"),
            "executed_action": info.get("executed_action") or info.get("teleop_replay_action"),
            "raw_vr_action": info.get("teleop_raw_action") or info.get("raw_vr_action"),
            "intervention_mask": info.get("intervention_mask"),
            "intervention_segment_id": info.get("intervention_segment_id"),
            "intervention_segment_step": info.get("intervention_segment_step"),
            "reward": info.get("reward"),
            "fault_code": info.get("fault_code"),
            "stamps": stamp_dict,
        }
        self._audit.publish_transition(payload, stamps if hasattr(stamps, "ros_time_ns") else now_stamps())
        self._transition_count += 1

    def poll_stop_request(self) -> StopRequest | None:
        """Episode controller should poll this between steps."""
        return drain_stop_queue(self.stop_queue)

    def request_stop(self, episode_id: str, reason: str) -> None:
        """Non-blocking: enqueue stop; main thread runs the stop sequence."""
        self.stop_queue.put(
            StopRequest(
                episode_id=episode_id,
                reason=reason,
                source="controller",
                stamps=now_stamps(),
            )
        )
        # Kick stop sequence on a helper only if already Recording — still
        # controller-owned via lock; watchdog never calls this path's internals
        # except via queue.
        self._ensure_stop_sequence(episode_id, reason)

    def _ensure_stop_sequence(self, episode_id: str, reason: str) -> None:
        with self._stop_lock:
            if self._stop_started:
                return
            row = self.db.get_session(episode_id)
            if row is None:
                return
            if row["session_state"] not in (STATE_RECORDING,):
                return
            self._stop_started = True
            # Run stop sequence synchronously on caller thread (episode controller).
            self._run_stop_sequence(episode_id, reason)

    def _run_stop_sequence(self, episode_id: str, reason: str) -> None:
        stamps = now_stamps()
        session_dir = self.config.session_dir(episode_id)
        self.db.migrate_session_state(episode_id, STATE_STOPPING, stamps, source="stop")
        self.db.update_fields(episode_id, stopped_at_ros_ns=stamps.ros_time_ns)

        # 1) terminal event already expected via record_event; ensure reason stored
        row = self.db.get_session(episode_id)
        if row and not row.get("result_type"):
            # map reason string to result if possible
            mapped = _reason_to_result(reason)
            if mapped:
                self.record_event(
                    ResultEvent(
                        episode_id=episode_id,
                        event_type=mapped,
                        source="stop_sequence",
                        stamps=stamps,
                        payload={"reason": reason},
                    )
                )

        # 2) control hold is caller's responsibility (runner already stopped)

        # 3) post-roll
        sleep_monotonic(self._post_roll_s)

        # 4) final state after terminal (best-effort check)
        terminal_ros = None
        row = self.db.get_session(episode_id)
        if row:
            terminal_ros = row.get("result_event_ros_ns")
        if (
            terminal_ros
            and self._last_state_ros_ns is not None
            and self._last_state_ros_ns < int(terminal_ros)
        ):
            # wait a bit more for late state
            sleep_monotonic(0.2)

        # 5) stop rosbag
        if self._watchdog is not None:
            report = self._watchdog.stop()
            self.db.update_fields(
                episode_id,
                watchdog_status="Stopped",
                watchdog_report_path=str(session_dir / "watchdog.report.json"),
            )
            _ = report
        rc = 0
        if self._handle is not None:
            rc = self.recorder.stop(self._handle)
            size = self._handle.bag_path.stat().st_size if self._handle.bag_path.exists() else 0
            self.db.upsert_bag(
                episode_id,
                "original",
                str(self._handle.bag_path),
                state="stopped",
                size_bytes=size,
            )
        # Stop source→canonical relays after rosbag has flushed.
        self.topic_relay.stop(self._relay_handle)
        self._relay_handle = None

        # 6) Finalizing async
        stamps2 = now_stamps()
        if "record_error" in reason or rc == -9:
            self.db.migrate_session_state(
                episode_id,
                STATE_FAILED_RECORD,
                stamps2,
                error_message=reason,
            )
            quarantine_episode(self.config, self.db, episode_id, reason=reason)
            self._finalized.set()
            self.db.export_session_json(episode_id, session_dir / "session.json")
            return

        self.db.migrate_session_state(episode_id, STATE_FINALIZING, stamps2)
        self._finalize_thread = threading.Thread(
            target=self._finalize_worker,
            args=(episode_id, reason),
            name="hil-finalize",
            daemon=True,
        )
        self._finalize_thread.start()

    def _finalize_worker(self, episode_id: str, reason: str) -> None:
        try:
            row = self.db.get_session(episode_id)
            assert row is not None
            resolved = ResolvedTopics.from_json(row["resolved_topics_json"])
            bag_path = self.config.bags_dir(episode_id) / "original.bag"
            staging = self.config.staging_dir(episode_id)
            session_dir = Path(row["session_dir"])
            report = run_quality_check(
                bag_path=bag_path,
                staging_dir=staging,
                session_dir=session_dir,
                resolved=resolved,
                dry_run=self._dry_run,
            )
            write_quality_report(session_dir, report)
            self.db.update_fields(
                episode_id,
                quality_status=report.status,
                quality_report_json=json.dumps(report.to_dict(), ensure_ascii=False),
            )
            stamps = now_stamps()
            if report.status == "Healthy":
                self.db.migrate_session_state(
                    episode_id, STATE_FINALIZED_HEALTHY, stamps, source="quality"
                )
            elif report.status == "Degraded" and self.config.allow_degraded_export:
                self.db.migrate_session_state(
                    episode_id, STATE_FINALIZED_DEGRADED, stamps, source="quality"
                )
            else:
                self.db.migrate_session_state(
                    episode_id,
                    STATE_FAILED_SELF_CHECK,
                    stamps,
                    error_message="; ".join(report.reasons) or "quality_failed",
                    source="quality",
                )
                quarantine_episode(
                    self.config,
                    self.db,
                    episode_id,
                    reason="; ".join(report.reasons) or "quality_failed",
                )
            self.db.export_session_json(episode_id, session_dir / "session.json")
        except Exception as exc:  # noqa: BLE001
            stamps = now_stamps()
            self.db.migrate_session_state(
                episode_id,
                STATE_FAILED_RECORD,
                stamps,
                error_message=f"finalize_exception:{exc}",
            )
            quarantine_episode(
                self.config, self.db, episode_id, reason=f"finalize_exception:{exc}"
            )
        finally:
            self._finalized.set()

    def wait_finalized(self, episode_id: str, timeout_s: float = 30.0) -> SessionSnapshot:
        # Ensure any pending stop request is processed on this thread.
        req = drain_stop_queue(self.stop_queue)
        if req is not None:
            self._ensure_stop_sequence(req.episode_id, req.reason)
        elif not self._stop_started:
            # caller forgot request_stop; still allow wait after explicit stop
            pass

        row = self.db.get_session(episode_id)
        if row and row["session_state"] == STATE_RECORDING:
            # force stop if still recording
            self.request_stop(episode_id, "wait_finalized")

        ok = self._finalized.wait(timeout=timeout_s)
        if not ok:
            # timeout — leave state as-is; caller sees Stopping/Finalizing
            pass
        if self._finalize_thread is not None:
            self._finalize_thread.join(timeout=max(0.0, timeout_s))
        return self.db.snapshot(episode_id)

    def publish_pending_review(self, episode_id: str) -> ExportReport:
        snap = self.db.snapshot(episode_id)
        if snap.session_state not in FINALIZED_OK:
            raise RuntimeError(
                f"publish_pending_review rejected: session_state={snap.session_state}"
            )
        report = publish_pending_review(self.config, self.db, episode_id)
        self.db.export_session_json(
            episode_id, self.config.session_dir(episode_id) / "session.json"
        )
        return report

    def publish_accepted(self, episode_id: str) -> ExportReport:
        report = publish_accepted(self.config, self.db, episode_id)
        self.db.export_session_json(
            episode_id, self.config.session_dir(episode_id) / "session.json"
        )
        return report

    def publish_replay(self, episode_id: str) -> ExportReport:
        """Deprecated alias for publish_pending_review (never skips label gate)."""
        return self.publish_pending_review(episode_id)

    def cancel(self, episode_id: str) -> SessionSnapshot:
        stamps = now_stamps()
        row = self.db.get_session(episode_id)
        if row is None:
            raise KeyError(episode_id)
        if row["session_state"] == STATE_RECORDING:
            self.request_stop(episode_id, "cancel")
            self.wait_finalized(episode_id, timeout_s=15.0)
        # From Preparing or after stop, mark canceled if still preparing
        row = self.db.get_session(episode_id)
        if row and row["session_state"] == STATE_PREPARING:
            self.db.migrate_session_state(episode_id, STATE_CANCELED, stamps)
        return self.db.snapshot(episode_id)

    def recover_interrupted(self) -> RecoveryReport:
        self._recovering = True
        report = RecoveryReport()
        try:
            for row in self.db.list_active_sessions():
                eid = row["episode_id"]
                pid = int(row.get("record_pid") or 0)
                cmd = row.get("record_command") or ""
                if pid > 0 and self.recorder.kill_orphan(pid, command_hint=cmd):
                    report.killed_pids.append(pid)
                active = self.config.root_dir / ".active"
                if active.exists():
                    try:
                        active.unlink()
                    except OSError as exc:
                        report.errors.append(str(exc))
                staging = Path(row["session_dir"]) / "staging"
                stamps = now_stamps()
                bag = Path(row["session_dir"]) / "bags" / "original.bag"
                # Whitelist path: Recording|Stopping|Finalizing → ...
                state = row["session_state"]
                if state == STATE_RECORDING:
                    self.db.migrate_session_state(
                        eid, STATE_STOPPING, stamps, source="recover"
                    )
                    stamps = now_stamps()
                    state = STATE_STOPPING
                if bag.exists() and bag.stat().st_size > 0:
                    if state == STATE_STOPPING:
                        self.db.migrate_session_state(
                            eid, STATE_FINALIZING, stamps, source="recover"
                        )
                    elif state != STATE_FINALIZING:
                        # already Finalizing
                        pass
                    # Run quality synchronously during recovery
                    resolved = ResolvedTopics.from_json(row["resolved_topics_json"])
                    q = run_quality_check(
                        bag_path=bag,
                        staging_dir=staging if staging.exists() else self.config.staging_dir(eid),
                        session_dir=Path(row["session_dir"]),
                        resolved=resolved,
                        dry_run=True,
                    )
                    write_quality_report(Path(row["session_dir"]), q)
                    if q.status == "Healthy":
                        self.db.migrate_session_state(
                            eid, STATE_FINALIZED_HEALTHY, now_stamps(), source="recover"
                        )
                        report.recovered.append(eid)
                    else:
                        self.db.migrate_session_state(
                            eid,
                            STATE_FAILED_SELF_CHECK,
                            now_stamps(),
                            error_message="; ".join(q.reasons),
                            source="recover",
                        )
                        quarantine_episode(
                            self.config, self.db, eid, reason="recover_quality_failed"
                        )
                        report.quarantined.append(eid)
                else:
                    self.db.migrate_session_state(
                        eid,
                        STATE_FAILED_RECORD,
                        stamps,
                        error_message="recover_no_bag",
                        source="recover",
                    )
                    quarantine_episode(
                        self.config, self.db, eid, reason="recover_no_bag"
                    )
                    report.quarantined.append(eid)
                # Always move leftover staging into quarantine if still present
                if staging.exists() and any(staging.iterdir()):
                    if eid not in report.quarantined:
                        quarantine_episode(
                            self.config, self.db, eid, reason="recover_staging_leftover"
                        )
                        if eid not in report.quarantined:
                            report.quarantined.append(eid)
        finally:
            self._recovering = False
        return report

    def staging_writer_root(self, episode_id: str) -> Path:
        """Root passed to HILReplayWriter for staging layout."""
        # HILReplayWriter writes root/experiment_id/replay/...
        # We want sessions/<eid>/staging/{transitions,frames}
        # So use a thin adapter via StagingReplayWriter instead.
        return self.config.staging_dir(episode_id)

    def close(self) -> None:
        if self._audit is not None:
            self._audit.close()
        self.db.close()


def _reason_to_result(reason: str) -> str | None:
    r = reason.lower()
    for key in ("estop", "fault", "abort", "failure", "success"):
        if key in r:
            return key
    if "record_error" in r:
        return None
    return None
