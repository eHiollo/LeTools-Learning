"""C0 unit tests for hil_collection (no ROS)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from kuavo_rl.hil_collection import (
    CollectionConfig,
    CollectionPhaseState,
    HILCollectionOrchestrator,
    StickEdgeDetector,
    make_episode_id,
)
from kuavo_rl.hil_recording import RecordingConfig, now_stamps
from kuavo_rl.hil_recording.database import HILDatabase
from kuavo_rl.hil_recording.models import (
    EXPORT_PENDING_REVIEW,
    EXPORT_PUBLISHED,
    EXPORT_QUARANTINED,
    PHASE_COLLECTION_ENDED,
    PHASE_RECORDING,
    PHASE_RESETTING,
    REVIEW_READY_MARKER,
    SCHEMA_VERSION,
    SCHEMA_VERSION_V001,
    TRAIN_READY_MARKER,
    EpisodeControlEvent,
    RecordRequest,
    ResultEvent,
    STATE_FINALIZED_HEALTHY,
)
from kuavo_rl.hil_recording.session import HILRecordingSession
from kuavo_rl.recording import HILReplayWriter, TransitionRecord


@pytest.fixture
def ros_clock():
    from kuavo_rl.hil_recording.timebase import set_ros_time_provider

    t0 = [1_700_000_000_000_000_000]

    def _now():
        t0[0] += 10_000_000
        return t0[0]

    set_ros_time_provider(_now)
    yield
    set_ros_time_provider(None)


def test_make_episode_id_format():
    eid = make_episode_id("box_to_chest_v1")
    assert "box_to_chest_v1" in eid
    assert len(eid.split("_")[-1]) == 8


def test_stick_edge_detector_rearms():
    det = StickEdgeDetector(
        trigger_threshold=0.8, rearm_neutral_threshold=0.2, debounce_s=0.0
    )
    assert det.update(0.9, 0.0) == "right_stick_right"
    assert det.update(0.9, 0.0) is None  # held
    assert det.update(0.0, 0.0) is None  # rearm
    assert det.update(-0.95, 0.0) == "right_stick_left"
    assert det.update(0.0, 0.0) is None
    assert det.update(0.0, -0.9) == "right_stick_down"


def test_phase_state_machine():
    st = CollectionPhaseState()
    assert st.phase == PHASE_RESETTING
    st.apply(
        EpisodeControlEvent("right_stick_right", "mock", now_stamps())
    )
    assert st.phase == PHASE_RECORDING
    st.apply(EpisodeControlEvent("right_stick_down", "mock", now_stamps()))
    assert st.phase == "FINALIZING"
    st.phase = PHASE_RESETTING
    st.apply(EpisodeControlEvent("right_stick_down", "mock", now_stamps()))
    assert st.phase == PHASE_COLLECTION_ENDED


def test_db_migration_v001_to_v002(tmp_path, ros_clock):
    db_path = tmp_path / "old.db"
    # Create v001-shaped DB manually
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE schema_version (version TEXT NOT NULL);
        INSERT INTO schema_version(version) VALUES ('hil-db-v001');
        CREATE TABLE hil_sessions (
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
          producers_json TEXT NOT NULL DEFAULT '[]'
        );
        INSERT INTO hil_sessions(
          episode_id, task_id, session_state, created_at_wall_ns,
          control_profile, resolved_topics_json, session_dir
        ) VALUES ('legacy_ep', 't', 'Preparing', 1, 'act', '{}', '/tmp/x');
        CREATE TABLE hil_bags (
          bag_id INTEGER PRIMARY KEY AUTOINCREMENT,
          episode_id TEXT NOT NULL,
          bag_type TEXT NOT NULL,
          path TEXT NOT NULL,
          state TEXT NOT NULL,
          size_bytes INTEGER DEFAULT 0,
          duration_sec REAL DEFAULT 0,
          quality_report_json TEXT,
          UNIQUE (episode_id, bag_type)
        );
        CREATE TABLE hil_events (
          event_id INTEGER PRIMARY KEY AUTOINCREMENT,
          episode_id TEXT NOT NULL,
          event_type TEXT NOT NULL,
          ros_time_ns INTEGER,
          monotonic_time_ns INTEGER NOT NULL,
          wall_time_ns INTEGER NOT NULL,
          source_header_stamp_ns INTEGER,
          source TEXT NOT NULL,
          payload_json TEXT
        );
        """
    )
    conn.commit()
    conn.close()

    db = HILDatabase(db_path)
    ver = db._conn.execute("SELECT version FROM schema_version").fetchone()[0]
    assert ver == SCHEMA_VERSION
    assert db.get_label("legacy_ep") is not None
    snap = db.snapshot("legacy_ep")
    assert snap.metadata == {}
    db.close()


def test_pending_review_then_accepted_label_gate(tmp_path, ros_clock):
    root = tmp_path / "hil"
    cfg = RecordingConfig(
        root_dir=root,
        dry_run_recorder=True,
        skip_gate_ros=True,
        post_roll_s=0.05,
        bag_stall_timeout_s=60.0,
    )
    session = HILRecordingSession(cfg)
    session.recover_interrupted()
    eid = "ep_label_gate"
    session.create(
        RecordRequest(
            episode_id=eid,
            task_id="t",
            control_profile="act",
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=0.05,
            metadata={"operator": "tester", "scene_id": "sim_a"},
        )
    )
    assert session.db.snapshot(eid).metadata["operator"] == "tester"
    session.register_producer("act_runner", 1, "policy")
    session.start(eid)

    staging = cfg.staging_dir(eid)
    writer = HILReplayWriter(tmp_path, "x", staging_dir=staging)
    for step in range(2):
        stamps = now_stamps()
        rec = TransitionRecord(
            experiment_id="x",
            episode_id=eid,
            step_id=step,
            timestamp=1.0,
            action=[0.0] * 16,
            reward=0.0,
            reward_source="none",
            terminated=step == 1,
            truncated=False,
            fault_code="NONE",
            is_intervention=False,
            action_clipped=False,
            extras={
                "intervention_mask": [0] * 16,
                "intervention_segment_step": 0,
            },
        )
        obs = {"observation.state": np.zeros(16, dtype=np.float32)}
        writer.log_transition(rec, observation=obs, next_observation=obs)
        session.update_transition(
            {
                "step_id": step,
                "stamps": stamps,
                "intervention_mask": [0] * 16,
                "intervention_segment_step": 0,
            }
        )
    writer.close()
    session.record_event(
        ResultEvent(episode_id=eid, event_type="success", source="t", stamps=now_stamps())
    )
    session.request_stop(eid, "success")
    final = session.wait_finalized(eid, timeout_s=15.0)
    assert final.session_state == STATE_FINALIZED_HEALTHY

    # Old alias must NOT create TRAIN_READY / Published
    report = session.publish_replay(eid)
    assert report.status == EXPORT_PENDING_REVIEW
    pending = cfg.pending_review_dir / eid
    assert (pending / REVIEW_READY_MARKER).exists()
    assert not (cfg.accepted_replay_dir / eid / TRAIN_READY_MARKER).exists()

    orch_cfg = CollectionConfig(root=root, skip_gate_ros=True, dry_run_recorder=True)
    orch = HILCollectionOrchestrator(orch_cfg)
    # Direct accept without review must fail
    bad = orch.publish_train_ready(eid)
    assert bad["status"] == "Rejected"

    orch.label_episode(eid, final_label="success", reason="verified", labeler="a")
    orch.review_episode(eid, approve=True, reviewer="a")
    good = orch.publish_train_ready(eid)
    assert good["status"] == EXPORT_PUBLISHED
    accepted = cfg.accepted_replay_dir / eid
    assert (accepted / TRAIN_READY_MARKER).exists()
    assert not pending.exists()
    orch.close()
    session.close()


def test_preflight_and_collect_cli_exit_codes(tmp_path):
    cfg_path = tmp_path / "col.yaml"
    root = tmp_path / "data"
    cfg_path.write_text(
        f"""
collection:
  task_id: t1
  root: {root}
  mode: act
recording:
  topics_profile: configs/rl/hil_topics_sim_v002.yaml
  skip_gate_ros: true
  dry_run_recorder: true
metadata:
  robot_type: Kuavo
  eef_type: leju_claw
runner:
  shadow_mode: false
""",
        encoding="utf-8",
    )
    script = Path("scripts/rl/collect_hil_dataset.py")
    py = sys.executable
    r = subprocess.run(
        [py, str(script), "--config", str(cfg_path), "preflight"],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr + r.stdout
    assert '"status": "Pass"' in r.stdout or '"status": "Pass"' in r.stdout.replace("'", '"')

    r2 = subprocess.run(
        [py, str(script), "--config", str(cfg_path), "collect"],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
    )
    assert r2.returncode == 2  # missing --confirm-live

    r3 = subprocess.run(
        [py, str(script), "--config", str(cfg_path), "collect", "--confirm-live"],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
    )
    assert r3.returncode == 3  # live ACT collect not wired yet

    r4 = subprocess.run(
        [
            py,
            str(script),
            "--config",
            str(cfg_path),
            "collect",
            "--dry-run",
            "--max-steps",
            "3",
            "--operator",
            "tester",
        ],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
    )
    assert r4.returncode == 0, r4.stderr + r4.stdout
    assert "pending_review" in r4.stdout


def test_orchestrator_manifest_and_inspect(tmp_path, ros_clock):
    root = tmp_path / "hil"
    cfg = CollectionConfig(
        root=root,
        task_id="box",
        skip_gate_ros=True,
        dry_run_recorder=True,
        config_path=tmp_path / "c.yaml",
    )
    (tmp_path / "c.yaml").write_text("collection: {}\n", encoding="utf-8")
    orch = HILCollectionOrchestrator(cfg)
    man = orch.ensure_collection_manifest(
        collection_id="20260717_box_a", operator="fulin", scene_id="table"
    )
    assert man["operator"] == "fulin"
    assert (root / "collection_manifest.json").exists()
    pf = orch.preflight()
    assert pf["status"] == "Pass"
    assert pf["quest_event_source_publishes_actions"] is False
    idx = orch.inspect()
    assert "episodes" in idx
    orch.close()


def test_dry_run_collect_to_pending_review(tmp_path, ros_clock):
    root = tmp_path / "hil"
    orch = HILCollectionOrchestrator(
        CollectionConfig(root=root, skip_gate_ros=True, dry_run_recorder=True, task_id="box")
    )
    out = orch.collect_episode_dry_run(
        episode_id="dry_ep1",
        max_steps=4,
        end_event="right_stick_right",
        operator="op",
    )
    assert out["status"] == "pending_review"
    assert out["review_ready"] is True
    assert out["accepted_blocked"] is True
    assert out["train_ready_blocked"] is True

    # rerecord path
    out2 = orch.collect_episode_dry_run(
        episode_id="dry_ep2",
        max_steps=2,
        end_event="right_stick_left",
    )
    assert out2["status"] == "quarantined_rerecord"
    assert (root / "quarantine" / "dry_ep2").exists()

    # C3 report + label/review
    report = orch.collection_report()
    assert report["pending_review_ready"] >= 1
    orch.label_episode("dry_ep1", final_label="success", reason="ok", labeler="a")
    orch.review_episode("dry_ep1", approve=True, reviewer="a")
    pub = orch.publish_train_ready("dry_ep1")
    assert pub["status"] == "Published"
    assert (root / "accepted_replay" / "dry_ep1" / "TRAIN_READY").exists()
    orch.close()


def test_batch_dry_run_with_rerecord_retry(tmp_path, ros_clock):
    root = tmp_path / "hil"
    orch = HILCollectionOrchestrator(
        CollectionConfig(
            root=root,
            skip_gate_ros=True,
            dry_run_recorder=True,
            task_id="batch_t",
            batch_stop_on_quarantine=True,
        )
    )
    out = orch.batch_dry_run(
        episodes=2,
        max_steps=2,
        operator="op",
        end_events=["right_stick_left", "right_stick_right", "right_stick_down"],
        collection_id="20260717_batch_t",
    )
    # first attempt rerecord (retry), then early_end, then collection_complete stops
    assert out["completed_episodes"] >= 1
    assert (root / "collection_manifest.json").exists()
    assert any(r["status"] == "quarantined_rerecord" for r in out["results"])
    assert any(r["status"] == "pending_review" for r in out["results"])
    orch.close()


def test_live_preflight_blocks_without_calibration(tmp_path):
    root = tmp_path / "hil"
    cal_path = tmp_path / "stick.yaml"
    orch = HILCollectionOrchestrator(
        CollectionConfig(
            root=root,
            skip_gate_ros=True,
            dry_run_recorder=True,
            stick_calibration_path=cal_path,
            episode_control="quest_right_stick",
            right_stick_exclusive=True,
            require_collection_mode_ack=True,
            require_stick_calibration_for_live=True,
        )
    )
    out = orch.preflight(
        for_live_collect=True,
        exclusive_overrides={
            "ros_param_get": lambda _k, d: False,
            "rosnode_list": lambda: [],
            "topic_present": lambda _t: True,
        },
    )
    assert out["status"] == "Block"
    assert out["stick_exclusive"]["status"] == "Block"
    orch.close()


def test_y_stick_preflight_skips_stick_exclusive(tmp_path):
    root = tmp_path / "hil"
    orch = HILCollectionOrchestrator(
        CollectionConfig(
            root=root,
            skip_gate_ros=True,
            dry_run_recorder=True,
            episode_control="quest_y_stick",
        )
    )
    out = orch.preflight(for_live_collect=True)
    assert out["status"] == "Pass"
    assert out["episode_control"] == "quest_y_stick"
    assert out["stick_exclusive"]["status"] == "Skipped"
    assert "Hold Y" in str(out["episode_control_card"])
    orch.close()
