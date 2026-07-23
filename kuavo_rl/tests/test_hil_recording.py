"""Unit tests for hil_recording (dry-run, no ROS / no real rosbag)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from kuavo_rl.hil_recording.config import RecordingConfig
from kuavo_rl.hil_recording.database import HILDatabase
from kuavo_rl.hil_recording.models import (
    EXPORT_PENDING_REVIEW,
    EXPORT_QUARANTINED,
    REVIEW_READY_MARKER,
    IllegalStateTransition,
    RecordRequest,
    ResultEvent,
    STATE_FAILED_RECORD,
    STATE_FINALIZED_HEALTHY,
    STATE_PREPARING,
    STATE_RECORDING,
)
from kuavo_rl.hil_recording.session import HILRecordingSession
from kuavo_rl.hil_recording.timebase import now_stamps, set_ros_time_provider
from kuavo_rl.hil_recording.topics import resolve_topics
from kuavo_rl.recording import HILReplayWriter, TransitionRecord


@pytest.fixture
def ros_clock():
    t0 = [1_700_000_000_000_000_000]

    def _now():
        t0[0] += 10_000_000  # +10ms
        return t0[0]

    set_ros_time_provider(_now)
    yield
    set_ros_time_provider(None)


def _cfg(tmp_path: Path) -> RecordingConfig:
    return RecordingConfig(
        root_dir=tmp_path / "hilserl_vr",
        dry_run_recorder=True,
        skip_gate_ros=True,
        post_roll_s=0.05,
        bag_stall_timeout_s=30.0,
        finalize_timeout_s=10.0,
    )


def test_timebase_samples_three_clocks(ros_clock):
    a = now_stamps()
    b = now_stamps(source_header_stamp_ns=a.ros_time_ns)
    assert a.ros_time_ns > 0
    assert a.monotonic_time_ns > 0
    assert a.wall_time_ns > 0
    assert b.align_key_ns() == a.ros_time_ns


def test_topic_profile_filters_act_vs_vr():
    act = resolve_topics(control_profile="act")
    names = {t.name for t in act.for_export()}
    assert "/kuavo_arm_traj" not in names
    vr = resolve_topics(control_profile="act_vr")
    names_vr = {t.name for t in vr.for_export() if t.applies_to("act_vr")}
    assert "/kuavo_arm_traj" in {t.name for t in vr.topics if t.applies_to("act_vr") and t.required_for_export}
    latched = [t for t in act.topics if t.name == "/tf_static"][0]
    assert latched.mode == "latched"
    assert latched.min_hz is None


def test_upper_cams_profile_relays_to_vla_canonical_names():
    root = Path(__file__).resolve().parents[2]
    profile = root / "configs/rl/hil_topics_real_upper_cams_v001.yaml"
    resolved = resolve_topics(
        control_profile="vr_only",
        profile_path=profile,
    )
    record = set(resolved.record_topic_names())
    assert "/cam_h/color/image_raw/compressed" in record
    assert "/cam_l/color/image_raw/compressed" in record
    assert "/cam_r/color/image_raw/compressed" in record
    assert "/camera/color/image_raw/compressed" not in record

    pairs = dict(resolved.relay_pairs())
    assert pairs["/camera/color/image_raw/compressed"] == "/cam_h/color/image_raw/compressed"
    assert (
        pairs["/left_wrist_camera/color/image_raw/compressed"]
        == "/cam_l/color/image_raw/compressed"
    )
    assert (
        pairs["/right_wrist_camera/color/image_raw/compressed"]
        == "/cam_r/color/image_raw/compressed"
    )

    head = next(t for t in resolved.topics if t.name.endswith("cam_h/color/image_raw/compressed"))
    assert head.bus_name == "/camera/color/image_raw/compressed"
    assert head.needs_relay()


def test_gate_probes_source_not_canonical_name(tmp_path, ros_clock):
    from kuavo_rl.hil_recording.gate import RecordGate
    from kuavo_rl.hil_recording.topics import ResolvedTopics, TopicSpec

    cfg = _cfg(tmp_path)
    probed: list[str] = []

    def present(name: str) -> bool:
        probed.append(name)
        return True

    gate = RecordGate(
        cfg,
        HILDatabase(tmp_path / "gate.db"),
        topic_present=present,
        ros_master_ok=lambda: True,
    )
    resolved = ResolvedTopics(
        version="t",
        robot_type="Kuavo",
        eef_type="leju_claw",
        control_profile="vr_only",
        topics=[
            TopicSpec(
                name="/cam_h/color/image_raw/compressed",
                source="/camera/color/image_raw/compressed",
                role="training",
                mode="streaming",
                required_for_start=True,
                required_for_export=True,
                min_hz=10,
                freshness_s=1.0,
            )
        ],
    )
    report = gate.evaluate(
        resolved=resolved,
        session_dir=tmp_path / "s",
        producers=[],
        skip_ros=False,
    )
    assert report.status == "Pass"
    assert "/camera/color/image_raw/compressed" in probed
    assert "/cam_h/color/image_raw/compressed" not in probed


def test_illegal_session_transition(tmp_path, ros_clock):
    db = HILDatabase(tmp_path / "t.db")
    db.insert_session(
        episode_id="e1",
        task_id="task",
        control_profile="act",
        session_dir=str(tmp_path / "s"),
        resolved_topics_json="{}",
        created_at_wall_ns=now_stamps().wall_time_ns,
    )
    with pytest.raises(IllegalStateTransition):
        db.migrate_session_state("e1", STATE_FINALIZED_HEALTHY, now_stamps())
    db.close()


def test_gate_allows_registered_act_producer(tmp_path, ros_clock):
    cfg = _cfg(tmp_path)
    session = HILRecordingSession(cfg)
    session.recover_interrupted()
    snap = session.create(
        RecordRequest(
            episode_id="ep-act",
            task_id="t",
            control_profile="act",
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=0.05,
        )
    )
    assert snap.session_state == STATE_PREPARING
    session.register_producer("act_runner", os.getpid(), kind="policy")
    snap = session.start("ep-act")
    assert snap.session_state == STATE_RECORDING
    assert any(p["name"] == "act_runner" for p in snap.producers)
    session.request_stop("ep-act", reason="success")
    final = session.wait_finalized("ep-act", timeout_s=10.0)
    assert final.session_state in (
        STATE_FINALIZED_HEALTHY,
        "Failed(self_check)",
        STATE_FAILED_RECORD,
    )
    session.close()


def test_full_dry_run_publish_path(tmp_path, ros_clock):
    cfg = _cfg(tmp_path)
    session = HILRecordingSession(cfg)
    session.recover_interrupted()
    eid = "ep-ok"
    session.create(
        RecordRequest(
            episode_id=eid,
            task_id="pick",
            control_profile="act",
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=0.05,
        )
    )
    session.register_producer("act_runner", os.getpid(), "policy")
    session.start(eid)

    writer = HILReplayWriter(tmp_path, "unused", staging_dir=cfg.staging_dir(eid))
    for step in range(3):
        stamps = now_stamps()
        rec = TransitionRecord(
            experiment_id="unused",
            episode_id=eid,
            step_id=step,
            timestamp=stamps.wall_time_ns / 1e9,
            action=[0.0] * 16,
            reward=0.0,
            reward_source="none",
            terminated=step == 2,
            truncated=False,
            fault_code="NONE",
            is_intervention=False,
            action_clipped=False,
            extras={
                "intervention_mask": [0] * 16,
                "intervention_segment_id": 0,
                "intervention_segment_step": 0,
                "policy_action": [0.0] * 16,
                "stamps": stamps.to_dict(),
            },
        )
        obs = {
            "observation.state": np.zeros(16, dtype=np.float32),
            "observation.images.head_cam_h": np.zeros((3, 4, 5), dtype=np.uint8),
        }
        writer.log_transition(rec, observation=obs, next_observation=obs)
        session.update_transition(
            {
                "step_id": step,
                "stamps": stamps,
                "intervention_mask": [0] * 16,
                "intervention_segment_id": 0,
                "intervention_segment_step": 0,
                "policy_action": [0.0] * 16,
                "reward": 0.0,
                "fault_code": "NONE",
            }
        )
    writer.close()

    session.record_event(
        ResultEvent(
            episode_id=eid,
            event_type="success",
            source="test",
            stamps=now_stamps(),
        )
    )
    session.request_stop(eid, reason="success")
    # Immediate publish must fail / be rejected
    with pytest.raises(RuntimeError):
        session.publish_replay(eid)

    final = session.wait_finalized(eid, timeout_s=15.0)
    assert final.session_state == STATE_FINALIZED_HEALTHY, final
    assert final.result_type == "success"
    report = session.publish_replay(eid)
    # publish_replay is pending_review only; TRAIN_READY requires label+review.
    assert report.status == EXPORT_PENDING_REVIEW
    pending = cfg.pending_review_dir / eid
    assert (pending / "transitions.jsonl").exists()
    assert (pending / REVIEW_READY_MARKER).exists()
    assert (pending / "publish_manifest.json").exists()
    man = json.loads((pending / "publish_manifest.json").read_text())
    assert man["replay_schema_version"] == "hil-replay-v002"
    assert man["export_stage"] == EXPORT_PENDING_REVIEW
    assert not (cfg.accepted_replay_dir / eid).exists()
    # training path must not see staging leftovers with data
    staging = cfg.session_dir(eid) / "staging"
    assert not (staging / "transitions.jsonl").exists()
    session.close()


def test_estop_goes_to_quarantine(tmp_path, ros_clock):
    cfg = _cfg(tmp_path)
    session = HILRecordingSession(cfg)
    session.recover_interrupted()
    eid = "ep-estop"
    session.create(
        RecordRequest(
            episode_id=eid,
            task_id="pick",
            control_profile="act",
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=0.05,
        )
    )
    session.register_producer("act_runner", os.getpid(), "policy")
    session.start(eid)
    writer = HILReplayWriter(tmp_path, "x", staging_dir=cfg.staging_dir(eid))
    stamps = now_stamps()
    rec = TransitionRecord(
        experiment_id="x",
        episode_id=eid,
        step_id=0,
        timestamp=1.0,
        action=[0.0] * 16,
        reward=0.0,
        reward_source="none",
        terminated=True,
        truncated=False,
        fault_code="ESTOP",
        is_intervention=False,
        action_clipped=False,
        extras={
            "intervention_mask": [0] * 16,
            "intervention_segment_step": 0,
        },
    )
    obs = {"observation.state": np.zeros(16, dtype=np.float32)}
    writer.log_transition(rec, observation=obs, next_observation=obs)
    writer.close()
    session.update_transition(
        {
            "step_id": 0,
            "stamps": stamps,
            "intervention_mask": [0] * 16,
            "intervention_segment_step": 0,
        }
    )
    session.record_event(
        ResultEvent(episode_id=eid, event_type="estop", source="vr", stamps=now_stamps())
    )
    session.request_stop(eid, reason="estop")
    final = session.wait_finalized(eid, timeout_s=15.0)
    assert final.result_type == "estop"
    if final.session_state == STATE_FINALIZED_HEALTHY:
        report = session.publish_replay(eid)
        assert report.status == EXPORT_QUARANTINED
        assert (cfg.quarantine_dir / eid).exists()
        assert not (cfg.accepted_replay_dir / eid).exists()
    session.close()


def test_second_active_session_blocked(tmp_path, ros_clock):
    cfg = _cfg(tmp_path)
    s1 = HILRecordingSession(cfg)
    s1.recover_interrupted()
    s1.create(
        RecordRequest(
            episode_id="a",
            task_id="t",
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=0.05,
        )
    )
    s1.register_producer("act_runner", os.getpid(), "policy")
    s1.start("a")

    s2 = HILRecordingSession(cfg)
    s2.create(
        RecordRequest(
            episode_id="b",
            task_id="t",
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=0.05,
        )
    )
    s2.register_producer("act_runner", os.getpid(), "policy")
    with pytest.raises(RuntimeError, match="gate blocked"):
        s2.start("b")
    s1.request_stop("a", "abort")
    s1.wait_finalized("a", timeout_s=15.0)
    s1.close()
    s2.close()


def test_legacy_hil_replay_writer_still_works(tmp_path):
    writer = HILReplayWriter(tmp_path, "hil")
    record = TransitionRecord(
        experiment_id="hil",
        episode_id="episode-1",
        step_id=1,
        timestamp=1.0,
        action=[0.0] * 16,
        reward=1.0,
        reward_source="manual_success",
        terminated=True,
        truncated=False,
        fault_code="NONE",
        is_intervention=True,
        action_clipped=False,
    )
    obs = {
        "observation.state": np.zeros(16, dtype=np.float32),
        "observation.images.head_cam_h": np.zeros((3, 4, 5), dtype=np.uint8),
    }
    writer.log_transition(record, observation=obs, next_observation=obs)
    writer.close()
    root = tmp_path / "hil" / "replay"
    assert (root / "episodes" / "episode-1" / "transitions.jsonl").exists()
