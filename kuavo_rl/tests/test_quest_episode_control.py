"""C1 unit tests: stick edge / calibration / exclusive gate (no live ROS)."""

from __future__ import annotations

from pathlib import Path

from kuavo_rl.quest_episode_control import (
    ButtonChordDetector,
    ModifierStickDetector,
    MockQuestEpisodeControlEventSource,
    StickAxisCalibration,
    StickEdgeDetector,
    infer_calibration_from_samples,
    load_stick_calibration,
    save_stick_calibration,
    verify_right_stick_exclusive,
)


def test_stick_debounce_and_rearm():
    clock = {"t": 0.0}

    def _now():
        return clock["t"]

    det = StickEdgeDetector(
        trigger_threshold=0.8,
        rearm_neutral_threshold=0.2,
        debounce_s=0.25,
        _clock=_now,
    )
    assert det.update(0.9, 0.0) == "right_stick_right"
    clock["t"] = 0.1
    assert det.update(0.0, 0.0) is None
    assert det.update(0.9, 0.0) is None  # debounce
    clock["t"] = 0.5
    assert det.update(0.9, 0.0) == "right_stick_right"


def test_calibration_invert_and_infer(tmp_path):
    # Raw tip-right is actually -x (inverted hardware)
    cal = infer_calibration_from_samples(
        tip_right=(-0.95, 0.05),
        tip_down=(0.02, 0.92),  # tip down reads as +y → need invert_y
        calibrated_by="tester",
    )
    assert cal.calibrated
    assert cal.apply(-0.95, 0.05)[0] > 0.5
    assert cal.apply(0.02, 0.92)[1] < -0.5

    path = tmp_path / "stick.yaml"
    save_stick_calibration(cal, path)
    loaded = load_stick_calibration(path)
    assert loaded.invert_right_x == cal.invert_right_x
    assert loaded.calibrated is True


def test_exclusive_gate_blocks_uncalibrated_live():
    cal = StickAxisCalibration(calibrated=False)
    report = verify_right_stick_exclusive(
        calibration=cal,
        require_ack=True,
        require_calibration=True,
        for_live_collect=True,
        ros_param_get=lambda _k, d: False,
        rosnode_list=lambda: [],
        topic_present=lambda _t: True,
    )
    assert report.status == "Block"
    assert "stick_axis_not_calibrated" in report.reasons
    assert "collection_mode_ack_missing" in report.reasons


def test_exclusive_gate_pass_with_ack_and_calibration():
    cal = StickAxisCalibration(calibrated=True)
    report = verify_right_stick_exclusive(
        calibration=cal,
        require_ack=True,
        require_calibration=True,
        for_live_collect=True,
        ros_param_get=lambda _k, d: True,
        rosnode_list=lambda: ["/some_ok_node"],
        topic_present=lambda _t: True,
    )
    assert report.status == "Pass"
    assert report.collection_mode_ack is True


def test_conflicting_consumer_without_ack_blocks():
    cal = StickAxisCalibration(calibrated=True)
    report = verify_right_stick_exclusive(
        calibration=cal,
        require_ack=False,
        require_calibration=True,
        for_live_collect=True,
        ros_param_get=lambda _k, d: False,
        rosnode_list=lambda: ["/control_woosh"],
        topic_present=lambda _t: True,
    )
    assert report.status == "Block"
    assert any("right_stick_consumers_active" in r for r in report.reasons)


def test_mock_source_never_publishes_actions():
    src = MockQuestEpisodeControlEventSource()
    src.start()
    assert src.publishes_robot_actions() is False
    ev = src.push_axes(0.95, 0.0)
    assert ev is not None
    assert ev.event_type == "right_stick_right"
    assert src.poll() is not None
    src.close()


def test_detector_with_calibration_path(tmp_path: Path):
    cal = StickAxisCalibration(invert_right_x=True, calibrated=True)
    path = tmp_path / "o.json"
    save_stick_calibration(cal, path)
    loaded = load_stick_calibration(path)
    det = StickEdgeDetector(calibration=loaded, debounce_s=0.0)
    # raw -0.9 becomes +0.9 after invert → right
    assert det.update(-0.95, 0.0) == "right_stick_right"


def test_y_stick_modifier_gates_axes():
    det = ModifierStickDetector(
        stick=StickEdgeDetector(trigger_threshold=0.8, rearm_neutral_threshold=0.2, debounce_s=0.0)
    )
    # Without Y: stick ignored
    assert det.update(y_mod=False, right_x=0.95, right_y=0.0) is None
    # Hold Y + tip right
    assert det.update(y_mod=True, right_x=0.95, right_y=0.0) == "right_stick_right"
    assert det.update(y_mod=True, right_x=0.95, right_y=0.0) is None  # held
    # Release Y rearms; without Y still ignored
    assert det.update(y_mod=False, right_x=0.0, right_y=0.0) is None
    assert det.update(y_mod=False, right_x=-0.95, right_y=0.0) is None
    # Hold Y + left / down
    assert det.update(y_mod=True, right_x=-0.95, right_y=0.0) == "right_stick_left"
    assert det.update(y_mod=False, right_x=0.0, right_y=0.0) is None
    assert det.update(y_mod=True, right_x=0.0, right_y=-0.95) == "right_stick_down"


def test_y_chord_early_end_rerecord_and_complete():
    clock = {"t": 0.0}

    def _now():
        return clock["t"]

    det = ButtonChordDetector(long_press_s=1.0, debounce_s=0.0, _clock=_now)

    # Y+A short → early_end / start
    assert det.update(y_mod=True, a_key=True, x_key=False) is None
    clock["t"] = 0.2
    assert det.update(y_mod=True, a_key=False, x_key=False) == "right_stick_right"
    # release all to rearm
    clock["t"] = 0.3
    assert det.update(y_mod=False, a_key=False, x_key=False) is None

    # Y+X → rerecord
    clock["t"] = 0.4
    assert det.update(y_mod=True, a_key=False, x_key=True) == "right_stick_left"
    clock["t"] = 0.5
    assert det.update(y_mod=False, a_key=False, x_key=False) is None

    # Y+A long → collection_complete
    clock["t"] = 1.0
    assert det.update(y_mod=True, a_key=True, x_key=False) is None
    clock["t"] = 2.1
    assert det.update(y_mod=True, a_key=True, x_key=False) == "right_stick_down"
