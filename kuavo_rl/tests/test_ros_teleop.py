from types import SimpleNamespace

import numpy as np

from kuavo_rl.ros_teleop import RosTeleopAdapter, RosTeleopConfig


def _joy(*, left=0.0, right=0.0, success=False):
    return SimpleNamespace(
        left_grip=left,
        right_grip=right,
        left_trigger=0.0,
        right_trigger=0.0,
        left_first_button_pressed=False,
        left_second_button_pressed=False,
        right_first_button_pressed=False,
        right_second_button_pressed=success,
    )


def test_ros_teleop_grip_activates_each_arm_independently_and_converts_degrees():
    adapter = RosTeleopAdapter(RosTeleopConfig(success_button="right_second_button_pressed"))
    adapter._arm_callback(SimpleNamespace(position=[180.0] * 14))
    adapter._joy_callback(_joy(left=0.9, right=0.0, success=True))
    event = adapter.poll()
    assert event.is_intervention is True
    np.testing.assert_allclose(event.action[:7], np.pi, atol=1e-6)
    np.testing.assert_allclose(event.action[8:15], 0.0, atol=1e-6)
    assert event.intervention_mask[:7].all()
    assert not event.intervention_mask[8:15].any()
    adapter._joy_callback(_joy(left=0.0, right=0.9, success=True))
    event = adapter.poll()
    assert event.is_intervention is True
    assert event.source == "quest3_ik"
    assert event.success is True
    np.testing.assert_allclose(event.action[0], np.pi, atol=1e-6)
    np.testing.assert_allclose(event.action[8], np.pi, atol=1e-6)
    assert not event.intervention_mask[:7].any()
    assert event.intervention_mask[8:15].all()


def test_ros_teleop_double_left_button_is_emergency_stop():
    adapter = RosTeleopAdapter()
    adapter._arm_callback(SimpleNamespace(position=[180.0] * 14))
    msg = _joy(left=1.0, right=1.0)
    msg.left_first_button_pressed = True
    msg.left_second_button_pressed = True
    adapter._joy_callback(msg)
    event = adapter.poll()
    assert event.stop is True
    assert event.is_intervention is False
    assert event.action is None


def test_ros_teleop_stale_arm_is_fail_safe():
    adapter = RosTeleopAdapter(RosTeleopConfig(teleop_timeout_s=0.0))
    adapter._arm_callback(SimpleNamespace(position=[0.0] * 14))
    adapter._joy_callback(_joy(left=1.0, right=1.0))
    event = adapter.poll()
    assert event.is_intervention is False
    assert event.action is None


def test_ros_teleop_rl100_records_arm_and_qiangnao_command_without_grip_deadman():
    adapter = RosTeleopAdapter(
        RosTeleopConfig(
            record_all_arm_commands=True,
            require_hand_command=True,
            hand_command_topic="/control_robot_hand_position",
        )
    )
    adapter._arm_callback(SimpleNamespace(position=[90.0] * 14))
    adapter._hand_command_callback(
        SimpleNamespace(
            left_hand_position=[25, 100, 25, 25, 25, 25],
            right_hand_position=[80, 100, 80, 80, 80, 80],
        )
    )
    adapter._joy_callback(_joy(left=0.0, right=0.0))

    status = adapter.command_stream_status()
    event = adapter.poll()

    assert status["ok"] is True
    assert event.is_intervention is True
    assert event.source == "quest3_command_stream"
    np.testing.assert_allclose(event.action[:7], np.pi / 2, atol=1e-6)
    np.testing.assert_allclose(event.action[8:15], np.pi / 2, atol=1e-6)
    assert event.action[7] == np.float32(0.25)
    assert event.action[15] == np.float32(0.8)
    assert event.intervention_mask.all()


def test_ros_teleop_rl100_requires_fresh_hand_command():
    adapter = RosTeleopAdapter(
        RosTeleopConfig(
            record_all_arm_commands=True,
            require_hand_command=True,
        )
    )
    adapter._arm_callback(SimpleNamespace(position=[0.0] * 14))
    assert adapter.command_stream_status()["ok"] is False
    event = adapter.poll()
    assert event.is_intervention is False
    assert event.action is None


def test_ros_teleop_qiangnao_accepts_ros_uint8_bytes():
    adapter = RosTeleopAdapter()
    adapter._hand_command_callback(
        SimpleNamespace(
            left_hand_position=bytes([40, 100, 40, 40, 40, 40]),
            right_hand_position=bytes([90, 100, 90, 90, 90, 90]),
        )
    )
    np.testing.assert_allclose(adapter._latest_hand, [0.4, 0.9], atol=1e-6)


def test_ros_teleop_unreserved_b_reward_gestures():
    clock = [0.0]
    adapter = RosTeleopAdapter(
        RosTeleopConfig(
            reward_button="right_second_button_pressed",
            reward_long_press_s=1.20,
        )
    )
    adapter._reward_clock = lambda: clock[0]

    def poll(pressed: bool):
        adapter._joy_callback(_joy(success=pressed))
        return adapter.poll()

    # Short press → success on release.
    poll(True)
    clock[0] = 0.10
    event = poll(False)
    assert event.success is True
    assert event.failure is False
    assert event.abort is False

    # Long press → failure while held (no success on release).
    adapter.reset()
    clock[0] = 2.0
    poll(True)
    clock[0] = 3.30
    event = poll(True)
    assert event.failure is True
    assert event.success is False
    assert event.abort is False
    clock[0] = 3.40
    event = poll(False)
    assert event.success is False
    assert event.failure is False
    assert event.abort is False

    # Double-click does not map to failure (second tap is another short press).
    adapter.reset()
    clock[0] = 4.0
    poll(True)
    clock[0] = 4.10
    poll(False)
    clock[0] = 4.30
    poll(True)
    clock[0] = 4.40
    event = poll(False)
    assert event.success is True
    assert event.failure is False
