from types import SimpleNamespace

import numpy as np

from kuavo_rl.ros_teleop import RosTeleopAdapter, RosTeleopConfig


def _joy(*, left=0.0, right=0.0, success=False, right_x=0.0, right_y=0.0):
    return SimpleNamespace(
        left_grip=left,
        right_grip=right,
        left_trigger=0.0,
        right_trigger=0.0,
        left_first_button_pressed=False,
        left_second_button_pressed=False,
        right_first_button_pressed=False,
        right_second_button_pressed=success,
        right_x=right_x,
        right_y=right_y,
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


def test_ros_teleop_b_success_is_one_shot_do_not_double_poll():
    """Regression: live_collect used to poll() then env.step()→poll() again.

    B short-press success is edge-triggered and only survives one poll().
    The second poll in the same control tick ate the label → abort discard.
    """
    clock = [0.0]
    adapter = RosTeleopAdapter(
        RosTeleopConfig(
            reward_button="right_second_button_pressed",
            reward_long_press_s=0.80,
        )
    )
    adapter._reward_clock = lambda: clock[0]

    adapter._joy_callback(_joy(success=True))
    clock[0] = 0.05
    adapter.poll()  # press edge
    clock[0] = 0.10
    adapter._joy_callback(_joy(success=False))
    # First poll (as env.step would) gets success.
    event = adapter.poll()
    assert event.success is True
    # Second poll in the same tick sees nothing — the old live_collect bug.
    event2 = adapter.poll()
    assert event2.success is False
    assert event2.failure is False


def test_ros_teleop_right_stick_ud_labels():
    """右摇杆下推=success，上推=failure；需要回中 rearm + debounce。"""
    clock = [0.0]
    adapter = RosTeleopAdapter(
        RosTeleopConfig(
            reward_gesture="right_stick_ud",
            reward_stick_threshold=0.80,
            reward_stick_rearm_threshold=0.20,
            reward_stick_debounce_s=0.25,
        )
    )
    adapter._reward_clock = lambda: clock[0]

    def poll(right_y: float):
        adapter._joy_callback(_joy(right_y=right_y))
        return adapter.poll()

    # 下推 → success
    clock[0] = 1.0
    event = poll(-0.95)
    assert event.success is True
    assert event.failure is False

    # 不回中再次下推 → 不触发（未 rearm）
    clock[0] = 1.1
    event = poll(-0.95)
    assert event.success is False

    # 回中 rearm
    clock[0] = 1.2
    event = poll(0.0)
    assert event.success is False

    # 上推 → failure
    clock[0] = 1.3
    event = poll(0.95)
    assert event.failure is True
    assert event.success is False

    # 回中后立即下推，在 debounce 内 → 不触发
    clock[0] = 1.4
    poll(0.0)
    clock[0] = 1.45
    event = poll(-0.95)
    assert event.success is False

    # debounce 过后下推 → success
    clock[0] = 1.8
    event = poll(-0.95)
    assert event.success is True

    # RESET 阶段禁用标签手势
    adapter.set_label_gestures_enabled(False)
    clock[0] = 2.0
    poll(0.0)
    clock[0] = 2.1
    event = poll(-0.95)
    assert event.success is False
    assert event.failure is False
