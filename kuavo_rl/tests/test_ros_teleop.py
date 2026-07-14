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


def test_ros_teleop_unreserved_b_reward_gestures():
    clock = [0.0]
    adapter = RosTeleopAdapter(
        RosTeleopConfig(
            reward_button="right_second_button_pressed",
            reward_double_press_s=0.35,
            reward_long_press_s=1.20,
        )
    )
    adapter._reward_clock = lambda: clock[0]

    def poll(pressed: bool):
        adapter._joy_callback(_joy(success=pressed))
        return adapter.poll()

    # Single click emits success only after the double-click window expires.
    poll(True)
    clock[0] = 0.10
    assert poll(False).success is False
    clock[0] = 0.46
    assert poll(False).success is True

    adapter.reset()
    clock[0] = 1.0
    poll(True)
    clock[0] = 1.10
    poll(False)
    clock[0] = 1.20
    poll(True)
    clock[0] = 1.30
    event = poll(False)
    assert event.failure is True
    assert event.success is False

    adapter.reset()
    clock[0] = 2.0
    poll(True)
    clock[0] = 3.21
    event = poll(True)
    assert event.abort is True
    assert event.success is False
    assert event.failure is False
