"""Optional bridge from Kuavo Quest3 IK output to HIL-SERL intervention events."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from kuavo_rl.contracts import ACTION_DIM
from kuavo_rl.teleop import TeleopAdapter, TeleopEvent


@dataclass
class RosTeleopConfig:
    joystick_topic: str = "/quest_joystick_data"
    arm_traj_topic: str = "/kuavo_arm_traj"
    teleop_timeout_s: float = 0.20
    grip_threshold: float = 0.80
    success_button: str | None = None
    failure_button: str | None = None
    abort_button: str | None = None
    abort_buttons: tuple[str, ...] | None = None
    reward_button: str | None = None
    reward_double_press_s: float = 0.35
    reward_long_press_s: float = 1.20


class RosTeleopAdapter(TeleopAdapter):
    """Adapt Quest3 IK output to the canonical 16-D intervention action.

    Kuavo's ``/kuavo_arm_traj`` contains 14 arm positions in degrees. The
    canonical action is ``[L7, left_claw, R7, right_claw]`` in radians and
    normalized claw units. Claws are held from the reference action because
    hand message layouts differ between qiangnao and claw hardware.
    """

    def __init__(self, config: RosTeleopConfig | None = None):
        super().__init__()
        self.config = config or RosTeleopConfig()
        self._latest_joy: Any | None = None
        self._latest_arm: np.ndarray | None = None
        self._latest_arm_time = 0.0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._ros: Any | None = None
        self._joy_type: Any | None = None
        self._subs: list[Any] = []
        self._reward_clock = time.monotonic
        self._reward_pressed = False
        self._reward_pressed_at = 0.0
        self._reward_last_release = 0.0
        self._reward_click_pending = False
        self._reward_long_fired = False

    def start(self) -> None:
        if self._ros is not None:
            return
        try:
            import rospy
            from rospy.msg import AnyMsg
            from kuavo_msgs.msg import JoySticks
            from sensor_msgs.msg import JointState
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "ROS teleop requires rospy, sensor_msgs and kuavo_msgs"
            ) from exc
        self._ros = rospy
        self._joy_type = JoySticks
        self._subs = [
            # AnyMsg avoids a ROS-Python generated-message handshake issue
            # seen in the deployed Kuavo workspace. Decode below and fail safe.
            rospy.Subscriber(self.config.joystick_topic, AnyMsg, self._joy_callback, queue_size=1),
            rospy.Subscriber(self.config.arm_traj_topic, JointState, self._arm_callback, queue_size=1),
        ]

    def close(self) -> None:
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:  # noqa: BLE001
                pass
        self._subs = []
        self._joy_type = None
        self._ros = None

    def reset(self) -> None:
        super().reset()
        self._reset_reward_gesture()
        # Keep latest ROS samples; freshness gating handles stale data.

    def _reset_reward_gesture(self) -> None:
        self._reward_pressed = False
        self._reward_pressed_at = 0.0
        self._reward_last_release = 0.0
        self._reward_click_pending = False
        self._reward_long_fired = False

    def set_reference_action(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (ACTION_DIM,):
            raise ValueError(f"reference action must be {ACTION_DIM}-D")
        self._last_action = a.copy()

    def set_claw_values(self, left: float, right: float) -> None:
        self._last_action[7] = float(np.clip(left, 0.0, 1.0))
        self._last_action[15] = float(np.clip(right, 0.0, 1.0))

    def _joy_callback(self, msg: Any) -> None:
        if hasattr(msg, "_buff") and self._joy_type is not None:
            try:
                msg = self._joy_type().deserialize(msg._buff)
            except Exception:  # noqa: BLE001
                return
        self._latest_joy = msg

    def _arm_callback(self, msg: Any) -> None:
        position = np.asarray(getattr(msg, "position", ()), dtype=np.float32).reshape(-1)
        if position.size != 14 or not np.all(np.isfinite(position)):
            return
        self._latest_arm = np.deg2rad(position).astype(np.float32)
        self._latest_arm_time = self._now()

    def _now(self) -> float:
        if self._ros is not None:
            try:
                return float(self._ros.get_time())
            except Exception:  # noqa: BLE001
                pass
        return time.time()

    def _button(self, msg: Any, name: str | None) -> bool:
        return bool(name and getattr(msg, name, False))

    def _buttons(self, msg: Any, names: tuple[str, ...] | None) -> bool:
        return bool(names) and all(bool(getattr(msg, name, False)) for name in names)

    def _reward_button_event(self, pressed: bool) -> tuple[bool, bool, bool]:
        """Return (success, failure, abort) for a single unreserved button.

        A single click becomes success after the double-click window, a double
        click becomes failure, and a long press becomes abort.  Delaying the
        single click prevents a double click from being recorded as success.
        """
        if not self.config.reward_button:
            return False, False, False
        now = float(self._reward_clock())
        success = failure = abort = False
        if pressed and not self._reward_pressed:
            self._reward_pressed = True
            self._reward_pressed_at = now
            self._reward_long_fired = False
        elif pressed and self._reward_pressed:
            if not self._reward_long_fired and now - self._reward_pressed_at >= self.config.reward_long_press_s:
                abort = True
                self._reward_long_fired = True
                self._reward_click_pending = False
        elif not pressed and self._reward_pressed:
            self._reward_pressed = False
            held_s = now - self._reward_pressed_at
            if self._reward_long_fired or held_s >= self.config.reward_long_press_s:
                abort = not self._reward_long_fired
                self._reward_long_fired = False
                self._reward_click_pending = False
            elif self._reward_click_pending and now - self._reward_last_release <= self.config.reward_double_press_s:
                failure = True
                self._reward_click_pending = False
            else:
                self._reward_click_pending = True
                self._reward_last_release = now
        elif not pressed and self._reward_click_pending and now - self._reward_last_release >= self.config.reward_double_press_s:
            success = True
            self._reward_click_pending = False
        return success, failure, abort

    def _active_sides(self, msg: Any) -> tuple[bool, bool]:
        left = float(getattr(msg, "left_grip", 0.0)) > self.config.grip_threshold
        right = float(getattr(msg, "right_grip", 0.0)) > self.config.grip_threshold
        return left, right

    def poll(self) -> TeleopEvent:
        msg = self._latest_joy
        age = self._now() - self._latest_arm_time if self._latest_arm_time else float("inf")
        fresh = self._latest_arm is not None and age <= self.config.teleop_timeout_s
        left_active, right_active = self._active_sides(msg) if msg is not None else (False, False)
        double_button_stop = bool(
            msg is not None
            and getattr(msg, "left_first_button_pressed", False)
            and getattr(msg, "left_second_button_pressed", False)
        )
        # Emergency stop has priority over ordinary intervention.
        active = (left_active or right_active) and not double_button_stop
        action = self._last_action.copy()
        if self._latest_arm is not None:
            if left_active:
                action[:7] = self._latest_arm[:7]
            if right_active:
                action[8:15] = self._latest_arm[7:]
            self._last_action = action.copy()
        active_and_fresh = bool(active and fresh)
        intervention_mask = np.zeros(ACTION_DIM, dtype=bool)
        if active_and_fresh:
            if left_active:
                intervention_mask[:7] = True
            if right_active:
                intervention_mask[8:15] = True
        gesture_success, gesture_failure, gesture_abort = self._reward_button_event(
            self._button(msg, self.config.reward_button) if msg is not None else False
        )
        abort = bool(
            gesture_abort
            or (
                msg is not None
                and (
                    self._button(msg, self.config.abort_button)
                    or self._buttons(msg, self.config.abort_buttons)
                )
            )
        )
        return TeleopEvent(
            action=action if active_and_fresh else None,
            is_intervention=active_and_fresh,
            intervention_mask=intervention_mask if active_and_fresh else None,
            success=(gesture_success or self._button(msg, self.config.success_button)) and not abort if msg is not None else gesture_success and not abort,
            failure=(gesture_failure or self._button(msg, self.config.failure_button)) and not abort if msg is not None else gesture_failure and not abort,
            abort=abort,
            # The original Kuavo Quest3 FSM uses both left buttons for stop.
            stop=double_button_stop,
            deadman=active_and_fresh,
            source="quest3_ik" if active_and_fresh else "none",
            age_s=float(age) if np.isfinite(age) else None,
        )
