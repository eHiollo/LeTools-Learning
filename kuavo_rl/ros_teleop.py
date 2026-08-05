"""Optional bridge from Kuavo Quest3 IK output to HIL-SERL intervention events."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from kuavo_rl.contracts import ACTION_DIM
from kuavo_rl.rl100_zarr.topic_native import TopicCommandCache, TopicCommandSnapshot
from kuavo_rl.teleop import TeleopAdapter, TeleopEvent


@dataclass
class RosTeleopConfig:
    joystick_topic: str = "/quest_joystick_data"
    arm_traj_topic: str = "/kuavo_arm_traj"
    hand_command_topic: str | None = None
    teleop_timeout_s: float = 0.20
    hand_command_timeout_s: float = 0.20
    grip_threshold: float = 0.80
    # RL-100 demonstrations must record the command stream even when Quest grip
    # is being used to control the hand rather than as an arm deadman.
    record_all_arm_commands: bool = False
    require_hand_command: bool = False
    # Qiangnao/Revo hand order is [thumb, thumb_aux, index, middle, ring, pinky].
    # The existing 1-DoF Kuavo policy contract uses thumb as the grasp scalar.
    qiangnao_scalar_index: int = 0
    success_button: str | None = None
    failure_button: str | None = None
    abort_button: str | None = None
    abort_buttons: tuple[str, ...] | None = None
    reward_button: str | None = None
    reward_double_press_s: float = 0.70  # unused; kept for config compat
    reward_long_press_s: float = 1.20
    # "button" = B short/long; "right_stick_ud" = stick down=success, up=failure.
    reward_gesture: str = "button"
    reward_stick_threshold: float = 0.80
    reward_stick_rearm_threshold: float = 0.20
    reward_stick_debounce_s: float = 0.25


class RosTeleopAdapter(TeleopAdapter):
    """Adapt Quest3 IK output to the canonical 16-D intervention action.

    Kuavo's ``/kuavo_arm_traj`` contains 14 arm positions in degrees. The
    canonical action is ``[L7, left_claw, R7, right_claw]`` in radians and
    normalized claw units. When ``hand_command_topic`` is configured, a
    Qiangnao ``robotHandPosition`` command is reduced to the same 1-DoF hand
    contract used by Kuavo deployment.
    """

    def __init__(self, config: RosTeleopConfig | None = None):
        super().__init__()
        self.config = config or RosTeleopConfig()
        self._latest_joy: Any | None = None
        self._latest_arm: np.ndarray | None = None
        self._latest_arm_time = 0.0
        self._latest_hand: np.ndarray | None = None
        self._latest_hand_time = 0.0
        self._last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self._ros: Any | None = None
        self._joy_type: Any | None = None
        self._subs: list[Any] = []
        self._reward_clock = time.monotonic
        self._reward_pressed = False
        self._reward_pressed_at = 0.0
        self._reward_long_fired = False
        self._label_gestures_enabled = True
        self._reward_stick_armed = True
        self._reward_stick_last_fire: float | None = None
        # RL-100 uses the complete command topics.  The cache is inactive
        # until begin_topic_native_episode(), so generic HIL use is unchanged.
        self._topic_native_cache = TopicCommandCache()

    def start(self) -> None:
        if self._ros is not None:
            return
        try:
            import rospy
            from rospy.msg import AnyMsg
            from kuavo_msgs.msg import JoySticks, robotHandPosition
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
        if self.config.hand_command_topic:
            self._subs.append(
                rospy.Subscriber(
                    self.config.hand_command_topic,
                    robotHandPosition,
                    self._hand_command_callback,
                    queue_size=1,
                )
            )

    def close(self) -> None:
        self._topic_native_cache.end_episode()
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
        self._reward_long_fired = False
        self._reward_stick_armed = True
        self._reward_stick_last_fire = None

    def set_label_gestures_enabled(self, enabled: bool) -> None:
        """Disable success/failure gestures during RESET without dropping teleop."""
        self._label_gestures_enabled = bool(enabled)
        if not enabled:
            self._reset_reward_gesture()

    def set_reference_action(self, action: np.ndarray) -> None:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (ACTION_DIM,):
            raise ValueError(f"reference action must be {ACTION_DIM}-D")
        self._last_action = a.copy()

    def last_gripper_command(self) -> tuple[float, float]:
        """Last hand-command gripper values, hold-last when stream is stale.

        /dexhand/state sensor readings are unreliable for RL-100 collection
        (sparse / low-rate), so the gripper columns of state are taken from
        the operator's hand command (/control_robot_hand_position) instead.
        _latest_hand retains the last received command until a new one arrives,
        which gives hold-last behaviour across brief demand-driven gaps.
        """
        if self._latest_hand is None:
            return 0.0, 0.0
        return float(self._latest_hand[0]), float(self._latest_hand[1])

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
        self._topic_native_cache.update_arm(msg)

    def _hand_command_callback(self, msg: Any) -> None:
        def _positions(value: Any) -> np.ndarray:
            # ROS uint8[] is commonly exposed as bytes under Python 3.
            if isinstance(value, (bytes, bytearray, memoryview)):
                return np.frombuffer(value, dtype=np.uint8).astype(np.float32)
            return np.asarray(value, dtype=np.float32).reshape(-1)

        left = _positions(getattr(msg, "left_hand_position", ()))
        right = _positions(getattr(msg, "right_hand_position", ()))
        idx = int(self.config.qiangnao_scalar_index)
        if idx < 0 or left.size <= idx or right.size <= idx:
            return
        grasp = np.asarray([left[idx], right[idx]], dtype=np.float32)
        if not np.all(np.isfinite(grasp)):
            return
        self._latest_hand = np.clip(grasp / 100.0, 0.0, 1.0).astype(np.float32)
        self._latest_hand_time = self._now()
        self._topic_native_cache.update_hand(msg)

    def begin_topic_native_episode(self, measured_arm14_rad: np.ndarray) -> int:
        """Reset the RL-100 command cache at the operator record boundary."""
        generation = self._topic_native_cache.generation + 1
        self._topic_native_cache.begin_episode(
            measured_arm14_rad,
            generation=generation,
            record_start_received_at=time.monotonic(),
        )
        return generation

    def end_topic_native_episode(self) -> None:
        self._topic_native_cache.end_episode()

    @property
    def topic_native_cache(self) -> TopicCommandCache:
        return self._topic_native_cache

    def topic_native_snapshot(self, sample_cutoff_received_at: float | None = None) -> TopicCommandSnapshot:
        return self._topic_native_cache.snapshot_and_clear_changed(sample_cutoff_received_at)

    def _now(self) -> float:
        if self._ros is not None:
            try:
                return float(self._ros.get_time())
            except Exception:  # noqa: BLE001
                pass
        return time.time()

    def command_stream_status(self) -> dict[str, Any]:
        """Return freshness diagnostics for collection preflight."""
        now = self._now()
        arm_age = now - self._latest_arm_time if self._latest_arm_time else float("inf")
        hand_age = now - self._latest_hand_time if self._latest_hand_time else float("inf")
        arm_ready = self._latest_arm is not None and arm_age <= self.config.teleop_timeout_s
        hand_ready = self._latest_hand is not None and hand_age <= self.config.hand_command_timeout_s
        return {
            "ok": bool(arm_ready and (hand_ready if self.config.require_hand_command else True)),
            "arm_topic": self.config.arm_traj_topic,
            "arm_ready": bool(arm_ready),
            "arm_age_s": float(arm_age) if np.isfinite(arm_age) else None,
            "hand_topic": self.config.hand_command_topic,
            "hand_ready": bool(hand_ready),
            "hand_age_s": float(hand_age) if np.isfinite(hand_age) else None,
        }

    def _button(self, msg: Any, name: str | None) -> bool:
        return bool(name and getattr(msg, name, False))

    def _buttons(self, msg: Any, names: tuple[str, ...] | None) -> bool:
        return bool(names) and all(bool(getattr(msg, name, False)) for name in names)

    def _reward_button_event(self, pressed: bool) -> tuple[bool, bool, bool]:
        """Return (success, failure, abort) for a single unreserved button.

        Short press (release before long threshold) → success.
        Long press (held ≥ reward_long_press_s) → failure.
        Double-click is not used (avoids mis-labeling).  abort is always False
        here (use left dual buttons or abort_button config for emergency stop).
        """
        if not self.config.reward_button:
            return False, False, False
        now = float(self._reward_clock())
        success = failure = False
        if pressed and not self._reward_pressed:
            self._reward_pressed = True
            self._reward_pressed_at = now
            self._reward_long_fired = False
        elif pressed and self._reward_pressed:
            if not self._reward_long_fired and now - self._reward_pressed_at >= self.config.reward_long_press_s:
                failure = True
                self._reward_long_fired = True
        elif not pressed and self._reward_pressed:
            self._reward_pressed = False
            held_s = now - self._reward_pressed_at
            if not self._reward_long_fired and held_s < self.config.reward_long_press_s:
                success = True
            self._reward_long_fired = False
        return success, failure, False

    def _reward_stick_event(self, msg: Any | None) -> tuple[bool, bool, bool]:
        """Right stick tip: down=success, up=failure (edge + rearm to neutral)."""
        if msg is None:
            return False, False, False
        ay = float(getattr(msg, "right_y", 0.0))
        ax = float(getattr(msg, "right_x", 0.0))
        mag = max(abs(ax), abs(ay))
        rearm = float(self.config.reward_stick_rearm_threshold)
        trig = float(self.config.reward_stick_threshold)
        if mag <= rearm:
            self._reward_stick_armed = True
            return False, False, False
        if not self._reward_stick_armed or mag < trig:
            return False, False, False
        if abs(ay) < abs(ax):
            return False, False, False
        now = float(self._reward_clock())
        if (
            self._reward_stick_last_fire is not None
            and (now - self._reward_stick_last_fire)
            < float(self.config.reward_stick_debounce_s)
        ):
            return False, False, False
        self._reward_stick_armed = False
        self._reward_stick_last_fire = now
        if ay < 0.0:
            return True, False, False
        return False, True, False

    def _reward_gesture_event(self, msg: Any | None) -> tuple[bool, bool, bool]:
        if not self._label_gestures_enabled:
            return False, False, False
        mode = str(self.config.reward_gesture or "button").strip().lower()
        if mode in {"right_stick_ud", "right_stick", "stick_ud"}:
            return self._reward_stick_event(msg)
        return self._reward_button_event(
            self._button(msg, self.config.reward_button) if msg is not None else False
        )

    def _active_sides(self, msg: Any) -> tuple[bool, bool]:
        left = float(getattr(msg, "left_grip", 0.0)) > self.config.grip_threshold
        right = float(getattr(msg, "right_grip", 0.0)) > self.config.grip_threshold
        return left, right

    def poll(self) -> TeleopEvent:
        msg = self._latest_joy
        now = self._now()
        arm_age = now - self._latest_arm_time if self._latest_arm_time else float("inf")
        arm_fresh = self._latest_arm is not None and arm_age <= self.config.teleop_timeout_s
        hand_age = now - self._latest_hand_time if self._latest_hand_time else float("inf")
        hand_fresh = (
            self._latest_hand is not None
            and hand_age <= self.config.hand_command_timeout_s
        )
        if self.config.record_all_arm_commands:
            left_active, right_active = True, True
        else:
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
        if self._latest_hand is not None:
            action[7] = self._latest_hand[0]
            action[15] = self._latest_hand[1]
        self._last_action = action.copy()
        streams_fresh = arm_fresh and (
            hand_fresh if self.config.require_hand_command else True
        )
        active_and_fresh = bool(active and streams_fresh)
        intervention_mask = np.zeros(ACTION_DIM, dtype=bool)
        if active_and_fresh:
            if left_active:
                intervention_mask[:7] = True
            if right_active:
                intervention_mask[8:15] = True
            if hand_fresh:
                intervention_mask[7] = True
                intervention_mask[15] = True
        gesture_success, gesture_failure, gesture_abort = self._reward_gesture_event(msg)
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
            source=(
                "quest3_command_stream"
                if active_and_fresh and self.config.record_all_arm_commands
                else "quest3_ik" if active_and_fresh else "none"
            ),
            age_s=(
                float(max(arm_age, hand_age))
                if active_and_fresh and self.config.require_hand_command
                else float(arm_age) if np.isfinite(arm_age) else None
            ),
        )
