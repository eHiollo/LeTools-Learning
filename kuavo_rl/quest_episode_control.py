"""Quest episode control (C1+): read-only ROS source, no robot actions.

Default control is **hold Y + right stick** (modifier gating)::

    Without Y  → stick stays available for waist / chassis (teleop)
    Hold Y     → stick left/right/down become episode control edges

Also available: ``quest_y_chord`` (Y+A / Y+X face buttons) and legacy
``quest_right_stick`` (stick-only, needs exclusive ack).

Consumes ``/quest_joystick_data`` and emits logical ``EpisodeControlEvent``s.
Never publishes arm/waist/chassis commands.

Kuavo Quest face-button convention (remappable)::

    left_first_button  = X
    left_second_button = Y   # collection modifier (unused by teleop table)
    right_first_button = A
    right_second_button = B  # reserved for success/failure/abort labels
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from kuavo_rl.hil_recording.models import EpisodeControlEvent
from kuavo_rl.hil_recording.timebase import now_stamps

DEFAULT_JOYSTICK_TOPIC = "/quest_joystick_data"
DEFAULT_ACK_PARAM = "/hil/collection_mode_ack"
DEFAULT_OVERRIDE_PATH = Path("configs/rl/local/stick_axis_override.yaml")

# Nodes known to consume right stick for waist/chassis (must be down or ack).
KNOWN_RIGHT_STICK_CONSUMER_NODES = (
    "control_woosh",
    "woosh_control",
    "kuavo_chassis_teleop",
    "quest_waist_control",
)


@dataclass
class StickAxisCalibration:
    """Machine-local axis remap; code never guesses Quest coordinate signs."""

    invert_right_x: bool = False
    invert_right_y: bool = False
    swap_xy: bool = False
    calibrated: bool = False
    calibrated_at_wall_ns: int | None = None
    calibrated_by: str | None = None
    notes: str = ""

    def apply(self, right_x: float, right_y: float) -> tuple[float, float]:
        x, y = float(right_x), float(right_y)
        if self.swap_xy:
            x, y = y, x
        if self.invert_right_x:
            x = -x
        if self.invert_right_y:
            y = -y
        return x, y

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "StickAxisCalibration":
        raw = raw or {}
        return cls(
            invert_right_x=bool(raw.get("invert_right_x", False)),
            invert_right_y=bool(raw.get("invert_right_y", False)),
            swap_xy=bool(raw.get("swap_xy", False)),
            calibrated=bool(raw.get("calibrated", False)),
            calibrated_at_wall_ns=raw.get("calibrated_at_wall_ns"),
            calibrated_by=raw.get("calibrated_by"),
            notes=str(raw.get("notes", "")),
        )


def load_stick_calibration(path: Path | str | None = None) -> StickAxisCalibration:
    path = Path(path or DEFAULT_OVERRIDE_PATH)
    if not path.exists():
        return StickAxisCalibration()
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML required to load stick override") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"invalid stick calibration: {path}")
    return StickAxisCalibration.from_dict(data)


def save_stick_calibration(
    cal: StickAxisCalibration,
    path: Path | str | None = None,
) -> Path:
    path = Path(path or DEFAULT_OVERRIDE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = cal.to_dict()
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("PyYAML required to save stick override") from exc
        path.write_text(
            yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    else:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


@dataclass
class StickEdgeDetector:
    """Edge trigger with debounce + re-arm after returning to neutral."""

    trigger_threshold: float = 0.80
    rearm_neutral_threshold: float = 0.20
    debounce_s: float = 0.25
    calibration: StickAxisCalibration = field(default_factory=StickAxisCalibration)
    armed: bool = True
    _last_fire_mono: float | None = None
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False)

    def update(self, right_x: float, right_y: float) -> str | None:
        """Return logical event or None. After calibration: +x right, +y up, -y down."""
        ax, ay = self.calibration.apply(right_x, right_y)
        magnitude = max(abs(ax), abs(ay))
        if magnitude <= self.rearm_neutral_threshold:
            self.armed = True
            return None
        if not self.armed or magnitude < self.trigger_threshold:
            return None
        now = float(self._clock())
        if self._last_fire_mono is not None and (now - self._last_fire_mono) < float(
            self.debounce_s
        ):
            return None
        if abs(ax) >= abs(ay):
            event = "right_stick_right" if ax > 0 else "right_stick_left"
        else:
            if ay < 0:
                event = "right_stick_down"
            else:
                return None  # up reserved as no-op in collection mode
        self.armed = False
        self._last_fire_mono = now
        return event


class EpisodeControlEventSource(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...
    def poll(self) -> EpisodeControlEvent | None: ...
    def publishes_robot_actions(self) -> bool: ...


@dataclass
class ExclusiveGateReport:
    status: str  # Pass | Block
    calibrated: bool
    collection_mode_ack: bool | None
    conflicting_nodes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    joystick_topic_present: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_right_stick_exclusive(
    *,
    calibration: StickAxisCalibration,
    require_ack: bool = True,
    require_calibration: bool = True,
    for_live_collect: bool = True,
    ack_param: str = DEFAULT_ACK_PARAM,
    known_consumers: tuple[str, ...] = KNOWN_RIGHT_STICK_CONSUMER_NODES,
    ros_param_get: Callable[[str, Any], Any] | None = None,
    rosnode_list: Callable[[], list[str]] | None = None,
    topic_present: Callable[[str], bool] | None = None,
    joystick_topic: str = DEFAULT_JOYSTICK_TOPIC,
) -> ExclusiveGateReport:
    """C1 Gate: calibration + exclusive stick ownership (no robot motion)."""
    reasons: list[str] = []
    conflicting: list[str] = []
    ack: bool | None = None
    joy_present: bool | None = None

    if require_calibration and for_live_collect and not calibration.calibrated:
        reasons.append("stick_axis_not_calibrated")

    if topic_present is not None:
        joy_present = bool(topic_present(joystick_topic))
        if for_live_collect and not joy_present:
            reasons.append(f"missing_topic:{joystick_topic}")

    if ros_param_get is not None:
        try:
            ack = bool(ros_param_get(ack_param, False))
        except Exception:  # noqa: BLE001
            ack = False
            reasons.append(f"ack_param_unreadable:{ack_param}")
    elif require_ack and for_live_collect:
        # Probe live ROS if available; otherwise Block when ack required.
        try:
            import rospy

            if rospy.core.is_initialized():
                ack = bool(rospy.get_param(ack_param, False))
            else:
                ack = None
                reasons.append("ros_not_initialized_for_ack")
        except Exception:  # noqa: BLE001
            ack = None
            reasons.append("ros_unavailable_for_ack")

    if require_ack and for_live_collect and ack is not True:
        reasons.append("collection_mode_ack_missing")

    nodes: list[str] = []
    if rosnode_list is not None:
        nodes = list(rosnode_list())
    elif for_live_collect:
        try:
            import rosnode

            nodes = list(rosnode.get_node_names())
        except Exception:  # noqa: BLE001
            nodes = []

    for name in nodes:
        short = name.rsplit("/", 1)[-1]
        for known in known_consumers:
            if known in short or known in name:
                conflicting.append(name)
                break
    if conflicting and ack is not True:
        reasons.append(f"right_stick_consumers_active:{','.join(conflicting)}")

    status = "Block" if reasons else "Pass"
    return ExclusiveGateReport(
        status=status,
        calibrated=bool(calibration.calibrated),
        collection_mode_ack=ack,
        conflicting_nodes=conflicting,
        reasons=reasons,
        joystick_topic_present=joy_present,
    )


@dataclass
class ModifierStickDetector:
    """Hold one face button (default Y) + right stick → episode events.

    Operator card::

        Hold Y + stick →  : start / early_end     (right_stick_right)
        Hold Y + stick ←  : rerecord              (right_stick_left)
        Hold Y + stick ↓  : collection_complete   (right_stick_down)
        Hold Y + stick ↑  : no-op
        Y released        : stick ignored here; teleop may use stick again

    Does not require stick exclusivity: teleop keeps stick when Y is up.
    ``X + stick`` waist master switch is untouched (different modifier).
    """

    mod_attr: str = "left_second_button_pressed"  # Y
    stick: StickEdgeDetector = field(default_factory=StickEdgeDetector)
    _mod_was_down: bool = False

    def update_from_msg(self, msg: Any) -> str | None:
        y = bool(getattr(msg, self.mod_attr, False))
        rx = float(getattr(msg, "right_x", 0.0))
        ry = float(getattr(msg, "right_y", 0.0))
        return self.update(y_mod=y, right_x=rx, right_y=ry)

    def update(self, *, y_mod: bool, right_x: float, right_y: float) -> str | None:
        if not y_mod:
            # Modifier up: rearm stick edge state and ignore axes.
            if self._mod_was_down:
                self.stick.armed = True
                self.stick._last_fire_mono = None
            self._mod_was_down = False
            return None
        self._mod_was_down = True
        return self.stick.update(right_x, right_y)


@dataclass
class ButtonChordDetector:
    """Y-modifier face-button chords (optional alternative to Y+stick).

    Operator card (does **not** use right stick / does **not** use B)::

        RESETTING / RECORDING:
          Y + A  short release  → start episode / early_end   (right_stick_right)
          Y + X  short          → rerecord                    (right_stick_left)
          Y + A  long (≥1.0s)   → collection_complete         (right_stick_down)

    Safe vs teleop table:
    - ``X + A`` remains teleop activate (different left button: X vs Y)
    - right stick free for waist / chassis rotate
    - ``B`` free for success/failure/abort gestures
    - triggers / grips unchanged
    """

    mod_attr: str = "left_second_button_pressed"  # Y
    early_attr: str = "right_first_button_pressed"  # A
    rerecord_attr: str = "left_first_button_pressed"  # X
    long_press_s: float = 1.0
    debounce_s: float = 0.25
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _ya_down_at: float | None = None
    _ya_long_fired: bool = False
    _yx_latched: bool = False
    _armed: bool = True
    _last_fire_mono: float | None = None

    def update_from_msg(self, msg: Any) -> str | None:
        y = bool(getattr(msg, self.mod_attr, False))
        a = bool(getattr(msg, self.early_attr, False))
        x = bool(getattr(msg, self.rerecord_attr, False))
        return self.update(y_mod=y, a_key=a, x_key=x)

    def update(self, *, y_mod: bool, a_key: bool, x_key: bool) -> str | None:
        now = float(self._clock())

        # Y+A short fires on chord break / full release before long threshold.
        if (
            self._armed
            and self._ya_down_at is not None
            and not self._ya_long_fired
            and not (y_mod and a_key)
        ):
            held = now - self._ya_down_at
            if held < float(self.long_press_s):
                if self._last_fire_mono is None or (now - self._last_fire_mono) >= float(
                    self.debounce_s
                ):
                    return self._fire("right_stick_right", now)
            self._ya_down_at = None

        # Rearm only after all chord keys released.
        if not y_mod and not a_key and not x_key:
            self._armed = True
            self._ya_down_at = None
            self._ya_long_fired = False
            self._yx_latched = False
            return None

        if not self._armed:
            return None
        if self._last_fire_mono is not None and (now - self._last_fire_mono) < float(
            self.debounce_s
        ):
            return None

        # Y+X rerecord: edge when both become pressed (A must be up).
        if y_mod and x_key and not a_key:
            if not self._yx_latched:
                self._yx_latched = True
                return self._fire("right_stick_left", now)
            return None

        # Y+A hold: arm timer; long press → collection_complete.
        if y_mod and a_key and not x_key:
            if self._ya_down_at is None:
                self._ya_down_at = now
                self._ya_long_fired = False
                return None
            held = now - self._ya_down_at
            if not self._ya_long_fired and held >= float(self.long_press_s):
                self._ya_long_fired = True
                return self._fire("right_stick_down", now)
            return None

        return None

    def _fire(self, event_type: str, now: float) -> str:
        self._armed = False
        self._last_fire_mono = now
        self._ya_down_at = None
        return event_type


class MockQuestEpisodeControlEventSource:
    """Inject raw (x,y) samples or logical events for unit tests (no ROS)."""

    def __init__(
        self,
        detector: StickEdgeDetector | None = None,
        *,
        source: str = "mock_quest",
    ):
        self.detector = detector or StickEdgeDetector()
        self.source = source
        self._queue: list[EpisodeControlEvent] = []
        self._started = False

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False

    def publishes_robot_actions(self) -> bool:
        return False

    def push_axes(self, right_x: float, right_y: float) -> EpisodeControlEvent | None:
        ev = self.detector.update(right_x, right_y)
        if ev is None:
            return None
        event = EpisodeControlEvent(ev, self.source, now_stamps())
        self._queue.append(event)
        return event

    def push_event(self, event_type: str) -> EpisodeControlEvent:
        event = EpisodeControlEvent(event_type, self.source, now_stamps())
        self._queue.append(event)
        return event

    def poll(self) -> EpisodeControlEvent | None:
        if not self._queue:
            return None
        return self._queue.pop(0)


class QuestEpisodeControlEventSource:
    """Subscribe to Quest joystick; emit episode events only (no action pubs)."""

    def __init__(
        self,
        *,
        topic: str = DEFAULT_JOYSTICK_TOPIC,
        mode: str = "quest_y_stick",
        detector: StickEdgeDetector | None = None,
        chord: ButtonChordDetector | None = None,
        mod_stick: ModifierStickDetector | None = None,
        calibration: StickAxisCalibration | None = None,
        source: str | None = None,
    ):
        self.topic = topic
        self.mode = mode
        self.source = source or mode
        cal = calibration or StickAxisCalibration()
        self.detector = detector or StickEdgeDetector(calibration=cal)
        if calibration is not None:
            self.detector.calibration = calibration
        self.chord = chord or ButtonChordDetector()
        self.mod_stick = mod_stick or ModifierStickDetector(
            stick=StickEdgeDetector(
                trigger_threshold=self.detector.trigger_threshold,
                rearm_neutral_threshold=self.detector.rearm_neutral_threshold,
                debounce_s=self.detector.debounce_s,
                calibration=cal,
            )
        )
        if calibration is not None:
            self.mod_stick.stick.calibration = calibration
        self._lock = threading.Lock()
        self._pending: EpisodeControlEvent | None = None
        self._latest_raw: tuple[float, float] | None = None
        self._latest_buttons: dict[str, bool] = {}
        self._ros: Any | None = None
        self._sub: Any | None = None
        self._joy_type: Any | None = None

    def publishes_robot_actions(self) -> bool:
        return False

    def start(self) -> None:
        if self._ros is not None:
            return
        try:
            import rospy
            from rospy.msg import AnyMsg
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("QuestEpisodeControlEventSource requires rospy") from exc
        self._ros = rospy
        try:
            from kuavo_msgs.msg import JoySticks

            self._joy_type = JoySticks
        except Exception:  # noqa: BLE001
            self._joy_type = None
        # AnyMsg avoids generated-message handshake issues (same as RosTeleopAdapter).
        self._sub = rospy.Subscriber(self.topic, AnyMsg, self._joy_callback, queue_size=1)

    def close(self) -> None:
        if self._sub is not None:
            try:
                self._sub.unregister()
            except Exception:  # noqa: BLE001
                pass
        self._sub = None
        self._ros = None
        self._joy_type = None

    def latest_raw_axes(self) -> tuple[float, float] | None:
        with self._lock:
            return self._latest_raw

    def poll(self) -> EpisodeControlEvent | None:
        with self._lock:
            ev = self._pending
            self._pending = None
            return ev

    def _joy_callback(self, msg: Any) -> None:
        joy = self._decode_msg(msg)
        right_x = float(getattr(joy, "right_x", 0.0))
        right_y = float(getattr(joy, "right_y", 0.0))
        with self._lock:
            self._latest_raw = (right_x, right_y)
            self._latest_buttons = {
                "Y": bool(getattr(joy, "left_second_button_pressed", False)),
                "X": bool(getattr(joy, "left_first_button_pressed", False)),
                "A": bool(getattr(joy, "right_first_button_pressed", False)),
                "B": bool(getattr(joy, "right_second_button_pressed", False)),
            }

        if self.mode == "quest_right_stick":
            event_type = self.detector.update(right_x, right_y)
        elif self.mode == "quest_y_chord":
            event_type = self.chord.update_from_msg(joy)
        else:
            # Default quest_y_stick: hold Y + tip right stick.
            event_type = self.mod_stick.update_from_msg(joy)

        if event_type is None:
            return
        event = EpisodeControlEvent(event_type, self.source, now_stamps())
        with self._lock:
            # Keep latest edge if consumer is slow; never coalesce into action.
            self._pending = event

    def _decode_msg(self, msg: Any) -> Any:
        if self._joy_type is not None and hasattr(msg, "_buff"):
            try:
                return self._joy_type().deserialize(msg._buff)
            except Exception:  # noqa: BLE001
                pass
        return msg

    def _decode_axes(self, msg: Any) -> tuple[float, float]:
        joy = self._decode_msg(msg)
        return float(getattr(joy, "right_x", 0.0)), float(getattr(joy, "right_y", 0.0))


def infer_calibration_from_samples(
    *,
    tip_right: tuple[float, float],
    tip_down: tuple[float, float],
    calibrated_by: str = "operator",
) -> StickAxisCalibration:
    """Infer invert/swap from operator tip-right and tip-down samples.

    Expects after remap: tip_right → +x dominant, tip_down → -y dominant.
    Tries the 8 discrete transforms and picks the first that satisfies both.
    """
    candidates = []
    for swap in (False, True):
        for ix in (False, True):
            for iy in (False, True):
                candidates.append(
                    StickAxisCalibration(
                        invert_right_x=ix,
                        invert_right_y=iy,
                        swap_xy=swap,
                    )
                )

    def _ok(cal: StickAxisCalibration) -> bool:
        rx, ry = cal.apply(*tip_right)
        dx, dy = cal.apply(*tip_down)
        right_ok = abs(rx) >= abs(ry) and rx > 0.5
        down_ok = abs(dy) >= abs(dx) and dy < -0.5
        return right_ok and down_ok

    for cal in candidates:
        if _ok(cal):
            cal.calibrated = True
            cal.calibrated_at_wall_ns = now_stamps().wall_time_ns
            cal.calibrated_by = calibrated_by
            cal.notes = "inferred_from_tip_right_and_tip_down"
            return cal
    raise RuntimeError(
        "cannot infer stick calibration from samples; "
        "check Quest mapping or provide manual override"
    )
