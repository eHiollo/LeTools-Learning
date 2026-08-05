"""Safety-first synchronous runner for RL-100 real-robot inference."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np

from kuavo_rl.backend import BackendObservation, RobotBackend
from kuavo_rl.config import SafetyConfig
from kuavo_rl.contracts import (
    ACTION_DIM,
    FaultCode,
    RL100_ACTION_DIM,
    RL100_ARM_JOINT_NAMES,
    RL100_ARM_SLICE_RAW20,
    RL100_STATE_DIM,
    RL100_TOPIC_NATIVE_CONTRACT,
    STATE_DIM,
    validate_action_shape,
)
from kuavo_rl.ros_adapter import build_published_command
from kuavo_rl.safety import SafetyGate
from kuavo_rl.rl100_zarr.ros_depth import PointCloudSample
from kuavo_rl.rl100_topic_executor import RL100TopicActionScheduler


class DeployState(str, Enum):
    INIT = "INIT"
    PREFLIGHT = "PREFLIGHT"
    READY = "READY"
    SHADOW = "SHADOW"
    ARMED = "ARMED"
    RUNNING = "RUNNING"
    HOLD = "HOLD"
    FAULT = "FAULT"
    STOPPED = "STOPPED"


@dataclass(frozen=True)
class TimedRL100Observation:
    state16: np.ndarray
    point_cloud: np.ndarray
    state_stamp_s: float
    hand_stamp_s: float
    point_cloud_stamp_s: float
    received_at_s: float
    state_age_s: float
    hand_age_s: float
    point_cloud_age_s: float
    state_hand_skew_s: float
    max_camera_skew_s: float
    raw_joint_dim: int


@dataclass(frozen=True)
class RunnerLimits:
    state_max_age_s: float = 0.15
    hand_max_age_s: float = 0.15
    depth_max_age_s: float = 0.15
    max_state_hand_skew_s: float = 0.10
    max_camera_skew_s: float = 0.10
    max_state_cloud_skew_s: float = 0.10
    inference_timeout_s: float = 0.10
    max_arm_state_jump_rad: float = 0.05
    max_gripper_state_jump: float = 0.10
    max_consecutive_source_failures: int = 1


@dataclass(frozen=True)
class TickResult:
    state: DeployState
    published: bool
    fault_code: FaultCode
    reason: str
    record: dict[str, Any]


class PolicyLike(Protocol):
    info: Any

    def predict(self, point_cloud_history: np.ndarray, state_history: np.ndarray) -> np.ndarray: ...


class ObservationHistory:
    def __init__(self, n_obs_steps: int) -> None:
        if n_obs_steps < 1:
            raise ValueError("n_obs_steps must be >= 1")
        self.n_obs_steps = int(n_obs_steps)
        self._points: deque[np.ndarray] = deque(maxlen=n_obs_steps)
        self._states: deque[np.ndarray] = deque(maxlen=n_obs_steps)
        self.padded_on_start = False

    def clear(self) -> None:
        self._points.clear()
        self._states.clear()
        self.padded_on_start = False

    def append(self, point_cloud: np.ndarray, state16: np.ndarray) -> None:
        point = np.asarray(point_cloud, dtype=np.float32)
        state = np.asarray(state16, dtype=np.float32).reshape(-1)
        if point.shape != (1024, 3):
            raise ValueError(f"point cloud shape {point.shape} != (1024, 3)")
        if state.shape != (STATE_DIM,):
            raise ValueError(f"state shape {state.shape} != ({STATE_DIM},)")
        if not np.isfinite(point).all() or not np.isfinite(state).all():
            raise ValueError("non-finite observation history input")
        if not self._points:
            self._points.extend(point.copy() for _ in range(self.n_obs_steps))
            self._states.extend(state.copy() for _ in range(self.n_obs_steps))
            self.padded_on_start = self.n_obs_steps > 1
            return
        self._points.append(point.copy())
        self._states.append(state.copy())

    @property
    def ready(self) -> bool:
        return len(self._points) == self.n_obs_steps and len(self._states) == self.n_obs_steps

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.ready:
            raise RuntimeError("observation history is not ready")
        return np.stack(self._points, axis=0), np.stack(self._states, axis=0)


def limit_action_to_measured_state(
    action: np.ndarray,
    state16: np.ndarray,
    *,
    max_arm_jump_rad: float,
    max_gripper_jump: float,
) -> tuple[np.ndarray, bool]:
    """Soft-limit an absolute target relative to the latest measured state."""
    target = validate_action_shape(action).copy()
    state = np.asarray(state16, dtype=np.float32).reshape(-1)
    if state.shape != (STATE_DIM,) or not np.isfinite(state).all():
        raise ValueError("measured state must be finite 16-D")
    if max_arm_jump_rad <= 0 or max_gripper_jump <= 0:
        raise ValueError("measured-state jump limits must be > 0")
    limits = np.full(ACTION_DIM, float(max_arm_jump_rad), dtype=np.float32)
    limits[7] = limits[15] = float(max_gripper_jump)
    delta = target - state
    bounded_delta = np.clip(delta, -limits, limits)
    clipped = state + bounded_delta
    return clipped.astype(np.float32), not np.allclose(clipped, target)


class RL100RealRunner:
    """One synchronous inference tick per observation; no stale action replay."""

    def __init__(
        self,
        *,
        policy: PolicyLike,
        backend: RobotBackend,
        point_cloud_source: Callable[[], PointCloudSample],
        safety: SafetyGate,
        limits: RunnerLimits = RunnerLimits(),
        shadow_mode: bool = True,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.policy = policy
        self.backend = backend
        self.point_cloud_source = point_cloud_source
        self.safety = safety
        self.limits = limits
        self.shadow_mode = bool(shadow_mode)
        self.audit_sink = audit_sink
        self.history = ObservationHistory(int(policy.info.n_obs_steps))
        self.state = DeployState.INIT
        self._source_failures = 0
        self._hold_sent = False

    def preflight(self) -> None:
        self.state = DeployState.PREFLIGHT
        info = self.policy.info
        if int(info.state_dim) != STATE_DIM or int(info.action_dim) != ACTION_DIM:
            self._fault(FaultCode.ACTION_SHAPE, "checkpoint state/action dim is not 16")
            raise RuntimeError("RL-100 checkpoint contract mismatch")
        if int(info.point_count) != 1024 or int(info.point_dim) != 3:
            self._fault(FaultCode.ACTION_SHAPE, "checkpoint point cloud shape is not 1024x3")
            raise RuntimeError("RL-100 checkpoint point cloud mismatch")
        self.state = DeployState.READY

    def arm_live(self) -> None:
        if self.shadow_mode:
            raise RuntimeError("runner is configured shadow_mode=true")
        if self.state not in {DeployState.READY, DeployState.SHADOW}:
            raise RuntimeError(f"cannot arm live runner from {self.state}")
        self.state = DeployState.ARMED

    def _timed_observation(self, backend_obs: BackendObservation, cloud: PointCloudSample) -> TimedRL100Observation:
        extras = backend_obs.extras
        required = ("state_stamp_s", "hand_stamp_s", "state_age_s", "hand_age_s")
        missing = [key for key in required if key not in extras]
        if missing:
            raise RuntimeError(f"timestamp-aware robot observation missing {missing}")
        state = np.asarray(backend_obs.state, dtype=np.float32).reshape(-1)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"backend state shape {state.shape} != ({STATE_DIM},)")
        return TimedRL100Observation(
            state16=state,
            point_cloud=np.asarray(cloud.points, dtype=np.float32),
            state_stamp_s=float(extras["state_stamp_s"]),
            hand_stamp_s=float(extras["hand_stamp_s"]),
            point_cloud_stamp_s=float(cloud.fused_stamp_s),
            received_at_s=float(cloud.received_at_s),
            state_age_s=float(extras["state_age_s"]),
            hand_age_s=float(extras["hand_age_s"]),
            point_cloud_age_s=float(cloud.oldest_age_s),
            state_hand_skew_s=abs(float(extras["state_stamp_s"]) - float(extras["hand_stamp_s"])),
            max_camera_skew_s=float(cloud.max_camera_skew_s),
            raw_joint_dim=int(backend_obs.raw_joint_dim),
        )

    def _validate_observation(self, obs: TimedRL100Observation) -> str | None:
        if obs.state16.shape != (STATE_DIM,) or obs.point_cloud.shape != (1024, 3):
            return "observation shape mismatch"
        if not np.isfinite(obs.state16).all() or not np.isfinite(obs.point_cloud).all():
            return "non-finite observation"
        if obs.state_age_s > self.limits.state_max_age_s:
            return f"state age {obs.state_age_s:.3f}s"
        if obs.hand_age_s > self.limits.hand_max_age_s:
            return f"hand age {obs.hand_age_s:.3f}s"
        if obs.point_cloud_age_s > self.limits.depth_max_age_s:
            return f"depth age {obs.point_cloud_age_s:.3f}s"
        if obs.state_hand_skew_s > self.limits.max_state_hand_skew_s:
            return f"state/hand skew {obs.state_hand_skew_s:.3f}s"
        if obs.max_camera_skew_s > self.limits.max_camera_skew_s:
            return f"camera skew {obs.max_camera_skew_s:.3f}s"
        if abs(obs.state_stamp_s - obs.point_cloud_stamp_s) > self.limits.max_state_cloud_skew_s:
            return "state/point-cloud skew too large"
        return None

    def tick(self) -> TickResult:
        started = time.monotonic()
        record: dict[str, Any] = {"state_before": self.state.value, "published": False}
        if self.state in {DeployState.FAULT, DeployState.STOPPED}:
            return self._result(FaultCode.STOP_SIGNAL, "runner is stopped/faulted", record)
        if self.backend.is_stop() or self.backend.is_shutdown():
            return self._fault_result(FaultCode.STOP_SIGNAL, "robot stop/shutdown", record)
        if self.backend.is_pause():
            self.state = DeployState.HOLD
            self.history.clear()
            return self._result(FaultCode.NONE, "paused", record)
        if self.state == DeployState.HOLD:
            self.state = DeployState.READY
        if self.state == DeployState.READY:
            if self.shadow_mode:
                self.state = DeployState.SHADOW
            else:
                return self._result(FaultCode.CONFIGURATION_ERROR, "live runner is not armed", record)
        if self.state == DeployState.ARMED:
            self.state = DeployState.RUNNING

        try:
            observation = self._timed_observation(self.backend.get_observation(), self.point_cloud_source())
            reason = self._validate_observation(observation)
            if reason:
                self._source_failures += 1
                if self._source_failures >= self.limits.max_consecutive_source_failures:
                    return self._fault_result(FaultCode.STALE_OBSERVATION, reason, record)
                return self._result(FaultCode.STALE_OBSERVATION, reason, record)
            self._source_failures = 0
            self.history.append(observation.point_cloud, observation.state16)
            points, states = self.history.arrays()
            infer_started = time.monotonic()
            chunk = np.asarray(self.policy.predict(points, states), dtype=np.float32)
            inference_s = time.monotonic() - infer_started
            if inference_s > self.limits.inference_timeout_s:
                return self._fault_result(FaultCode.INFERENCE_TIMEOUT, f"inference {inference_s:.3f}s", record)
            if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM or chunk.shape[0] < 1:
                return self._fault_result(FaultCode.ACTION_SHAPE, f"policy chunk shape {chunk.shape}", record)
            if not np.isfinite(chunk).all():
                return self._fault_result(FaultCode.ACTION_NAN, "policy chunk has NaN/Inf", record)
            raw_action = chunk[0]
            measured_limited, measured_clipped = limit_action_to_measured_state(
                raw_action,
                observation.state16,
                max_arm_jump_rad=self.limits.max_arm_state_jump_rad,
                max_gripper_jump=self.limits.max_gripper_state_jump,
            )
            safe = self.safety.check(
                measured_limited,
                stop=self.backend.is_stop(),
                ros_shutdown=self.backend.is_shutdown(),
                observation_age_s=max(observation.state_age_s, observation.hand_age_s, observation.point_cloud_age_s),
                cross_topic_skew_s=max(observation.state_hand_skew_s, observation.max_camera_skew_s),
            )
            if not safe.ok:
                return self._fault_result(safe.fault_code, safe.reason, record)
            if self.safety.clips_exceeded():
                return self._fault_result(FaultCode.ACTION_LIMIT, "consecutive safety clips exceeded", record)
            command = build_published_command(raw_action, safe.action)
            record.update(
                {
                    "state_stamps": [observation.state_stamp_s, observation.hand_stamp_s, observation.point_cloud_stamp_s],
                    "ages_s": [observation.state_age_s, observation.hand_age_s, observation.point_cloud_age_s],
                    "inference_s": inference_s,
                    "raw_action": raw_action.tolist(),
                    "clipped_action": safe.action.tolist(),
                    "predicted_chunk": chunk.tolist(),
                    "history_padded": self.history.padded_on_start,
                    "measured_state_clipped": measured_clipped,
                    "safety_clipped": safe.clipped,
                    "valid_points": observation.point_cloud.shape[0],
                }
            )
            if not self.shadow_mode:
                self.backend.publish(command)
                record["published"] = True
            self.state = DeployState.SHADOW if self.shadow_mode else DeployState.RUNNING
            record["loop_s"] = time.monotonic() - started
            return self._result(FaultCode.NONE, "", record)
        except Exception as exc:  # noqa: BLE001 - hardware boundary must fault closed
            return self._fault_result(FaultCode.SDK_EXCEPTION, str(exc), record)

    def _fault(self, code: FaultCode, reason: str) -> None:
        self.state = DeployState.FAULT
        self.history.clear()

    def _fault_result(self, code: FaultCode, reason: str, record: dict[str, Any]) -> TickResult:
        self._fault(code, reason)
        return self._result(code, reason, record)

    def _result(self, code: FaultCode, reason: str, record: dict[str, Any]) -> TickResult:
        record.update({"state": self.state.value, "fault_code": code.value, "reason": reason})
        if self.audit_sink is not None:
            self.audit_sink(record)
        return TickResult(self.state, bool(record.get("published")), code, reason, record)


def make_safety_config(
    arm_low: np.ndarray,
    arm_high: np.ndarray,
    *,
    max_arm_step_rad: float,
    max_gripper_step: float,
    control_dt_s: float,
    max_consecutive_clips: int,
) -> SafetyConfig:
    """Build the canonical 16-D safety vectors from 14 arm values and two claws."""
    low14 = np.asarray(arm_low, dtype=np.float32).reshape(14)
    high14 = np.asarray(arm_high, dtype=np.float32).reshape(14)
    low = np.concatenate([low14[:7], [0.0], low14[7:], [0.0]]).astype(np.float32)
    high = np.concatenate([high14[:7], [1.0], high14[7:], [1.0]]).astype(np.float32)
    step = np.full(ACTION_DIM, float(max_arm_step_rad), dtype=np.float32)
    step[7] = step[15] = float(max_gripper_step)
    return SafetyConfig(
        joint_position_low=low,
        joint_position_high=high,
        max_delta_rad=step,
        max_consecutive_clips=int(max_consecutive_clips),
        control_dt_s=float(control_dt_s),
    )


# ---------------------------------------------------------------------------
# RL-100 topic-native deployment path.  The legacy 16-D runner above remains
# available for existing HIL/ACT policies; these classes are intentionally
# independent of its SDK/radian action bridge.


@dataclass(frozen=True)
class RL100TopicPublishedCommand:
    raw_action26: np.ndarray
    limited_action26: np.ndarray
    arm14_rad: np.ndarray
    arm14_deg: np.ndarray
    hand12_raw: np.ndarray
    hand12_uint8: np.ndarray
    clip_mask26: np.ndarray


@dataclass(frozen=True)
class RL100TopicSafetyResult:
    ok: bool
    command: RL100TopicPublishedCommand
    clipped: bool
    fault_code: FaultCode
    reason: str = ""


def build_rl100_topic_command(action26: np.ndarray) -> RL100TopicPublishedCommand:
    """Validate model output and create topic-native fields without publishing."""
    from kuavo_rl.rl100_zarr.topic_native import validate_topic_native_prediction

    raw = validate_topic_native_prediction(action26)
    arm_deg = raw[:14].copy()
    hand_raw = np.clip(raw[14:], 0.0, 100.0).astype(np.float32)
    hand_uint8 = np.rint(hand_raw).astype(np.uint8)
    return RL100TopicPublishedCommand(
        raw_action26=raw.copy(),
        limited_action26=raw.copy(),
        arm14_rad=np.deg2rad(arm_deg).astype(np.float32),
        arm14_deg=arm_deg,
        hand12_raw=hand_raw,
        hand12_uint8=hand_uint8,
        clip_mask26=np.zeros(RL100_ACTION_DIM, dtype=bool),
    )


class RL100TopicSafetyGate:
    """Safety checks in physical units, then returns degree/raw topic fields."""

    def __init__(
        self,
        *,
        arm_low_rad: np.ndarray,
        arm_high_rad: np.ndarray,
        max_arm_step_rad: float | np.ndarray,
        max_hand_step: float | np.ndarray,
        max_consecutive_clips: int = 3,
        max_arm_state_jump_rad: float = 0.05,
        max_hand_state_jump: float = 5.0,
    ) -> None:
        self.arm_low_rad = np.asarray(arm_low_rad, dtype=np.float32).reshape(-1)
        self.arm_high_rad = np.asarray(arm_high_rad, dtype=np.float32).reshape(-1)
        if self.arm_low_rad.shape != (14,) or self.arm_high_rad.shape != (14,):
            raise ValueError("topic-native arm limits must be 14-D")
        if np.any(~np.isfinite(self.arm_low_rad)) or np.any(~np.isfinite(self.arm_high_rad)):
            raise ValueError("topic-native arm limits contain NaN/Inf")
        if np.any(self.arm_low_rad >= self.arm_high_rad):
            raise ValueError("topic-native arm low limits must be less than high limits")
        self.max_arm_step_rad = np.broadcast_to(
            np.asarray(max_arm_step_rad, dtype=np.float32), (14,)
        ).copy()
        self.max_hand_step = np.broadcast_to(
            np.asarray(max_hand_step, dtype=np.float32), (12,)
        ).copy()
        if np.any(self.max_arm_step_rad <= 0) or np.any(self.max_hand_step <= 0):
            raise ValueError("topic-native step limits must be positive")
        self.max_consecutive_clips = int(max_consecutive_clips)
        self.max_arm_state_jump_rad = float(max_arm_state_jump_rad)
        self.max_hand_state_jump = float(max_hand_state_jump)
        if self.max_arm_state_jump_rad <= 0 or self.max_hand_state_jump <= 0:
            raise ValueError("measured-state jump limits must be positive")
        self._last_action26: np.ndarray | None = None
        self.consecutive_clips = 0

    def reset(self) -> None:
        self._last_action26 = None
        self.consecutive_clips = 0

    def check(
        self,
        action26: np.ndarray,
        measured_state32: np.ndarray,
        *,
        stop: bool = False,
        ros_shutdown: bool = False,
    ) -> RL100TopicSafetyResult:
        from kuavo_rl.rl100_zarr.topic_native import validate_topic_native_prediction

        try:
            raw = validate_topic_native_prediction(action26)
        except ValueError as exc:
            empty = np.zeros(RL100_ACTION_DIM, dtype=np.float32)
            return RL100TopicSafetyResult(
                False,
                self._empty_command(empty),
                False,
                FaultCode.ACTION_SHAPE,
                str(exc),
            )
        measured = np.asarray(measured_state32, dtype=np.float32).reshape(-1)
        if measured.shape != (RL100_STATE_DIM,) or not np.isfinite(measured).all():
            return RL100TopicSafetyResult(
                False,
                self._empty_command(raw),
                False,
                FaultCode.STALE_OBSERVATION,
                "measured topic-native state is not finite 32-D",
            )
        if stop:
            return RL100TopicSafetyResult(False, self._hold_command(measured), False, FaultCode.STOP_SIGNAL, "stop")
        if ros_shutdown:
            return RL100TopicSafetyResult(False, self._hold_command(measured), False, FaultCode.ROS_SHUTDOWN, "ros shutdown")

        measured_arm = measured[RL100_ARM_SLICE_RAW20]
        measured_hand = measured[20:32]
        if np.any(measured_hand < 0.0) or np.any(measured_hand > 100.0):
            return RL100TopicSafetyResult(
                False,
                self._empty_command(raw),
                False,
                FaultCode.STALE_OBSERVATION,
                "measured dexhand state is outside [0,100]",
            )
        target_arm = np.deg2rad(raw[:14]).astype(np.float32)
        target_hand = raw[14:].copy()
        clipped = False
        clip_mask = np.zeros(RL100_ACTION_DIM, dtype=bool)

        limited_arm = np.clip(target_arm, self.arm_low_rad, self.arm_high_rad)
        arm_mask = ~np.isclose(limited_arm, target_arm)
        if np.any(arm_mask):
            clipped = True
            clip_mask[:14] |= arm_mask

        delta_measured_arm = limited_arm - measured_arm
        measured_arm_limited = np.clip(
            delta_measured_arm,
            -self.max_arm_state_jump_rad,
            self.max_arm_state_jump_rad,
        ) + measured_arm
        if not np.allclose(measured_arm_limited, limited_arm):
            clipped = True
            clip_mask[:14] |= ~np.isclose(measured_arm_limited, limited_arm)
            limited_arm = measured_arm_limited

        limited_hand = np.clip(target_hand, 0.0, 100.0)
        hand_mask = ~np.isclose(limited_hand, target_hand)
        if np.any(hand_mask):
            clipped = True
            clip_mask[14:] |= hand_mask
        delta_measured_hand = limited_hand - measured_hand
        measured_hand_limited = np.clip(
            delta_measured_hand,
            -self.max_hand_state_jump,
            self.max_hand_state_jump,
        ) + measured_hand
        if not np.allclose(measured_hand_limited, limited_hand):
            clipped = True
            clip_mask[14:] |= ~np.isclose(measured_hand_limited, limited_hand)
            limited_hand = measured_hand_limited

        limited = np.concatenate([np.rad2deg(limited_arm), limited_hand]).astype(np.float32)
        if self._last_action26 is not None:
            previous_arm = np.deg2rad(self._last_action26[:14]).astype(np.float32)
            previous_hand = self._last_action26[14:]
            arm_step = np.clip(limited_arm - previous_arm, -self.max_arm_step_rad, self.max_arm_step_rad)
            hand_step = np.clip(limited_hand - previous_hand, -self.max_hand_step, self.max_hand_step)
            stepped = np.concatenate([np.rad2deg(previous_arm + arm_step), previous_hand + hand_step]).astype(np.float32)
            if not np.allclose(stepped, limited):
                clipped = True
                clip_mask |= ~np.isclose(stepped, limited)
                limited = stepped

        if clipped:
            self.consecutive_clips += 1
        else:
            self.consecutive_clips = 0
        command = self._command_from_values(raw, limited, clip_mask)
        if self.max_consecutive_clips > 0 and self.consecutive_clips >= self.max_consecutive_clips:
            return RL100TopicSafetyResult(
                False,
                command,
                True,
                FaultCode.ACTION_LIMIT,
                "consecutive topic-native safety clips exceeded",
            )
        self._last_action26 = limited.copy()
        return RL100TopicSafetyResult(True, command, clipped, FaultCode.NONE, "")

    @staticmethod
    def _command_from_values(raw: np.ndarray, limited: np.ndarray, clip_mask: np.ndarray) -> RL100TopicPublishedCommand:
        arm_deg = limited[:14].astype(np.float32)
        hand = limited[14:].astype(np.float32)
        return RL100TopicPublishedCommand(
            raw_action26=raw.copy(),
            limited_action26=limited.copy(),
            arm14_rad=np.deg2rad(arm_deg).astype(np.float32),
            arm14_deg=arm_deg.copy(),
            hand12_raw=hand.copy(),
            hand12_uint8=np.rint(np.clip(hand, 0.0, 100.0)).astype(np.uint8),
            clip_mask26=np.asarray(clip_mask, dtype=bool).copy(),
        )

    def _empty_command(self, raw: np.ndarray) -> RL100TopicPublishedCommand:
        return self._command_from_values(raw, raw, np.zeros(RL100_ACTION_DIM, dtype=bool))

    def _hold_command(self, measured: np.ndarray) -> RL100TopicPublishedCommand:
        arm_deg = np.rad2deg(measured[RL100_ARM_SLICE_RAW20]).astype(np.float32)
        hand = measured[20:32] if self._last_action26 is None else self._last_action26[14:]
        action = np.concatenate([arm_deg, hand]).astype(np.float32)
        return self._command_from_values(action, action, np.zeros(RL100_ACTION_DIM, dtype=bool))

    def hold_command(self, measured: np.ndarray) -> RL100TopicPublishedCommand:
        """Build one measured-state hold for the fault boundary."""
        return self._hold_command(measured)


class RL100TopicCommandPublisher:
    """Publish the exact Kuavo arm and Qiangnao hand command messages."""

    def __init__(
        self,
        *,
        arm_topic: str = "/kuavo_arm_traj",
        hand_topic: str = "/control_robot_hand_position",
        ros: Any | None = None,
    ) -> None:
        self.arm_topic = arm_topic
        self.hand_topic = hand_topic
        self._ros = ros
        self._arm_pub: Any | None = None
        self._hand_pub: Any | None = None
        self.publish_count = 0

    def start(self) -> None:
        if self._arm_pub is not None:
            return
        if self._ros is None:
            import rospy

            self._ros = rospy
        from kuavo_msgs.msg import robotHandPosition
        from sensor_msgs.msg import JointState

        self._arm_pub = self._ros.Publisher(self.arm_topic, JointState, queue_size=1)
        self._hand_pub = self._ros.Publisher(self.hand_topic, robotHandPosition, queue_size=1)

    def close(self) -> None:
        for pub in (self._arm_pub, self._hand_pub):
            try:
                if pub is not None:
                    pub.unregister()
            except Exception:  # noqa: BLE001
                pass
        self._arm_pub = None
        self._hand_pub = None

    def connection_counts(self) -> dict[str, int]:
        return {
            "arm": int(self._arm_pub.get_num_connections()) if self._arm_pub is not None else 0,
            "hand": int(self._hand_pub.get_num_connections()) if self._hand_pub is not None else 0,
        }

    def publish(self, command: RL100TopicPublishedCommand) -> None:
        if self._arm_pub is None or self._hand_pub is None:
            raise RuntimeError("RL100TopicCommandPublisher is not started")
        from kuavo_msgs.msg import robotHandPosition
        from sensor_msgs.msg import JointState

        stamp = self._ros.Time.now()
        arm_msg = JointState()
        arm_msg.header.stamp = stamp
        arm_msg.name = list(RL100_ARM_JOINT_NAMES)
        arm_msg.position = [float(v) for v in command.arm14_deg]
        hand_msg = robotHandPosition()
        hand_msg.header.stamp = stamp
        hand_msg.left_hand_position = [int(v) for v in command.hand12_uint8[:6]]
        hand_msg.right_hand_position = [int(v) for v in command.hand12_uint8[6:]]
        self._arm_pub.publish(arm_msg)
        self._hand_pub.publish(hand_msg)
        self.publish_count += 1


@dataclass(frozen=True)
class RL100TopicRunnerLimits:
    control_hz: float = 10.0
    execute_steps: int = 1
    action_buffer_size: int = 8
    action_low_watermark: int = 2
    joint_state_max_age_s: float = 0.15
    dexhand_state_max_age_s: float = 0.15
    state_max_skew_s: float = 0.10
    depth_max_age_s: float = 0.15
    max_camera_skew_s: float = 0.10
    max_camera_receive_skew_s: float = 0.20
    max_state_cloud_skew_s: float = 0.10
    inference_timeout_s: float = 0.10
    max_consecutive_source_failures: int = 1
    fault_hold_once_if_state_fresh: bool = True


@dataclass(frozen=True)
class RL100TopicTickResult:
    state: DeployState
    published: bool
    fault_code: FaultCode
    reason: str
    record: dict[str, Any]


class RL100TopicObservationHistory:
    def __init__(self, n_obs_steps: int) -> None:
        if int(n_obs_steps) < 1:
            raise ValueError("n_obs_steps must be >= 1")
        self.n_obs_steps = int(n_obs_steps)
        self._points: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self._states: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self.padded_on_start = False

    def clear(self) -> None:
        self._points.clear()
        self._states.clear()
        self.padded_on_start = False

    def append(self, point_cloud: np.ndarray, state32: np.ndarray) -> None:
        point = np.asarray(point_cloud, dtype=np.float32)
        state = np.asarray(state32, dtype=np.float32).reshape(-1)
        if point.shape != (1024, 3):
            raise ValueError(f"point cloud shape {point.shape} != (1024, 3)")
        if state.shape != (RL100_STATE_DIM,):
            raise ValueError(f"state shape {state.shape} != ({RL100_STATE_DIM},)")
        if not np.isfinite(point).all() or not np.isfinite(state).all():
            raise ValueError("non-finite topic-native observation")
        if not self._points:
            self._points.extend(point.copy() for _ in range(self.n_obs_steps))
            self._states.extend(state.copy() for _ in range(self.n_obs_steps))
            self.padded_on_start = self.n_obs_steps > 1
            return
        self._points.append(point.copy())
        self._states.append(state.copy())

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self._points) != self.n_obs_steps:
            raise RuntimeError("topic-native observation history is not ready")
        return np.stack(self._points), np.stack(self._states)


class RL100TopicRealRunner:
    """Buffered 32/26 RL-100 runner using direct ROS command topics.

    Actions are consumed at the training/control rate while the next action
    chunk is inferred in the background.  The ROS controller remains
    responsible for its native high-rate trajectory tracking.
    """

    def __init__(
        self,
        *,
        policy: PolicyLike,
        state_hub: Any,
        point_cloud_source: Callable[..., PointCloudSample],
        publisher: RL100TopicCommandPublisher,
        safety: RL100TopicSafetyGate,
        limits: RL100TopicRunnerLimits = RL100TopicRunnerLimits(),
        shadow_mode: bool = True,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
        stop_source: Callable[[], tuple[bool, bool, bool]] | None = None,
    ) -> None:
        self.policy = policy
        self.state_hub = state_hub
        self.point_cloud_source = point_cloud_source
        self.publisher = publisher
        self.safety = safety
        self.limits = limits
        self.shadow_mode = bool(shadow_mode)
        self.audit_sink = audit_sink
        self.stop_source = stop_source or (lambda: (False, False, False))
        self.history = RL100TopicObservationHistory(int(policy.info.n_obs_steps))
        self.state = DeployState.INIT
        self._source_failures = 0
        self._fault_hold_sent = False
        self._buffer_empty_ticks = 0
        self._action_scheduler = RL100TopicActionScheduler(
            policy=policy,
            action_hz=limits.control_hz,
            execute_steps=limits.execute_steps,
            buffer_size=limits.action_buffer_size,
            low_watermark=limits.action_low_watermark,
            inference_timeout_s=limits.inference_timeout_s,
        )

    def preflight(self) -> None:
        self.state = DeployState.PREFLIGHT
        info = self.policy.info
        if (
            str(getattr(info, "contract", RL100_TOPIC_NATIVE_CONTRACT))
            != RL100_TOPIC_NATIVE_CONTRACT
            or int(info.state_dim) != RL100_STATE_DIM
            or int(info.action_dim) != RL100_ACTION_DIM
        ):
            self._fault(FaultCode.ACTION_SHAPE, "checkpoint is not RL-100 topic-native 32/26")
            raise RuntimeError("RL-100 topic-native checkpoint contract mismatch")
        if int(info.point_count) != 1024 or int(info.point_dim) != 3:
            self._fault(FaultCode.ACTION_SHAPE, "checkpoint point cloud shape is not 1024x3")
            raise RuntimeError("RL-100 topic-native point cloud mismatch")
        checkpoint_steps = int(getattr(info, "n_action_steps", self.limits.execute_steps))
        if checkpoint_steps < self.limits.execute_steps:
            self._fault(
                FaultCode.CONFIGURATION_ERROR,
                f"execute_steps={self.limits.execute_steps} exceeds checkpoint n_action_steps={checkpoint_steps}",
            )
            raise RuntimeError("RL-100 execute_steps exceeds checkpoint action horizon")
        self.safety.reset()
        self._action_scheduler.reset()
        self._buffer_empty_ticks = 0
        self._fault_hold_sent = False
        self.state = DeployState.READY

    def close(self) -> None:
        """Stop the background inference worker."""
        self._action_scheduler.close()

    def arm_live(self) -> None:
        if self.shadow_mode:
            raise RuntimeError("runner is configured shadow_mode=true")
        if self.state not in {DeployState.READY, DeployState.SHADOW}:
            raise RuntimeError(f"cannot arm live runner from {self.state}")
        self.state = DeployState.ARMED

    def tick(self) -> RL100TopicTickResult:
        started = time.monotonic()
        record: dict[str, Any] = {"state_before": self.state.value, "published": False}
        if self.state in {DeployState.FAULT, DeployState.STOPPED}:
            return self._result(FaultCode.STOP_SIGNAL, "runner is stopped/faulted", record)
        stop, paused, shutdown = self.stop_source()
        if stop or shutdown:
            return self._fault_result(
                FaultCode.STOP_SIGNAL if stop else FaultCode.ROS_SHUTDOWN,
                "stop/shutdown signal",
                record,
            )
        if paused:
            self.state = DeployState.HOLD
            self.history.clear()
            self._action_scheduler.reset()
            self._buffer_empty_ticks = 0
            return self._result(FaultCode.NONE, "paused", record)
        if self.state == DeployState.HOLD:
            self.state = DeployState.READY
        if self.state == DeployState.READY:
            if self.shadow_mode:
                self.state = DeployState.SHADOW
            else:
                return self._result(FaultCode.CONFIGURATION_ERROR, "live runner is not armed", record)
        if self.state == DeployState.ARMED:
            self.state = DeployState.RUNNING

        state_sample = None
        phase = "source"
        try:
            cutoff_monotonic_s = time.monotonic()
            try:
                state_sample = self.state_hub.snapshot(
                    self.limits.joint_state_max_age_s,
                    self.limits.dexhand_state_max_age_s,
                    self.limits.state_max_skew_s,
                    cutoff_monotonic_s=cutoff_monotonic_s,
                )
            except TypeError as exc:
                # Preserve injected/legacy state hubs that do not expose the
                # optional causal cutoff parameter.
                if "cutoff_monotonic_s" not in str(exc):
                    raise
                state_sample = self.state_hub.snapshot(
                    self.limits.joint_state_max_age_s,
                    self.limits.dexhand_state_max_age_s,
                    self.limits.state_max_skew_s,
                )
            try:
                cloud = self.point_cloud_source(cutoff_monotonic_s=cutoff_monotonic_s)
            except TypeError as exc:
                # Preserve injected/legacy no-argument point-cloud sources.
                if "cutoff_monotonic_s" not in str(exc):
                    raise
                cloud = self.point_cloud_source()
            if cloud.points.shape != (1024, 3) or not np.isfinite(cloud.points).all():
                raise ValueError(f"point cloud shape/finite check failed: {cloud.points.shape}")
            if cloud.oldest_age_s > self.limits.depth_max_age_s:
                raise RuntimeError(f"point cloud stale: {cloud.oldest_age_s:.3f}s")
            if cloud.max_camera_skew_s > self.limits.max_camera_skew_s:
                raise RuntimeError(f"camera skew: {cloud.max_camera_skew_s:.3f}s")
            if cloud.max_receive_skew_s > self.limits.max_camera_receive_skew_s:
                raise RuntimeError(
                    f"camera receive skew: {cloud.max_receive_skew_s:.3f}s"
                )
            if abs(state_sample.joint_stamp_s - cloud.fused_stamp_s) > self.limits.max_state_cloud_skew_s:
                raise RuntimeError("state/point-cloud timestamp skew too large")
            self._source_failures = 0
            self.history.append(cloud.points, state_sample.state32)
            points, states = self.history.arrays()
            phase = "inference"
            scheduled = self._action_scheduler.step(points, states)
            scheduler_state = scheduled.state
            record.update(
                {
                    "action_buffer_pending": scheduler_state.pending,
                    "inference_running": scheduler_state.inference_running,
                    "inference_ready": scheduler_state.inference_ready,
                    "inference_s": scheduler_state.inference_s,
                    "stale_action_steps_dropped": scheduler_state.stale_steps_dropped,
                    "inference_error": scheduler_state.last_error,
                    "predicted_chunk": (
                        scheduler_state.predicted_chunk.tolist()
                        if scheduler_state.predicted_chunk is not None
                        else None
                    ),
                }
            )
            if scheduled.step is None:
                self._buffer_empty_ticks += 1
                record["action_buffer_empty_ticks"] = self._buffer_empty_ticks
                if scheduler_state.last_error and not scheduler_state.inference_running:
                    error_code = (
                        FaultCode.INFERENCE_TIMEOUT
                        if "consumed the entire" in scheduler_state.last_error
                        else FaultCode.ACTION_SHAPE
                    )
                    return self._fault_result(
                        error_code,
                        scheduler_state.last_error,
                        record,
                        state_sample=state_sample,
                    )
                if self._buffer_empty_ticks > self._action_scheduler.max_empty_ticks:
                    return self._fault_result(
                        FaultCode.INFERENCE_TIMEOUT,
                        "action buffer exhausted while inference was not ready",
                        record,
                        state_sample=state_sample,
                    )
                if not self.shadow_mode:
                    self.publisher.publish(self.safety.hold_command(state_sample.state32))
                    record["published"] = True
                record["loop_s"] = time.monotonic() - started
                self.state = DeployState.SHADOW if self.shadow_mode else DeployState.RUNNING
                return self._result(FaultCode.NONE, "waiting for action chunk", record)

            self._buffer_empty_ticks = 0
            raw_action = scheduled.step.action
            safe = self.safety.check(raw_action, state_sample.state32)
            record.update(
                {
                    "contract": RL100_TOPIC_NATIVE_CONTRACT,
                    "state_stamps": [state_sample.joint_stamp_s, state_sample.hand_stamp_s, cloud.fused_stamp_s],
                    "ages_s": [state_sample.joint_age_s, state_sample.hand_age_s, cloud.oldest_age_s],
                    "camera_sync": {
                        "reference_camera": cloud.reference_camera,
                        "reference_stamp_s": cloud.reference_stamp_s,
                        "header_skew_s": cloud.max_header_skew_s,
                        "receive_skew_s": cloud.max_receive_skew_s,
                        "camera_stamps": cloud.camera_stamps,
                        "camera_received_ages_s": cloud.camera_received_ages_s or {},
                    },
                    "inference_s": scheduled.step.inference_s,
                    "raw_action": raw_action.tolist(),
                    "limited_action": safe.command.limited_action26.tolist(),
                    "arm14_deg": safe.command.arm14_deg.tolist(),
                    "hand12_raw": safe.command.hand12_raw.tolist(),
                    "clip_mask": safe.command.clip_mask26.tolist(),
                    "history_padded": self.history.padded_on_start,
                    "safety_clipped": safe.clipped,
                    "valid_points": cloud.valid_points,
                }
            )
            if not safe.ok:
                return self._fault_result(safe.fault_code, safe.reason, record, state_sample=state_sample)
            if not self.shadow_mode:
                phase = "publish"
                self.publisher.publish(safe.command)
                record["published"] = True
            record["loop_s"] = time.monotonic() - started
            self.state = DeployState.SHADOW if self.shadow_mode else DeployState.RUNNING
            return self._result(FaultCode.NONE, "", record)
        except Exception as exc:  # noqa: BLE001
            self._source_failures += 1
            if self._source_failures < self.limits.max_consecutive_source_failures:
                return self._result(FaultCode.STALE_OBSERVATION, str(exc), record)
            return self._fault_result(
                FaultCode.STALE_OBSERVATION if phase == "source" else FaultCode.SDK_EXCEPTION,
                str(exc),
                record,
                state_sample=state_sample,
            )

    def _fault(
        self,
        code: FaultCode,
        reason: str,
        *,
        state_sample: Any | None = None,
    ) -> None:
        self.state = DeployState.FAULT
        self.history.clear()
        self._action_scheduler.reset()
        self._buffer_empty_ticks = 0
        if (
            state_sample is not None
            and not self.shadow_mode
            and self.limits.fault_hold_once_if_state_fresh
            and not self._fault_hold_sent
        ):
            try:
                hold = self.safety.hold_command(state_sample.state32)  # single final measured hold
                self.publisher.publish(hold)
                self._fault_hold_sent = True
            except Exception:  # noqa: BLE001
                self._fault_hold_sent = True

    def _fault_result(
        self,
        code: FaultCode,
        reason: str,
        record: dict[str, Any],
        *,
        state_sample: Any | None = None,
    ) -> RL100TopicTickResult:
        self._fault(code, reason, state_sample=state_sample)
        return self._result(code, reason, record)

    def _result(self, code: FaultCode, reason: str, record: dict[str, Any]) -> RL100TopicTickResult:
        record.update({"state": self.state.value, "fault_code": code.value, "reason": reason})
        if self.audit_sink is not None:
            self.audit_sink(record)
        return RL100TopicTickResult(self.state, bool(record.get("published")), code, reason, record)
