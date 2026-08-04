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
from kuavo_rl.contracts import ACTION_DIM, FaultCode, STATE_DIM, validate_action_shape
from kuavo_rl.ros_adapter import build_published_command
from kuavo_rl.safety import SafetyGate


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
class PointCloudSample:
    points: np.ndarray
    fused_stamp_s: float
    received_at_s: float
    oldest_age_s: float
    max_camera_skew_s: float
    camera_stamps: dict[str, float]
    valid_points: int


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
