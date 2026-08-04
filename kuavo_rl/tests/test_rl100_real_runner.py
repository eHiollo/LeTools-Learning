from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest

from kuavo_rl.backend import BackendObservation, RobotBackend
from kuavo_rl.config import default_safety_config
from kuavo_rl.contracts import FaultCode
from kuavo_rl.rl100_real_runner import (
    ObservationHistory,
    PointCloudSample,
    RL100RealRunner,
    RunnerLimits,
    limit_action_to_measured_state,
)
from kuavo_rl.safety import SafetyGate


class FakePolicy:
    def __init__(self, action: np.ndarray, *, delay_s: float = 0.0):
        self.info = SimpleNamespace(n_obs_steps=2, state_dim=16, action_dim=16, point_count=1024, point_dim=3)
        self.action = action.astype(np.float32)
        self.delay_s = delay_s
        self.calls = 0

    def predict(self, points, states):
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        return np.stack([self.action, self.action + 0.3], axis=0)


class FakeBackend(RobotBackend):
    def __init__(self, *, age_s: float = 0.01, publish_error: bool = False):
        self.state = np.zeros(16, dtype=np.float32)
        self.age_s = age_s
        self.publish_error = publish_error
        self.published = []

    def reset(self, *, seed=None):
        return self.get_observation()

    def get_observation(self):
        stamp = time.time() - self.age_s
        return BackendObservation(
            state=self.state.copy(), images={}, timestamp_s=stamp, raw_joint_dim=20,
            extras={"state_stamp_s": stamp, "hand_stamp_s": stamp, "state_age_s": self.age_s, "hand_age_s": self.age_s},
        )

    def publish(self, command):
        if self.publish_error:
            raise RuntimeError("sdk failed")
        self.published.append(command)
        self.state = command.clipped_action.copy()

    def is_stop(self): return False
    def is_pause(self): return False
    def is_shutdown(self): return False


def _cloud():
    now = time.time()
    return PointCloudSample(np.zeros((1024, 3), np.float32), now, now, 0.01, 0.0, {"cam": now}, 1024)


def _runner(*, action=None, shadow=True, age_s=0.01, timeout=0.1, publish_error=False, delay_s=0.0):
    backend = FakeBackend(age_s=age_s, publish_error=publish_error)
    policy = FakePolicy(np.zeros(16, np.float32) if action is None else action, delay_s=delay_s)
    runner = RL100RealRunner(
        policy=policy, backend=backend, point_cloud_source=_cloud,
        safety=SafetyGate(default_safety_config()),
        limits=RunnerLimits(inference_timeout_s=timeout), shadow_mode=shadow,
    )
    runner.preflight()
    if not shadow:
        runner.arm_live()
    return runner, backend, policy


def test_history_repeats_first_then_slides():
    h = ObservationHistory(2)
    one = np.ones((1024, 3), np.float32)
    state = np.ones(16, np.float32)
    h.append(one, state)
    points, states = h.arrays()
    assert h.padded_on_start and np.all(points[0] == points[1])
    h.append(one * 2, state * 2)
    points, states = h.arrays()
    assert np.all(points[0] == 1) and np.all(points[1] == 2)
    assert np.all(states[0] == 1) and np.all(states[1] == 2)


def test_shadow_predicts_but_never_publishes_and_uses_first_chunk_step():
    action = np.zeros(16, np.float32); action[0] = 0.01
    runner, backend, policy = _runner(action=action, shadow=True)
    result = runner.tick()
    assert result.fault_code == FaultCode.NONE
    assert not result.published and not backend.published and policy.calls == 1
    assert result.record["raw_action"][0] == pytest.approx(0.01)


def test_stale_observation_faults_closed():
    runner, backend, _ = _runner(age_s=1.0)
    result = runner.tick()
    assert result.fault_code == FaultCode.STALE_OBSERVATION
    assert not backend.published


def test_measured_state_gate_uses_distinct_arm_and_gripper_limits():
    target = np.ones(16, np.float32)
    clipped, was_clipped = limit_action_to_measured_state(target, np.zeros(16), max_arm_jump_rad=0.02, max_gripper_jump=0.05)
    assert was_clipped
    assert clipped[0] == np.float32(0.02)
    assert clipped[7] == np.float32(0.05)


def test_live_sdk_exception_faults_and_does_not_retry_publish():
    runner, backend, _ = _runner(shadow=False, publish_error=True)
    result = runner.tick()
    assert result.state.value == "FAULT"
    assert result.fault_code == FaultCode.SDK_EXCEPTION
    assert not backend.published
    again = runner.tick()
    assert again.state.value == "FAULT" and not backend.published


def test_inference_timeout_faults_without_publish():
    runner, backend, _ = _runner(shadow=False, timeout=0.0001, delay_s=0.005)
    result = runner.tick()
    assert result.fault_code == FaultCode.INFERENCE_TIMEOUT
    assert result.state.value == "FAULT" and not backend.published


def test_live_runner_must_be_explicitly_armed():
    runner, backend, _ = _runner(shadow=False)
    runner.state = runner.state.READY
    result = runner.tick()
    assert result.fault_code == FaultCode.CONFIGURATION_ERROR
    assert not backend.published
