from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from kuavo_rl.contracts import (
    RL100_DEXHAND_JOINT_NAMES,
    RL100_HAND_DEFAULT,
    compose_rl100_topic_action,
    compose_rl100_topic_state,
)
from kuavo_rl.contracts import FaultCode
from kuavo_rl.rl100_real_runner import (
    DeployState,
    PointCloudSample,
    RL100TopicRealRunner,
    RL100TopicRunnerLimits,
    RL100TopicSafetyGate,
    build_rl100_topic_command,
)
from kuavo_rl.rl100_zarr.ros_state import TopicStateHub
from kuavo_rl.rl100_zarr.topic_native import TopicCommandCache


def _header(stamp: float):
    return SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: stamp))


def _arm(values, stamp: float = 1.0):
    return SimpleNamespace(position=list(values), header=_header(stamp))


def _hand(values, stamp: float = 1.0):
    return SimpleNamespace(
        left_hand_position=list(values[:6]),
        right_hand_position=list(values[6:]),
        header=_header(stamp),
    )


def test_topic_compose_preserves_source_order_and_units():
    raw = np.arange(20, dtype=np.float32)
    hand = np.arange(12, dtype=np.float32) + 20
    arm = np.arange(14, dtype=np.float32) + 100
    command_hand = np.arange(12, dtype=np.float32) + 50
    np.testing.assert_array_equal(compose_rl100_topic_state(raw, hand), np.arange(32, dtype=np.float32))
    np.testing.assert_array_equal(
        compose_rl100_topic_action(arm, command_hand), np.concatenate([arm, command_hand])
    )


def test_command_cache_arm_first_then_hand_hold_and_causality():
    cache = TopicCommandCache()
    cache.begin_episode(np.zeros(14, dtype=np.float32), generation=7, record_start_received_at=10.0)

    # A pre-record callback cannot start or contaminate the episode.
    assert not cache.update_arm(_arm([99.0] * 14, 0.5), received_at_s=9.9, generation=7)
    first_hand = [10, 20, 30, 40, 50, 60, 70, 80, 90, 0, 1, 2]
    assert cache.update_hand(_hand(first_hand, 1.0), received_at_s=10.1, generation=7)
    snap = cache.snapshot_and_clear_changed(10.15)
    assert not snap.arm_seen and snap.hand_seen
    np.testing.assert_allclose(snap.arm14_deg, 0.0)
    np.testing.assert_array_equal(snap.hand12_raw, first_hand)
    assert snap.hand_changed

    # A command after the cutoff belongs to the next sample, never this one.
    assert cache.update_arm(_arm([90.0] * 14, 2.0), received_at_s=10.3, generation=7)
    snap = cache.snapshot_and_clear_changed(10.2)
    assert not snap.arm_seen
    np.testing.assert_allclose(snap.arm14_deg, 0.0)
    snap = cache.snapshot_and_clear_changed(10.4)
    assert snap.arm_seen and snap.hand_seen and snap.arm_changed
    np.testing.assert_allclose(snap.arm14_deg, 90.0)
    # No new callback: both values are held and changed is cleared.
    snap = cache.snapshot_and_clear_changed(10.5)
    assert not snap.arm_changed and not snap.hand_changed
    np.testing.assert_array_equal(snap.hand12_raw, first_hand)


def test_command_cache_rejects_invalid_hand_and_backwards_header():
    cache = TopicCommandCache()
    cache.begin_episode(np.zeros(14), record_start_received_at=1.0)
    valid = _arm([1.0] * 14, 10.0)
    assert cache.update_arm(valid, received_at_s=1.1)
    assert not cache.update_arm(_arm([2.0] * 14, 9.0), received_at_s=1.2)
    assert not cache.update_hand(_hand([101.0] * 12), received_at_s=1.3)
    assert cache.invalid_counts["arm_header_backwards"] == 1
    snap = cache.snapshot_and_clear_changed(1.4)
    np.testing.assert_allclose(snap.arm14_deg, 1.0)
    np.testing.assert_array_equal(snap.hand12_raw, RL100_HAND_DEFAULT)


def test_state_hub_preserves_20_plus_12_and_checks_hand_names():
    hub = TopicStateHub()
    now = __import__("time").time()
    raw = np.arange(20, dtype=np.float32)
    hand = np.arange(12, dtype=np.float32)
    assert hub.update_joint(raw, stamp_s=now, received_at_s=now)
    assert hub.update_hand(
        hand,
        names=np.asarray(RL100_DEXHAND_JOINT_NAMES),
        stamp_s=now,
        received_at_s=now,
    )
    sample = hub.snapshot(1.0, 1.0, 0.1)
    np.testing.assert_array_equal(sample.state32, np.concatenate([raw, hand]))
    report = hub.preflight(timeout_s=0.0)
    np.testing.assert_array_equal(report["joint_value_min"], raw)
    np.testing.assert_array_equal(report["dexhand_value_max"], hand)
    assert not hub.update_hand(hand, names=list(RL100_DEXHAND_JOINT_NAMES[:-1]), received_at_s=now)
    assert hub.invalid_counts["hand_names"] == 1


def test_safety_gate_converts_arm_and_clips_hand_before_publish():
    gate = RL100TopicSafetyGate(
        arm_low_rad=np.full(14, -2.0),
        arm_high_rad=np.full(14, 2.0),
        max_arm_step_rad=0.5,
        max_hand_step=100.0,
        max_consecutive_clips=3,
        max_arm_state_jump_rad=0.5,
        max_hand_state_jump=100.0,
    )
    measured = compose_rl100_topic_state(np.zeros(20), RL100_HAND_DEFAULT)
    raw = np.concatenate([np.full(14, 90.0), np.full(12, 120.0)]).astype(np.float32)
    result = gate.check(raw, measured)
    assert result.ok and result.clipped and result.fault_code == FaultCode.NONE
    np.testing.assert_allclose(result.command.arm14_deg, np.rad2deg(np.full(14, 0.5)))
    np.testing.assert_array_equal(result.command.hand12_uint8, np.full(12, 100, dtype=np.uint8))
    np.testing.assert_array_equal(gate.hold_command(measured).hand12_uint8, np.full(12, 100, dtype=np.uint8))
    direct = build_rl100_topic_command(raw)
    np.testing.assert_array_equal(direct.hand12_uint8, np.full(12, 100, dtype=np.uint8))


class _FakeStateHub:
    def __init__(self):
        now = __import__("time").time()
        raw = np.zeros(20, dtype=np.float32)
        self.sample = SimpleNamespace(
            state32=compose_rl100_topic_state(raw, RL100_HAND_DEFAULT),
            raw_joint_q20=raw,
            dexhand_position12=RL100_HAND_DEFAULT.copy(),
            joint_stamp_s=now,
            hand_stamp_s=now,
            joint_age_s=0.01,
            hand_age_s=0.01,
            joint_hand_skew_s=0.0,
            joint_received_at_s=now,
            hand_received_at_s=now,
        )

    def snapshot(self, *_args):
        return self.sample


class _FakePolicy:
    info = SimpleNamespace(
        contract="rl100_topic_native_v1",
        state_dim=32,
        action_dim=26,
        point_count=1024,
        point_dim=3,
        n_obs_steps=2,
    )

    def predict(self, points, states):
        assert points.shape == (2, 1024, 3)
        assert states.shape == (2, 32)
        return np.zeros((1, 26), dtype=np.float32)


class _FakePublisher:
    def __init__(self):
        self.calls = []

    def publish(self, command):
        self.calls.append(command)


def _topic_runner(shadow: bool):
    state_hub = _FakeStateHub()
    publisher = _FakePublisher()
    safety = RL100TopicSafetyGate(
        arm_low_rad=np.full(14, -2.0),
        arm_high_rad=np.full(14, 2.0),
        max_arm_step_rad=1.0,
        max_hand_step=100.0,
        max_consecutive_clips=3,
        max_arm_state_jump_rad=1.0,
        max_hand_state_jump=100.0,
    )
    now = __import__("time").time()
    cloud = PointCloudSample(
        points=np.zeros((1024, 3), dtype=np.float32),
        fused_stamp_s=now,
        received_at_s=now,
        oldest_age_s=0.01,
        max_camera_skew_s=0.0,
        camera_stamps={"cam": now},
        valid_points=1024,
    )
    runner = RL100TopicRealRunner(
        policy=_FakePolicy(),
        state_hub=state_hub,
        point_cloud_source=lambda: cloud,
        publisher=publisher,
        safety=safety,
        limits=RL100TopicRunnerLimits(
            joint_state_max_age_s=1.0,
            dexhand_state_max_age_s=1.0,
            depth_max_age_s=1.0,
            inference_timeout_s=1.0,
            max_consecutive_source_failures=1,
        ),
        shadow_mode=shadow,
    )
    runner.preflight()
    if not shadow:
        runner.arm_live()
    return runner, publisher


def test_topic_runner_shadow_never_publishes_and_live_publishes_once():
    shadow, shadow_pub = _topic_runner(True)
    shadow_result = shadow.tick()
    assert shadow_result.fault_code == FaultCode.NONE
    assert shadow_result.state == DeployState.SHADOW
    assert not shadow_result.published and not shadow_pub.calls

    live, live_pub = _topic_runner(False)
    live_result = live.tick()
    assert live_result.fault_code == FaultCode.NONE
    assert live_result.published and len(live_pub.calls) == 1
    assert live_pub.calls[0].hand12_uint8.shape == (12,)
