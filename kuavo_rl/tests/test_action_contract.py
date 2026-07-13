import numpy as np
import pytest

from kuavo_rl.contracts import ACTION_DIM, compose_arm14, split_action, validate_action_shape
from kuavo_rl.ros_adapter import (
    arm_rad_to_traj_deg,
    arm_slice_for_raw_dim,
    build_published_command,
    claws_norm_to_command,
    compose_state16,
    slice_arm_state,
)


def test_action_dim_and_split():
    a = np.arange(ACTION_DIM, dtype=np.float32) * 0.01
    left, lc, right, rc = split_action(a)
    assert left.shape == (7,)
    assert right.shape == (7,)
    assert abs(lc - a[7]) < 1e-6
    assert abs(rc - a[15]) < 1e-6
    arm = compose_arm14(a)
    assert arm.shape == (14,)
    np.testing.assert_allclose(arm[:7], left)
    np.testing.assert_allclose(arm[7:], right)


def test_reject_silent_reshape():
    with pytest.raises(ValueError):
        validate_action_shape(np.zeros(14))


def test_raw_state_slice_rules():
    assert arm_slice_for_raw_dim(28) == slice(12, 26)
    assert arm_slice_for_raw_dim(20) == slice(4, 18)
    assert arm_slice_for_raw_dim(14) == slice(0, 14)
    raw28 = np.arange(28, dtype=np.float32)
    arm = slice_arm_state(raw28)
    np.testing.assert_array_equal(arm, raw28[12:26])


def test_rad_deg_and_claw_scaling():
    arm = np.array([0.0, np.pi / 2] + [0.0] * 12, dtype=np.float32)
    deg = arm_rad_to_traj_deg(arm)
    assert abs(deg[1] - 90.0) < 1e-4
    cmd = claws_norm_to_command(np.array([0.5, 1.0]), scale=100.0)
    np.testing.assert_allclose(cmd, [50.0, 100.0])


def test_build_published_command_audit():
    raw = np.zeros(16, dtype=np.float32)
    raw[0] = 0.2
    raw[7] = 0.3
    cmd = build_published_command(raw, raw)
    assert cmd.arm14_rad.shape == (14,)
    assert cmd.arm14_deg.shape == (14,)
    assert cmd.claws_command[0] == pytest.approx(30.0)


def test_compose_state16_order():
    arm14 = np.arange(14, dtype=np.float32)
    claws = np.array([0.1, 0.2], dtype=np.float32)
    state = compose_state16(arm14, claws)
    assert state.shape == (16,)
    np.testing.assert_array_equal(state[:7], arm14[:7])
    assert state[7] == pytest.approx(0.1)
    np.testing.assert_array_equal(state[8:15], arm14[7:])
    assert state[15] == pytest.approx(0.2)
