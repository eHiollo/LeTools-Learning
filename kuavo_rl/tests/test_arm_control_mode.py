from __future__ import annotations

import pytest

from kuavo_rl.arm_control_mode import ArmControlModeSession


def test_successful_entry_restores_mode_zero_once():
    calls = []
    session = ArmControlModeSession(lambda mode: calls.append(mode) or True)

    session.enter()
    session.restore()
    session.restore()

    assert calls == [2, 0]


def test_rejected_entry_never_attempts_restore():
    calls = []
    session = ArmControlModeSession(lambda mode: calls.append(mode) and False)

    with pytest.raises(RuntimeError, match="external"):
        session.enter()
    session.restore()

    assert calls == [2]


def test_failed_restore_is_not_retried():
    calls = []
    session = ArmControlModeSession(lambda mode: calls.append(mode) or mode == 2)

    session.enter()
    with pytest.raises(RuntimeError, match="restore"):
        session.restore()
    session.restore()

    assert calls == [2, 0]
