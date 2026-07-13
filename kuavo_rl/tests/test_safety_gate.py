import numpy as np

from kuavo_rl.config import default_safety_config
from kuavo_rl.contracts import FaultCode
from kuavo_rl.safety import SafetyGate


def test_shape_and_nan():
    gate = SafetyGate(default_safety_config())
    gate.reset(np.zeros(16, dtype=np.float32))
    r = gate.check(np.zeros(14, dtype=np.float32))
    assert not r.ok and r.fault_code == FaultCode.ACTION_SHAPE
    bad = np.zeros(16, dtype=np.float32)
    bad[3] = np.inf
    r = gate.check(bad)
    assert not r.ok and r.fault_code == FaultCode.ACTION_NAN


def test_stop_and_stale():
    cfg = default_safety_config()
    gate = SafetyGate(cfg)
    gate.reset(np.zeros(16))
    r = gate.check(np.zeros(16), stop=True)
    assert r.fault_code == FaultCode.STOP_SIGNAL
    r = gate.check(np.zeros(16), observation_age_s=cfg.observation_max_age_s + 0.1)
    assert r.fault_code == FaultCode.STALE_OBSERVATION


def test_position_clip_and_delta():
    cfg = default_safety_config(control_dt_s=0.1)
    gate = SafetyGate(cfg)
    start = np.zeros(16, dtype=np.float32)
    gate.reset(start)
    huge = np.full(16, 10.0, dtype=np.float32)
    huge[7] = huge[15] = 0.5
    r = gate.check(huge)
    assert r.ok
    assert r.clipped
    # First step from 0: arms limited to max_delta, not full jump to pi
    assert np.all(np.abs(r.action[0:7] - start[0:7]) <= cfg.max_delta_rad[0:7] + 1e-5)
    # next step toward same huge target remains delta-limited
    r2 = gate.check(huge)
    assert r2.ok
    assert np.all(np.abs(r2.action[0:7] - r.action[0:7]) <= cfg.max_delta_rad[0:7] + 1e-5)
