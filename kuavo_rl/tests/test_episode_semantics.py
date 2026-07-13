import numpy as np

from kuavo_rl.adapter import make_kuavo_hilserl_env
from kuavo_rl.backend import MockBackend
from kuavo_rl.config import EnvConfig, EpisodeConfig, default_safety_config
from kuavo_rl.contracts import FaultCode
from kuavo_rl.teleop import TeleopAdapter


def test_timeout_truncates():
    cfg = EnvConfig(
        safety=default_safety_config(),
        episode=EpisodeConfig(max_steps=3, max_duration_s=100),
    )
    env = make_kuavo_hilserl_env(cfg, use_stub_robometer=True)
    env.reset(seed=1)
    truncated = False
    for _ in range(5):
        _obs, _r, term, trunc, info = env.step(np.zeros(16, dtype=np.float32))
        if trunc:
            truncated = True
            assert info["fault_code"] == FaultCode.EPISODE_TIMEOUT.value
            break
        assert not term
    assert truncated
    env.close()


def test_stop_terminates_with_safety_penalty():
    backend = MockBackend()
    teleop = TeleopAdapter()
    cfg = EnvConfig(safety=default_safety_config())
    from kuavo_rl.env import KuavoHILSerlEnv
    from kuavo_rl.reward import RobometerRewardWorker, stub_scorer

    env = KuavoHILSerlEnv(
        config=cfg,
        backend=backend,
        teleop=teleop,
        reward_worker=RobometerRewardWorker(cfg.reward, scorer=stub_scorer),
    )
    env.reset(seed=0)
    backend.set_signals(stop=True)
    _obs, reward, terminated, truncated, info = env.step(np.zeros(16, dtype=np.float32))
    assert terminated is True
    assert truncated is False
    assert reward == -1.0
    assert info["fault_code"] == FaultCode.STOP_SIGNAL.value
    env.close()


def test_manual_success():
    teleop = TeleopAdapter()
    cfg = EnvConfig(safety=default_safety_config())
    from kuavo_rl.env import KuavoHILSerlEnv
    from kuavo_rl.reward import RobometerRewardWorker, stub_scorer

    env = KuavoHILSerlEnv(
        config=cfg,
        backend=MockBackend(),
        teleop=teleop,
        reward_worker=RobometerRewardWorker(cfg.reward, scorer=stub_scorer),
    )
    env.reset(seed=0)
    teleop.push_success()
    _obs, reward, terminated, truncated, info = env.step(np.zeros(16, dtype=np.float32))
    assert terminated and not truncated
    assert reward == 1.0
    assert info["success"] is True
    env.close()


def test_nan_action_fault():
    env = make_kuavo_hilserl_env(use_stub_robometer=True)
    env.reset(seed=0)
    bad = np.zeros(16, dtype=np.float32)
    bad[0] = np.nan
    _obs, reward, terminated, _trunc, info = env.step(bad)
    assert terminated
    assert reward == -1.0
    assert info["fault_code"] == FaultCode.ACTION_NAN.value
    env.close()
