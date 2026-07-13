import numpy as np
import pytest

from kuavo_rl.act_runner import ActExecuteFirstRunner, ConstantChunkPolicy
from kuavo_rl.adapter import make_kuavo_hilserl_env
from kuavo_rl.config import ActRunnerConfig


def test_execute_first_only_and_no_backlog():
    chunk = np.arange(10 * 16, dtype=np.float32).reshape(10, 16) * 0.001
    runner = ActExecuteFirstRunner(ConstantChunkPolicy(chunk), ActRunnerConfig(chunk_size=10))
    result = runner.select_action({})
    np.testing.assert_allclose(result.action, chunk[0])
    assert result.executed_index == 0
    assert result.discarded_tail.shape == (9, 16)
    assert runner.pending_queue == []


def test_reject_wrong_action_dim_in_chunk():
    chunk = np.zeros((10, 14), dtype=np.float32)
    runner = ActExecuteFirstRunner(ConstantChunkPolicy(chunk))
    with pytest.raises(ValueError, match="action dim"):
        runner.select_action({})


def test_execute_steps_must_be_one():
    with pytest.raises(ValueError):
        ActRunnerConfig(chunk_size=10, execute_steps=3)


def test_run_episode_repredicts_each_step():
    # chunk rows differ; if backlog were used, later steps would replay old rows
    base = np.zeros((10, 16), dtype=np.float32)

    class CountingPolicy:
        def __init__(self):
            self.n = 0

        def predict_action_chunk(self, obs):
            self.n += 1
            out = base.copy()
            out[0, 0] = float(self.n) * 0.01
            return out

    policy = CountingPolicy()
    runner = ActExecuteFirstRunner(policy, ActRunnerConfig(chunk_size=10))
    env = make_kuavo_hilserl_env(use_stub_robometer=True)
    out = runner.run_episode(env, max_steps=5)
    assert out["n"] == 5
    assert policy.n == 5
    assert all(s["discarded"] == 9 for s in out["steps"])
    env.close()
