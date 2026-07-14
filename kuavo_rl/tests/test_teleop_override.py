import numpy as np

from kuavo_rl.act_runner import ActExecuteFirstRunner, ConstantChunkPolicy
from kuavo_rl.adapter import make_kuavo_hilserl_env
from kuavo_rl.backend import MockBackend
from kuavo_rl.teleop import TeleopAdapter, TeleopEvent


def test_intervention_does_not_duplicate_policy_publish():
    backend = MockBackend()
    teleop = TeleopAdapter()
    env = make_kuavo_hilserl_env(backend=backend, teleop=teleop, use_stub_robometer=True)
    env.reset(seed=0)
    teleop.push_intervention(np.zeros(16, dtype=np.float32), deadman=True)
    env.step(np.ones(16, dtype=np.float32) * 0.01)
    assert backend.publish_count == 0
    env.close()


def test_intervention_audit_keeps_policy_and_vr_actions():
    backend = MockBackend()
    vr_action = np.zeros(16, dtype=np.float32)
    teleop = TeleopAdapter(
        poll_fn=lambda: TeleopEvent(
            action=vr_action, is_intervention=True, deadman=True, source="quest3_ik"
        )
    )
    env = make_kuavo_hilserl_env(backend=backend, teleop=teleop, use_stub_robometer=True)
    policy_action = np.full(16, 0.1, dtype=np.float32)

    result = ActExecuteFirstRunner(ConstantChunkPolicy(policy_action[None, :])).run_episode(
        env, max_steps=1
    )
    step = result["steps"][0]
    info = step["info"]

    assert np.allclose(step["policy_action"], policy_action)
    assert np.allclose(step["executed_action"], vr_action)
    assert info["is_intervention"] is True
    assert info["teleop_source"] == "quest3_ik"
    assert np.allclose(info["teleop_raw_action"], vr_action)
    assert info["teleop_events"]["deadman"] is True
    env.close()


def test_intervention_replay_action_only_overwrites_masked_side():
    backend = MockBackend()
    vr_action = np.ones(16, dtype=np.float32)
    mask = np.zeros(16, dtype=bool)
    mask[:7] = True
    teleop = TeleopAdapter(
        poll_fn=lambda: TeleopEvent(
            action=vr_action,
            is_intervention=True,
            intervention_mask=mask,
            deadman=True,
            source="quest3_ik",
        )
    )
    env = make_kuavo_hilserl_env(backend=backend, teleop=teleop, use_stub_robometer=True)
    result = ActExecuteFirstRunner(ConstantChunkPolicy(np.full((1, 16), 0.2, dtype=np.float32))).run_episode(
        env, max_steps=1
    )
    step = result["steps"][0]
    info = step["info"]
    executed = np.asarray(step["executed_action"], dtype=np.float32)

    assert info["intervention_mask"] == mask.tolist()
    assert info["intervention_segment_id"] == 1
    assert info["intervention_segment_step"] == 1
    assert np.allclose(np.asarray(info["teleop_raw_action"]), vr_action)
    assert not np.allclose(executed[8:15], vr_action[8:15])
    assert np.any(executed[:7] != 0.0)
    env.close()
