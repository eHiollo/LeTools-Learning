import numpy as np

from kuavo_rl.adapter import make_kuavo_hilserl_env
from kuavo_rl.ros_adapter import observation_contract_check


def test_reset_observation_contract():
    env = make_kuavo_hilserl_env(use_stub_robometer=True)
    obs, info = env.reset(seed=0)
    errors = observation_contract_check(obs)
    assert errors == []
    assert obs["observation.state"].dtype == np.float32
    assert obs["observation.state"].shape == (16,)
    for key in (
        "observation.images.head_cam_h",
        "observation.images.wrist_cam_l",
        "observation.images.wrist_cam_r",
    ):
        assert key in obs
        assert obs[key].ndim == 3
    env.close()
