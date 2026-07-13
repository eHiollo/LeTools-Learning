"""Stage-B HIL-SERL environment factory bridge (main-repo side)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kuavo_rl.backend import MockBackend, ROSBackend, RobotBackend
from kuavo_rl.config import ActRunnerConfig, EnvConfig, build_env_config_from_dict, load_yaml
from kuavo_rl.env import KuavoHILSerlEnv
from kuavo_rl.reward import RobometerRewardWorker, stub_scorer
from kuavo_rl.teleop import TeleopAdapter


def make_kuavo_hilserl_env(
    config: EnvConfig | dict | str | Path | None = None,
    *,
    backend: RobotBackend | None = None,
    use_stub_robometer: bool = True,
    kuavo_gym_env: Any | None = None,
) -> KuavoHILSerlEnv:
    """
    Create KuavoHILSerlEnv for sim/shadow/real.

    - Default backend is MockBackend (no ROS).
    - Pass an existing Kuavo Gym env as `kuavo_gym_env` to wrap ROSBackend.
    """
    if isinstance(config, (str, Path)):
        cfg = build_env_config_from_dict(load_yaml(config))
    elif isinstance(config, dict):
        cfg = build_env_config_from_dict(config)
    elif config is None:
        cfg = EnvConfig()
    else:
        cfg = config

    if backend is None:
        if kuavo_gym_env is not None:
            backend = ROSBackend(kuavo_gym_env, publish_unit=cfg.arm_publish_unit)
        else:
            backend = MockBackend()

    worker = RobometerRewardWorker(
        cfg.reward,
        scorer=stub_scorer if use_stub_robometer else None,
    )
    return KuavoHILSerlEnv(
        config=cfg,
        backend=backend,
        teleop=TeleopAdapter(),
        reward_worker=worker,
    )


def make_act_runner_config(raw: dict | None = None) -> ActRunnerConfig:
    raw = raw or {}
    return ActRunnerConfig(
        chunk_size=int(raw.get("chunk_size", 10)),
        execute_steps=int(raw.get("execute_steps", 1)),
        fps=int(raw.get("fps", 10)),
    )
