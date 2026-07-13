"""Unit tests for runtime patches (no third_party edits)."""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")


def test_apply_patches_idempotent():
    from kuavo_rl.lerobot_patches import apply_hilserl_patches

    apply_hilserl_patches()
    apply_hilserl_patches()  # second call no-op


def test_null_dataset_validate_patch(tmp_path):
    from kuavo_rl.lerobot_patches import apply_hilserl_patches

    apply_hilserl_patches()
    from lerobot.policies.gaussian_actor.configuration_gaussian_actor import GaussianActorConfig
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig

    cfg = TrainRLServerPipelineConfig()
    cfg.dataset = None
    cfg.policy = GaussianActorConfig(push_to_hub=False)
    cfg.output_dir = tmp_path / "rl_out"
    cfg.resume = False
    cfg.wandb.enable = False
    cfg.validate()
    assert cfg.dataset is None


def test_headless_make_robot_env_routes_to_base(monkeypatch):
    from kuavo_rl.lerobot_patches import apply_hilserl_patches

    apply_hilserl_patches()
    import sys
    import types

    import lerobot.rl.gym_manipulator as gm
    from lerobot.utils import import_utils

    called: dict = {}

    def fake_hil_make_env(env_id, **kwargs):
        called["env_id"] = env_id
        called["kwargs"] = kwargs
        return "fake_env"

    factory = types.ModuleType("gym_hil.wrappers.factory")
    factory.make_env = fake_hil_make_env
    wrappers = types.ModuleType("gym_hil.wrappers")
    wrappers.factory = factory
    gym_hil = types.ModuleType("gym_hil")
    gym_hil.wrappers = wrappers
    monkeypatch.setitem(sys.modules, "gym_hil", gym_hil)
    monkeypatch.setitem(sys.modules, "gym_hil.wrappers", wrappers)
    monkeypatch.setitem(sys.modules, "gym_hil.wrappers.factory", factory)
    monkeypatch.setattr(import_utils, "require_package", lambda *a, **k: None)
    monkeypatch.setenv("LEROBOT_GYM_HIL_HEADLESS", "1")

    class FakeGripper:
        use_gripper = True
        gripper_penalty = -0.02

    class FakeProc:
        gripper = FakeGripper()

    class FakeCfg:
        name = "gym_hil"
        task = "PandaPickCubeKeyboard-v0"
        robot = None
        teleop = None
        processor = FakeProc()

    env, teleop = gm.make_robot_env(FakeCfg())
    assert env == "fake_env"
    assert teleop is None
    assert called["env_id"] == "gym_hil/PandaPickCubeBase-v0"
    assert called["kwargs"]["use_inputs_control"] is False
