"""Runtime monkeypatches for LeRobot HIL-SERL without editing third_party/lerobot.

Handbook constraint: do not fork Kuavo logic into the upstream submodule.
These patches are applied from kuavo_rl CLI wrappers before actor/learner start.
"""

from __future__ import annotations

import logging
import os

_LOGGER = logging.getLogger(__name__)
_APPLIED = False


def apply_hilserl_patches() -> None:
    """Idempotent: patch TrainRL validate + gym_hil/kuavo make_robot_env for Docker/headless."""
    global _APPLIED
    if _APPLIED:
        return
    _patch_train_rl_null_dataset_validate()
    _patch_logging_include_traceback()
    _patch_gym_hil_make_robot_env()
    _patch_kuavo_hilserl_env()
    _APPLIED = True
    _LOGGER.info("kuavo_rl lerobot HIL-SERL runtime patches applied")


def _patch_logging_include_traceback() -> None:
    """Upstream init_logging custom_format drops exc_info; keep stack traces in actor logs."""
    from lerobot.utils import utils as lerobot_utils

    original = lerobot_utils.init_logging

    def init_logging(*args, **kwargs):  # type: ignore[no-untyped-def]
        original(*args, **kwargs)
        import logging

        root = logging.getLogger()
        for handler in root.handlers:
            fmt = handler.formatter
            if fmt is None or not callable(getattr(fmt, "format", None)):
                continue
            inner = fmt.format

            def format_with_exc(record: logging.LogRecord, _inner=inner) -> str:  # type: ignore[no-untyped-def]
                msg = _inner(record)
                if record.exc_info:
                    import traceback

                    # record.exc_info may already be formatted into record.exc_text by Formatter
                    if not getattr(record, "exc_text", None):
                        record.exc_text = "".join(traceback.format_exception(*record.exc_info))
                    if record.exc_text and record.exc_text not in msg:
                        msg = f"{msg}\n{record.exc_text}"
                return msg

            fmt.format = format_with_exc  # type: ignore[method-assign]

    lerobot_utils.init_logging = init_logging  # type: ignore[assignment]


def _patch_train_rl_null_dataset_validate() -> None:
    """Allow dataset=None (online-only) through TrainPipelineConfig.validate()."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.rl.train_rl import TrainRLServerPipelineConfig

    original = TrainRLServerPipelineConfig.validate

    def validate(self):  # type: ignore[no-untyped-def]
        dataset = self.dataset
        if dataset is None:
            self.dataset = DatasetConfig(repo_id="__rl_online_only__")
        try:
            return original(self)
        finally:
            if dataset is None:
                self.dataset = None

    TrainRLServerPipelineConfig.validate = validate  # type: ignore[method-assign]


def _patch_gym_hil_make_robot_env() -> None:
    """Headless Docker: Base factory without X11; optional render_mode override."""
    import gymnasium as gym

    from lerobot.rl import gym_manipulator as gm

    original = gm.make_robot_env

    def make_robot_env(cfg):  # type: ignore[no-untyped-def]
        if getattr(cfg, "name", None) == "kuavo_hilserl":
            return _make_kuavo_hilserl_env(cfg)

        if getattr(cfg, "name", None) != "gym_hil":
            return original(cfg)

        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        from lerobot.utils.import_utils import require_package

        require_package("gym-hil", extra="hilserl", import_name="gym_hil")
        import gym_hil  # noqa: F401

        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        headless = os.environ.get("LEROBOT_GYM_HIL_HEADLESS", "").lower() in {"1", "true", "yes"}
        render_mode = os.environ.get("LEROBOT_GYM_HIL_RENDER_MODE", "human")

        if headless:
            from gym_hil.wrappers.factory import make_env as hil_make_env

            base_task = cfg.task.replace("Keyboard-v0", "Base-v0").replace("Gamepad-v0", "Base-v0")
            if "Base-v0" not in base_task:
                base_task = base_task.replace("-v0", "Base-v0")
            env = hil_make_env(
                f"gym_hil/{base_task}",
                use_viewer=False,
                use_gamepad=False,
                use_gripper=use_gripper,
                use_inputs_control=False,
                show_ui=False,
                gripper_penalty=gripper_penalty,
                image_obs=True,
            )
            return env, None

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode=render_mode,
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )
        return env, None

    gm.make_robot_env = make_robot_env  # type: ignore[assignment]
    # actor.py imports make_robot_env by name; patch both module attrs if already bound.
    try:
        import lerobot.rl.actor as actor_mod

        if getattr(actor_mod, "make_robot_env", None) is not None:
            actor_mod.make_robot_env = make_robot_env  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass


def _make_kuavo_hilserl_env(cfg):  # type: ignore[no-untyped-def]
    """Build Stage-B KuavoHILSerlEnv (MockBackend by default; ROS via KUAVO_HILSERL_BACKEND=ros)."""
    from kuavo_rl.adapter import make_kuavo_hilserl_env
    from kuavo_rl.config import EnvConfig, EpisodeConfig, RewardConfig, default_safety_config
    from kuavo_rl.contracts import IMAGE_KEYS, IMAGE_SHAPE_CHW

    # Prefer policy/env feature shapes when present (smoke may use 128x128).
    image_shape = IMAGE_SHAPE_CHW
    features = getattr(cfg, "features", None) or {}
    for key in IMAGE_KEYS:
        feat = features.get(key)
        shape = getattr(feat, "shape", None) if feat is not None else None
        if shape is None and isinstance(feat, dict):
            shape = feat.get("shape")
        if shape is not None and len(shape) == 3:
            image_shape = tuple(int(x) for x in shape)
            break

    max_steps = 20
    if cfg.processor is not None and cfg.processor.reset is not None:
        max_steps = max(1, int(cfg.processor.reset.control_time_s * cfg.fps))

    safety = default_safety_config()
    safety.max_consecutive_clips = 50  # SAC exploration soft-clips often
    env_cfg = EnvConfig(
        fps=int(cfg.fps),
        task=str(cfg.task or "box_to_chest_mvp"),
        shadow_mode=os.environ.get("KUAVO_HILSERL_SHADOW", "").lower() in {"1", "true", "yes"},
        image_shape_chw=image_shape,
        safety=safety,
        episode=EpisodeConfig(max_steps=max_steps, max_duration_s=float(max_steps) / max(cfg.fps, 1)),
        reward=RewardConfig(use_robometer=False, robometer_mode="disabled"),
    )

    backend_mode = os.environ.get("KUAVO_HILSERL_BACKEND", "mock").lower()
    kuavo_gym_env = None
    backend = None
    if backend_mode == "ros":
        import gymnasium as gym
        import kuavo_deploy.kuavo_env  # noqa: F401
        from kuavo_deploy.config import load_kuavo_config

        deploy_path = os.environ.get(
            "KUAVO_DEPLOY_CONFIG",
            "configs/deploy/total/deploy_sim_smoke_cams_total.yaml",
        )
        deploy_cfg = load_kuavo_config(deploy_path)
        kuavo_gym_env = gym.make(
            deploy_cfg.env.env_name,
            max_episode_steps=int(deploy_cfg.inference.max_episode_steps),
            config=deploy_cfg,
        )
    elif backend_mode == "proxy":
        from kuavo_rl.backend import ProxyBackend

        backend = ProxyBackend(
            host=os.environ.get("KUAVO_ROS_BRIDGE_HOST", "127.0.0.1"),
            port=int(os.environ.get("KUAVO_ROS_BRIDGE_PORT", "8877")),
            image_shape_chw=image_shape,
        )

    env = make_kuavo_hilserl_env(
        env_cfg,
        backend=backend,
        kuavo_gym_env=kuavo_gym_env,
        use_stub_robometer=True,
    )
    return env, None


def _patch_kuavo_hilserl_env() -> None:
    """Route name=kuavo_hilserl processors through kuavo_rl helpers."""
    from lerobot.rl import gym_manipulator as gm

    original_processors = gm.make_processors

    def make_processors(env, teleop_device, cfg, device="cpu"):  # type: ignore[no-untyped-def]
        if getattr(cfg, "name", None) == "kuavo_hilserl":
            from kuavo_rl.hilserl_processors import make_kuavo_processors

            return make_kuavo_processors(cfg, device=device)
        return original_processors(env, teleop_device, cfg, device)

    gm.make_processors = make_processors  # type: ignore[assignment]
    try:
        import lerobot.rl.actor as actor_mod

        if getattr(actor_mod, "make_processors", None) is not None:
            actor_mod.make_processors = make_processors  # type: ignore[assignment]
    except Exception:  # noqa: BLE001
        pass
