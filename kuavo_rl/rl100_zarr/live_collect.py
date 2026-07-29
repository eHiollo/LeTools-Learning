"""Live VR collection → episode NPZ staging for RL-100 zarr.

Aligned with real-robot HIL collect (Kuavo-Real + Quest B labels).
Does not modify HIL recording defaults.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kuavo_rl.rl100_zarr.config import RL100CollectConfig
from kuavo_rl.rl100_zarr.ros_depth import DepthPointCloudHub
from kuavo_rl.rl100_zarr.staging import save_episode_npz


def _say(msg: str) -> None:
    print(f"[rl100-collect] {msg}", flush=True)


@dataclass
class LiveEpisodeResult:
    status: str
    episode_id: str
    steps: int
    result_type: str
    path: str | None


class HoldStatePolicy:
    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        return state.reshape(1, -1)


def _make_kuavo_gym(deploy_config: Path, *, max_episode_steps: int):
    """Create Kuavo gym env for real or sim (HIL helper only allows Sim)."""
    import gymnasium as gym
    import kuavo_deploy.kuavo_env  # noqa: F401
    from kuavo_deploy.config import load_kuavo_config

    deploy_cfg = load_kuavo_config(str(deploy_config))
    env_name = str(deploy_cfg.env.env_name)
    if env_name not in {"Kuavo-Sim", "Kuavo-Real"}:
        raise RuntimeError(
            f"deploy env_name={env_name!r}; expected Kuavo-Sim or Kuavo-Real"
        )
    _say(f"gym.make({env_name}) deploy={deploy_config}")
    return gym.make(
        env_name,
        max_episode_steps=int(max_episode_steps),
        config=deploy_cfg,
    )


def run_live_collect_session(config: RL100CollectConfig) -> dict[str, Any]:
    """Long-running VR session: RESET → RECORD → B success/fail → save NPZ."""
    if not config.confirm_live:
        raise RuntimeError(
            "refusing live motion without confirm_live=true / --confirm-live"
        )

    import rospy

    from kuavo_rl.act_runner import ActExecuteFirstRunner
    from kuavo_rl.adapter import make_kuavo_hilserl_env
    from kuavo_rl.config import ActRunnerConfig, build_env_config_from_dict, load_yaml
    from kuavo_rl.hil_collect_live import _quiet_robot_logs
    from kuavo_rl.quest_episode_control import (
        ModifierStickDetector,
        QuestEpisodeControlEventSource,
        StickEdgeDetector,
        load_stick_calibration,
    )
    from kuavo_rl.ros_teleop import RosTeleopAdapter, RosTeleopConfig

    if not rospy.core.is_initialized():
        rospy.init_node("rl100_zarr_collect", anonymous=True)

    episode_dir = config.output_dir() / "episodes"
    episode_dir.mkdir(parents=True, exist_ok=True)

    deploy_config = Path(config.deploy_config)
    env_config = Path(config.env_config)
    live_max_steps = max(int(config.live_max_steps), 1)
    live_max_duration_s = float(config.live_max_duration_s)

    raw = load_yaml(env_config) if env_config.exists() else {"env": {}}
    env_cfg = build_env_config_from_dict(raw)
    env_cfg.shadow_mode = bool(getattr(config, "shadow_mode", True))
    # Override short MVP episode limits — B ends episodes (same as HIL VR collect).
    if getattr(env_cfg, "episode", None) is not None:
        env_cfg.episode.max_steps = live_max_steps
        env_cfg.episode.max_duration_s = live_max_duration_s
    if getattr(env_cfg, "safety", None) is not None:
        env_cfg.safety.max_consecutive_clips = 0

    _quiet_robot_logs()
    kuavo_gym = _make_kuavo_gym(deploy_config, max_episode_steps=live_max_steps)

    teleop_raw = raw.get("teleop", {}) if isinstance(raw, dict) else {}
    allowed = {
        k: teleop_raw[k]
        for k in RosTeleopConfig.__dataclass_fields__
        if k in teleop_raw
    }
    allowed.setdefault("joystick_topic", config.joystick_topic)
    allowed.setdefault("arm_traj_topic", config.arm_traj_topic)
    # JoySticks field name — not the UI letter "B".
    allowed.setdefault("reward_button", "right_second_button_pressed")
    teleop = RosTeleopAdapter(RosTeleopConfig(**allowed))
    teleop.start()

    env = make_kuavo_hilserl_env(
        env_cfg,
        kuavo_gym_env=kuavo_gym,
        use_stub_robometer=True,
        teleop=teleop,
    )
    runner = ActExecuteFirstRunner(
        HoldStatePolicy(),
        ActRunnerConfig(chunk_size=1, execute_steps=1, fps=int(config.fps)),
    )
    _quiet_robot_logs()

    pc_hub = DepthPointCloudHub(config)
    pc_hub.start()
    pf = pc_hub.preflight(timeout_s=8.0)
    _say(f"pointcloud preflight: {pf}")
    if not pf.get("ok"):
        pc_hub.close()
        teleop.close()
        env.close()
        raise RuntimeError(f"depth/tf preflight failed: {pf}")

    cal = load_stick_calibration()
    event_src = QuestEpisodeControlEventSource(
        mode="quest_y_stick",
        mod_stick=ModifierStickDetector(stick=StickEdgeDetector(calibration=cal)),
        calibration=cal,
    )
    try:
        event_src.start()
    except Exception as exc:  # noqa: BLE001
        _say(f"WARN: Quest episode control unavailable ({exc})")
        event_src = None

    results: list[dict[str, Any]] = []
    _say("========== RL-100 zarr live collect (REAL) ==========")
    _say(f"task={config.task}  staging={episode_dir}")
    _say(f"deploy={deploy_config}  env={env_config}")
    _say(
        f"episode ceiling steps={live_max_steps} duration_s={live_max_duration_s} "
        f"(B ends episode; defaults match HIL safety)"
    )
    _say("B click=success  B double=failure  B hold=abort (abort discarded)")
    _say("Y+stick → start   Y+stick ← rerecord   Y+stick ↓ end session")
    _say("Failures are staged by default; use build --only-success to drop them")
    _say("====================================================")

    try:
        while not rospy.is_shutdown():
            _say("RESET — teleop ok, not recording. Y+→ to start.")
            # Reset once per idle phase (HIL idle_reset); do NOT reset every tick.
            obs, _ = env.reset()
            teleop.set_reference_action(obs["observation.state"])
            while not rospy.is_shutdown():
                te = teleop.poll()
                ev = event_src.poll() if event_src is not None else None
                et = getattr(ev, "event_type", None) if ev is not None else None
                if et == "right_stick_right":
                    break
                if et in {"right_stick_down", "collection_complete"}:
                    _say("session end requested")
                    return {
                        "status": "ended",
                        "results": results,
                        "episode_dir": str(episode_dir),
                    }
                if te is not None and te.is_intervention and te.action is not None:
                    obs, _, term, trunc, _info = env.step(te.action)
                else:
                    obs, _, term, trunc, _info = env.step(obs["observation.state"])
                teleop.set_reference_action(obs["observation.state"])
                if term or trunc:
                    obs, _ = env.reset()
                    teleop.set_reference_action(obs["observation.state"])
                time.sleep(1.0 / max(config.fps, 1.0))

            if rospy.is_shutdown():
                break

            ep = _record_one_episode(
                env=env,
                runner=runner,
                teleop=teleop,
                event_src=event_src,
                pc_hub=pc_hub,
                config=config,
                episode_dir=episode_dir,
            )
            results.append(
                {
                    "status": ep.status,
                    "episode_id": ep.episode_id,
                    "steps": ep.steps,
                    "result_type": ep.result_type,
                    "path": ep.path,
                }
            )
            _say(
                f"episode status={ep.status} type={ep.result_type} "
                f"steps={ep.steps} path={ep.path}"
            )
    finally:
        if event_src is not None:
            event_src.close()
        pc_hub.close()
        teleop.close()
        env.close()

    return {"status": "done", "results": results, "episode_dir": str(episode_dir)}


def _record_one_episode(
    *,
    env,
    runner,
    teleop,
    event_src,
    pc_hub: DepthPointCloudHub,
    config: RL100CollectConfig,
    episode_dir: Path,
) -> LiveEpisodeResult:
    eid = uuid.uuid4().hex[:12]
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    pcs: list[np.ndarray] = []
    result_type = "abort"
    stop_reason = "unknown"

    obs, info = env.reset()
    teleop.set_reference_action(obs["observation.state"])
    _say(f"RECORD {eid} — end with B success/fail")

    dt = 1.0 / max(config.fps, 1.0)
    t0 = time.monotonic()
    max_steps = max(int(config.live_max_steps), 1)
    max_duration_s = float(config.live_max_duration_s)

    while True:
        if len(states) >= max_steps:
            result_type, stop_reason = "abort", "max_steps"
            break
        if (time.monotonic() - t0) >= max_duration_s:
            result_type, stop_reason = "abort", "max_duration"
            break

        te = teleop.poll()
        if te is not None and te.success:
            result_type, stop_reason = "success", "success_button"
            break
        if te is not None and te.failure:
            result_type, stop_reason = "failure", "failure_button"
            break
        if te is not None and te.abort:
            result_type, stop_reason = "abort", "abort_button"
            break

        ev = event_src.poll() if event_src is not None else None
        et = getattr(ev, "event_type", None) if ev is not None else None
        if et == "right_stick_left":
            result_type, stop_reason = "abort", "rerecord"
            break
        if et in {"right_stick_down", "collection_complete"}:
            result_type, stop_reason = "abort", "collection_complete"
            break

        if te is not None and te.is_intervention and te.action is not None:
            action = np.asarray(te.action, dtype=np.float32)
        else:
            action = np.asarray(runner.select_action(obs).action, dtype=np.float32)

        try:
            pc = pc_hub.get_point_cloud()
        except Exception as exc:  # noqa: BLE001
            _say(f"pointcloud failed: {exc}")
            result_type, stop_reason = "abort", "pointcloud_error"
            break

        states.append(np.asarray(obs["observation.state"], dtype=np.float32).copy())
        actions.append(action.copy())
        pcs.append(pc.copy())

        obs, _, term, trunc, info = env.step(action)
        teleop.set_reference_action(obs["observation.state"])
        if term or trunc:
            if result_type == "abort" and stop_reason == "unknown":
                stop_reason = str(info.get("fault_code", "env_terminate"))
            break
        time.sleep(dt)

    if result_type == "abort" or not states:
        return LiveEpisodeResult(
            status="discarded",
            episode_id=eid,
            steps=len(states),
            result_type=result_type,
            path=None,
        )

    path = save_episode_npz(
        episode_dir / f"{eid}_{result_type}.npz",
        states=states,
        actions=actions,
        point_clouds=pcs,
        result_type=result_type,
        meta={
            "stop_reason": stop_reason,
            "task": config.task,
            "deploy_config": config.deploy_config,
            "env_config": config.env_config,
        },
    )
    return LiveEpisodeResult(
        status="saved",
        episode_id=eid,
        steps=len(states),
        result_type=result_type,
        path=str(path),
    )
