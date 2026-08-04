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


_SEP = "===================================================="
_SUBSEP = "----------------------------------------------------"


def _say(msg: str) -> None:
    print(f"[rl100-collect] {msg}", flush=True)


def _rule(*, sub: bool = False) -> None:
    _say(_SUBSEP if sub else _SEP)


def _totals_str(*, saved: int, fail: int, discard: int, attempts: int) -> str:
    return f"saved={saved}  fail={fail}  discard={discard}  attempts={attempts}"


@dataclass
class LiveEpisodeResult:
    status: str
    episode_id: str
    steps: int
    result_type: str
    path: str | None
    stop_reason: str = "unknown"


def episode_action_quality_errors(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    *,
    min_arm_action_state_delta_rad: float,
    require_gripper_motion: bool,
    min_gripper_action_range: float,
) -> list[str]:
    """Reject invalid demonstration labels before writing an episode."""
    if not states or len(states) != len(actions):
        return [f"state/action length mismatch: {len(states)} != {len(actions)}"]
    state = np.stack(states).astype(np.float32)
    action = np.stack(actions).astype(np.float32)
    if state.shape[1:] != (16,) or action.shape[1:] != (16,):
        return [f"expected state/action (*,16), got {state.shape}/{action.shape}"]
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        return ["state/action contains NaN or Inf"]

    errors: list[str] = []
    arm_idx = np.asarray([*range(7), *range(8, 15)], dtype=np.int64)
    arm_delta = float(np.max(np.abs(action[:, arm_idx] - state[:, arm_idx])))
    if arm_delta < float(min_arm_action_state_delta_rad):
        errors.append(
            "arm action is indistinguishable from measured state "
            f"(max |action-state|={arm_delta:.6g} rad)"
        )

    grip = action[:, [7, 15]]
    if np.any(grip < 0.0) or np.any(grip > 1.0):
        errors.append(
            f"gripper action outside [0,1] (min={grip.min():.6g}, max={grip.max():.6g})"
        )
    grip_range = np.ptp(grip, axis=0)
    if require_gripper_motion and float(np.max(grip_range)) < float(min_gripper_action_range):
        errors.append(
            "no effective gripper command motion "
            f"(left range={grip_range[0]:.6g}, right range={grip_range[1]:.6g})"
        )
    return errors


class HoldStatePolicy:
    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        return state.reshape(1, -1)


def _make_kuavo_gym(deploy_config: Path, *, max_episode_steps: int):
    """Create Kuavo gym env for real or sim (HIL helper only allows Sim)."""
    from kuavo_rl.ros_msg_compat import ensure_foot_pose_6d_msgs

    ensure_foot_pose_6d_msgs()
    import torch  # noqa: F401 — load before cv_bridge (static TLS on some hosts)

    import gymnasium as gym
    import kuavo_deploy.kuavo_env  # noqa: F401
    from kuavo_deploy.config import load_kuavo_config

    deploy_cfg = load_kuavo_config(str(deploy_config))
    env_name = str(deploy_cfg.env.env_name)
    if env_name not in {"Kuavo-Sim", "Kuavo-Real"}:
        raise RuntimeError(
            f"deploy env_name={env_name!r}; expected Kuavo-Sim or Kuavo-Real"
        )
    # Log resolved topics so wrong Brain/upper-cam wiring is obvious immediately.
    obs_map = getattr(deploy_cfg.env, "obs_key_map", {}) or {}
    topic_summary = {
        k: (v.get("topic") if isinstance(v, dict) else v)
        for k, v in obs_map.items()
    }
    _say(f"gym.make({env_name}) deploy={deploy_config}")
    _say(f"obs topics={topic_summary}")
    return gym.make(
        env_name,
        max_episode_steps=int(max_episode_steps),
        config=deploy_cfg,
    )


def run_live_collect_session(config: RL100CollectConfig) -> dict[str, Any]:
    """Long-running VR session: RESET → RECORD → B success/fail → save NPZ."""
    from kuavo_rl.ros_msg_compat import ensure_foot_pose_6d_msgs

    ensure_foot_pose_6d_msgs()

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
        AButtonGestureDetector,
        ModifierStickDetector,
        QuestEpisodeControlEventSource,
        StickEdgeDetector,
        YButtonGestureDetector,
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
    # Align with main HIL: B short=success, B long=failure; Y click/double/long for session.
    episode_control = str(getattr(config, "episode_control", "quest_y_button") or "quest_y_button")
    long_press_s = float(getattr(config, "chord_long_press_s", 0.8) or 0.8)

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

    teleop_raw = dict(raw.get("teleop", {}) or {}) if isinstance(raw, dict) else {}
    teleop_raw["reward_long_press_s"] = long_press_s
    allowed = {
        k: teleop_raw[k]
        for k in RosTeleopConfig.__dataclass_fields__
        if k in teleop_raw
    }
    allowed.setdefault("joystick_topic", config.joystick_topic)
    allowed.setdefault("arm_traj_topic", config.arm_traj_topic)
    # RL-100 collection records the actual Quest/IK command streams. This is
    # intentionally scoped here rather than changing generic HIL intervention.
    allowed["hand_command_topic"] = config.hand_command_topic
    allowed["hand_command_timeout_s"] = config.hand_command_timeout_s
    allowed["qiangnao_scalar_index"] = config.qiangnao_scalar_index
    allowed["record_all_arm_commands"] = True
    allowed["require_hand_command"] = True
    # JoySticks field name — not the UI letter "B".
    allowed.setdefault("reward_button", "right_second_button_pressed")
    allowed.setdefault("reward_gesture", config.reward_gesture)
    allowed.setdefault("reward_stick_threshold", config.reward_stick_threshold)
    allowed.setdefault("reward_stick_rearm_threshold", config.reward_stick_rearm_threshold)
    allowed.setdefault("reward_stick_debounce_s", config.reward_stick_debounce_s)
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

    # /kuavo_arm_traj and /control_robot_hand_position are demand-driven:
    # they only publish while the operator is actively teleoping. A hard
    # preflight would block collection before any motion occurs. Treat as
    # a soft warning; the per-step require_command_action gate still
    # discards episodes that never receive a command stream.
    command_pf = teleop.command_stream_status()
    _say(f"command-stream preflight: {command_pf}")
    if not command_pf["ok"]:
        _say(
            "WARN: arm/hand command stream not yet publishing "
            "(demand-driven topics). Start teleop during RECORD; "
            "episodes without commands will be discarded."
        )

    cal = load_stick_calibration()
    mod = ModifierStickDetector(stick=StickEdgeDetector(calibration=cal))
    event_src = QuestEpisodeControlEventSource(
        mode=episode_control,
        mod_stick=mod,
        a_button=AButtonGestureDetector(double_press_s=0.70, long_press_s=long_press_s),
        y_button=YButtonGestureDetector(double_press_s=0.40, long_press_s=long_press_s),
        calibration=cal,
    )
    try:
        event_src.start()
    except Exception as exc:  # noqa: BLE001
        _say(f"WARN: Quest episode control unavailable ({exc})")
        event_src = None

    results: list[dict[str, Any]] = []
    attempt = 0
    n_saved_success = 0
    n_saved_failure = 0
    n_discard = 0
    lp = f"{long_press_s:g}s"
    _rule()
    _say("RL-100 zarr live collect (REAL)")
    _say(f"task={config.task}  staging={episode_dir}")
    _say(f"deploy={deploy_config}  env={env_config}")
    _say(f"episode_control={episode_control}  ceiling steps={live_max_steps}")
    if episode_control == "quest_y_button":
        _say(f"Y click → start  |  Y double → rerecord  |  Y long (≥{lp}) → end session")
        _say("Right stick free for waist/chassis")
    elif episode_control == "quest_a_button":
        _say(f"A click → start  |  A double → rerecord  |  A long (≥{lp}) → end session")
    else:
        _say(f"Y+stick → start  |  Y+stick ← rerecord  |  Y+stick ↓ (≥{lp}) → end")
    _say(f"B short → SUCCESS  |  B long (≥{lp}) → FAILURE")
    _say("only_success build drops failures; abort/rerecord never staged")
    _rule()

    session_status = "done"
    try:
        while not rospy.is_shutdown():
            next_n = attempt + 1
            _rule()
            _say(
                f"[RESET]  ready  |  next=#{next_n}  "
                + _totals_str(
                    saved=n_saved_success,
                    fail=n_saved_failure,
                    discard=n_discard,
                    attempts=attempt,
                )
            )
            _say(f"         Y click → start   |   Y long (≥{lp}) → end session")
            _rule()
            # Reset once per idle phase (HIL idle_reset); do NOT reset every tick.
            obs, _ = env.reset()
            teleop.set_reference_action(obs["observation.state"])
            teleop.reset()
            teleop.set_label_gestures_enabled(False)
            end_session = False
            while not rospy.is_shutdown():
                # Do NOT call teleop.poll() here: env.step() already polls once.
                # Double-polling eats B short-press edges (success is one-shot).
                ev = event_src.poll() if event_src is not None else None
                et = getattr(ev, "event_type", None) if ev is not None else None
                if et == "right_stick_right":
                    break
                if et in {"right_stick_down", "collection_complete"}:
                    end_session = True
                    break
                # Env applies Quest grip intervention internally from its own poll.
                obs, _, term, trunc, _info = env.step(obs["observation.state"])
                teleop.set_reference_action(obs["observation.state"])
                if term or trunc:
                    obs, _ = env.reset()
                    teleop.set_reference_action(obs["observation.state"])
                    teleop.reset()
                time.sleep(1.0 / max(config.fps, 1.0))

            if end_session or rospy.is_shutdown():
                session_status = "ended" if end_session else session_status
                break

            attempt += 1
            ep = _record_one_episode(
                env=env,
                runner=runner,
                teleop=teleop,
                event_src=event_src,
                pc_hub=pc_hub,
                config=config,
                episode_dir=episode_dir,
                attempt=attempt,
            )
            if ep.status == "saved" and ep.result_type == "success":
                n_saved_success += 1
            elif ep.status == "saved" and ep.result_type == "failure":
                n_saved_failure += 1
            else:
                n_discard += 1
            results.append(
                {
                    "status": ep.status,
                    "episode_id": ep.episode_id,
                    "steps": ep.steps,
                    "result_type": ep.result_type,
                    "path": ep.path,
                    "stop_reason": ep.stop_reason,
                    "attempt": attempt,
                }
            )
            outcome = "SAVED" if ep.status == "saved" else "DISCARDED"
            _rule(sub=True)
            _say(
                f"[DONE  #{attempt}]   {outcome}  {ep.result_type}  "
                f"steps={ep.steps}  reason={ep.stop_reason}"
            )
            if ep.path:
                _say(f"         path={ep.path}")
            _say(
                "         totals: "
                + _totals_str(
                    saved=n_saved_success,
                    fail=n_saved_failure,
                    discard=n_discard,
                    attempts=attempt,
                )
            )
            _rule()
    finally:
        if event_src is not None:
            event_src.close()
        pc_hub.close()
        teleop.close()
        env.close()

    _rule()
    _say(
        f"[SESSION]  {session_status}  |  "
        + _totals_str(
            saved=n_saved_success,
            fail=n_saved_failure,
            discard=n_discard,
            attempts=attempt,
        )
    )
    _say(f"staging={episode_dir}")
    _rule()
    return {
        "status": session_status,
        "results": results,
        "episode_dir": str(episode_dir),
        "attempts": attempt,
        "saved_success": n_saved_success,
        "saved_failure": n_saved_failure,
        "discard": n_discard,
    }


def _label_from_teleop_info(info: dict | None) -> tuple[str, str] | None:
    """Map env info teleop_events → (result_type, stop_reason)."""
    te = (info or {}).get("teleop_events") or {}
    if te.get("success"):
        return "success", "success_button"
    if te.get("failure"):
        return "failure", "failure_button"
    if te.get("abort") or te.get("stop"):
        return "abort", "abort_button" if te.get("abort") else "estop"
    # Manual B also sets reward_source when env consumes the edge in step().
    src = str((info or {}).get("reward_source", "") or "")
    if src == "manual_success":
        return "success", "success_button"
    if src == "manual_failure":
        return "failure", "failure_button"
    if src == "manual_abort":
        return "abort", "abort_button"
    return None


def _action_from_step_info(info: dict | None) -> np.ndarray | None:
    """Prefer the command actually applied / seen by env.step's single teleop.poll()."""
    info = info or {}
    replay = info.get("teleop_replay_action")
    if replay is not None:
        return np.asarray(replay, dtype=np.float32).reshape(-1)
    raw = info.get("teleop_raw_action")
    if raw is not None and info.get("is_intervention"):
        return np.asarray(raw, dtype=np.float32).reshape(-1)
    return None


def _record_one_episode(
    *,
    env,
    runner,
    teleop,
    event_src,
    pc_hub: DepthPointCloudHub,
    config: RL100CollectConfig,
    episode_dir: Path,
    attempt: int = 1,
) -> LiveEpisodeResult:
    """Record state + point cloud every frame. Action = next_state.

    Does NOT depend on /kuavo_arm_traj or /control_robot_hand_command — those
    are demand-driven and only publish while the operator actively controls.
    Instead:
      - state is recorded every frame (always available from /sensors_data_raw)
      - action[t] = state[t+1]  (absolute joint target = next frame's state)
      - last frame action = state[-1]  (hold)
    """
    eid = uuid.uuid4().hex[:12]
    states: list[np.ndarray] = []
    pcs: list[np.ndarray] = []
    result_type = "abort"
    stop_reason = "unknown"
    tag = f"[RECORD #{attempt}]"

    obs, info = env.reset()
    teleop.set_reference_action(obs["observation.state"])
    teleop.reset()
    teleop.set_label_gestures_enabled(True)
    _say(f"{tag}  id={eid}")
    if str(config.reward_gesture).lower().startswith("right_stick"):
        _say("         右摇杆下=success  |  右摇杆上=failure  |  Y双击=rerecord")
    else:
        _say("         B short=success  |  B long=failure  |  Y double=rerecord")

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

        ev = event_src.poll() if event_src is not None else None
        et = getattr(ev, "event_type", None) if ev is not None else None
        if et == "right_stick_left":
            result_type, stop_reason = "abort", "rerecord"
            _say(f"{tag}  Y double-click → rerecord (discard)")
            break
        if et == "right_stick_right":
            _say(f"{tag}  Y click ignored — end with B (short=success, long=failure)")
        elif et in {"right_stick_down", "collection_complete"}:
            _say(f"{tag}  Y long ignored — end with B first, then Y long in RESET")

        try:
            pc = pc_hub.get_point_cloud()
        except Exception as exc:  # noqa: BLE001
            _say(f"{tag}  pointcloud failed: {exc}")
            result_type, stop_reason = "abort", "pointcloud_error"
            break

        # Record state every frame. env.step still polls teleop for B labels.
        state_t = np.asarray(obs["observation.state"], dtype=np.float32).copy()
        # Gripper: if hand command is fresh, keep actual value; else use open (0.0).
        cmd_status = teleop.command_stream_status()
        if not cmd_status.get("hand_ready", False):
            state_t[7] = 0.0   # left gripper open
            state_t[15] = 0.0  # right gripper open
        hold = np.asarray(runner.select_action(obs).action, dtype=np.float32)
        obs, _, term, trunc, info = env.step(hold)
        teleop.set_reference_action(obs["observation.state"])

        # Check B label from env step's teleop poll.
        labeled = _label_from_teleop_info(info)

        states.append(state_t)
        pcs.append(pc.copy())

        if labeled is not None:
            result_type, stop_reason = labeled
            _say(f"{tag}  B → {result_type} — ending episode")
            break

        if term or trunc:
            stop_reason = str(info.get("fault_code", "env_terminate"))
            result_type = "abort"
            _say(
                f"{tag}  env stop at step {len(states)} "
                f"(fault={stop_reason}, source={info.get('reward_source')}) — discard"
            )
            break
        time.sleep(dt)

    # Only discard on explicit rerecord or empty episodes. Everything else
    # (B success, B failure, env terminate, etc.) is kept.
    if stop_reason == "rerecord" or len(states) < 2:
        return LiveEpisodeResult(
            status="discarded",
            episode_id=eid,
            steps=len(states),
            result_type=result_type,
            path=None,
            stop_reason=stop_reason,
        )

    # action[t] = state[t+1]  (absolute joint target = next frame's state)
    state_arr = np.stack(states, axis=0)
    actions = np.zeros_like(state_arr)
    actions[:-1] = state_arr[1:]
    actions[-1] = state_arr[-1]  # last frame: hold

    path = save_episode_npz(
        episode_dir / f"{eid}_{result_type}.npz",
        states=states,
        actions=[actions[i] for i in range(len(actions))],
        point_clouds=pcs,
        result_type=result_type,
        meta={
            "stop_reason": stop_reason,
            "task": config.task,
            "deploy_config": config.deploy_config,
            "env_config": config.env_config,
            "action_source": "next_state",
        },
    )
    return LiveEpisodeResult(
        status="saved",
        episode_id=eid,
        steps=len(states),
        result_type=result_type,
        path=str(path),
        stop_reason=stop_reason,
    )
