"""Live VR collection → episode NPZ staging for RL-100 zarr.

Aligned with real-robot HIL collect (Kuavo-Real + Quest B labels).
Does not modify HIL recording defaults.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kuavo_rl.rl100_zarr.config import RL100CollectConfig
from kuavo_rl.rl100_zarr.ros_depth import CameraSyncError, DepthPointCloudHub
from kuavo_rl.rl100_zarr.ros_state import TopicStateHub
from kuavo_rl.rl100_zarr.staging import save_episode_npz
from kuavo_rl.contracts import RL100_ACTION_DIM, RL100_ARM_SLICE_RAW20, RL100_STATE_DIM


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
    min_arm_action_state_delta_rad: float = 0.0,
    require_gripper_motion: bool = False,
    min_gripper_action_range: float = 0.0,
    audit: dict[str, np.ndarray] | None = None,
) -> list[str]:
    """Reject invalid demonstration labels before writing an episode."""
    if not states or len(states) != len(actions):
        return [f"state/action length mismatch: {len(states)} != {len(actions)}"]
    state = np.stack(states).astype(np.float32)
    action = np.stack(actions).astype(np.float32)
    if state.shape[1:] != (RL100_STATE_DIM,) or action.shape[1:] != (RL100_ACTION_DIM,):
        return [
            f"expected state/action (*,{RL100_STATE_DIM})/(*,{RL100_ACTION_DIM}), "
            f"got {state.shape}/{action.shape}"
        ]
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
        return ["state/action contains NaN or Inf"]

    errors: list[str] = []
    if np.any(state[:, 20:] < 0.0) or np.any(state[:, 20:] > 100.0):
        errors.append("dexhand state outside [0,100]")
    arm_deg = action[:, :14]
    hand = action[:, 14:]
    if np.any(hand < 0.0) or np.any(hand > 100.0):
        errors.append(
            f"hand action outside [0,100] (min={hand.min():.6g}, max={hand.max():.6g})"
        )
    hand_range = np.ptp(hand, axis=0)
    if require_gripper_motion and float(np.max(hand_range)) < float(min_gripper_action_range):
        errors.append(
            "no effective hand command motion "
            f"(max per-finger range={np.max(hand_range):.6g})"
        )
    if not np.isfinite(arm_deg).all():
        errors.append("arm command contains NaN/Inf")
    if audit is not None:
        arm_seen = np.asarray(audit.get("arm_command_seen", []), dtype=bool)
        hand_seen = np.asarray(audit.get("hand_command_seen", []), dtype=bool)
        if not (arm_seen.shape == hand_seen.shape == (len(actions),)):
            errors.append("command seen audit shape mismatch")
        elif not bool(arm_seen[0] or hand_seen[0]) or not np.any(arm_seen | hand_seen):
            errors.append("first sample has no valid arm/hand command after record start")
        cutoff = np.asarray(audit.get("sample_cutoff_received_at", []), dtype=np.float64)
        arm_received = np.asarray(audit.get("arm_command_received_at", []), dtype=np.float64)
        hand_received = np.asarray(audit.get("hand_command_received_at", []), dtype=np.float64)
        if cutoff.shape == arm_received.shape == hand_received.shape == (len(actions),):
            if (
                not np.isfinite(cutoff).all()
                or not np.isfinite(arm_received).all()
                or not np.isfinite(hand_received).all()
            ):
                errors.append("command timing audit contains NaN/Inf")
            elif np.any(arm_received > cutoff) or np.any(hand_received > cutoff):
                errors.append("command causality violation: received_at > sample cutoff")
            if np.any(np.diff(arm_received) < 0.0) or np.any(np.diff(hand_received) < 0.0):
                errors.append("command received_at is not monotonic")
        else:
            errors.append("command timing audit shape mismatch")
        for key in ("arm_command_stamp", "hand_command_stamp"):
            stamps = np.asarray(audit.get(key, []), dtype=np.float64)
            if stamps.shape == (len(actions),):
                valid = stamps > 0.0
                if valid.sum() > 1 and np.any(np.diff(stamps[valid]) < -1e-6):
                    errors.append(f"{key} is not monotonic for valid header stamps")
    return errors


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(arr.size),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def _display_source(value: Any) -> str:
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def episode_action_quality_report(
    states: list[np.ndarray],
    actions: list[np.ndarray],
    *,
    audit: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Return inspectable ranges and timing ratios for one topic-native episode."""
    state = np.stack(states).astype(np.float32)
    action = np.stack(actions).astype(np.float32)
    report: dict[str, Any] = {
        "state_shape": list(state.shape),
        "action_shape": list(action.shape),
        "state_joint20": {
            "min": state[:, :20].min(axis=0).tolist(),
            "max": state[:, :20].max(axis=0).tolist(),
        },
        "state_hand12": {
            "min": state[:, 20:].min(axis=0).tolist(),
            "max": state[:, 20:].max(axis=0).tolist(),
            "range": np.ptp(state[:, 20:], axis=0).tolist(),
        },
        "action_arm14_deg": {
            "min": action[:, :14].min(axis=0).tolist(),
            "max": action[:, :14].max(axis=0).tolist(),
            "range": np.ptp(action[:, :14], axis=0).tolist(),
        },
        "action_hand12_raw": {
            "min": action[:, 14:].min(axis=0).tolist(),
            "max": action[:, 14:].max(axis=0).tolist(),
            "range": np.ptp(action[:, 14:], axis=0).tolist(),
        },
    }
    if audit is not None:
        arm_changed = np.asarray(audit.get("arm_command_changed", []), dtype=bool)
        hand_changed = np.asarray(audit.get("hand_command_changed", []), dtype=bool)
        report["arm_command_changed_ratio"] = float(arm_changed.mean()) if arm_changed.size else None
        report["hand_command_changed_ratio"] = float(hand_changed.mean()) if hand_changed.size else None
        report["initial_command_source"] = {
            "arm": _display_source(np.asarray(audit.get("arm_command_source", ["unknown"]), dtype=object)[0]),
            "hand": _display_source(np.asarray(audit.get("hand_command_source", ["unknown"]), dtype=object)[0]),
        }
        for key, out_key in (
            ("joint_age", "joint_age_s"),
            ("hand_age", "hand_age_s"),
            ("joint_hand_skew", "joint_hand_skew_s"),
        ):
            report[out_key] = _summary_stats(np.asarray(audit.get(key, []), dtype=np.float64))
        cutoff = np.asarray(audit.get("sample_cutoff_received_at", []), dtype=np.float64)
        arm_received = np.asarray(audit.get("arm_command_received_at", []), dtype=np.float64)
        hand_received = np.asarray(audit.get("hand_command_received_at", []), dtype=np.float64)
        if cutoff.shape == arm_received.shape == hand_received.shape and cutoff.size:
            hold_age = np.maximum(cutoff - np.maximum(arm_received, hand_received), 0.0)
            report["command_hold_duration_s"] = _summary_stats(hold_age)
            report["causality_violation_count"] = int(
                np.count_nonzero((arm_received > cutoff) | (hand_received > cutoff))
            )
        else:
            report["command_hold_duration_s"] = _summary_stats(np.asarray([], dtype=np.float64))
            report["causality_violation_count"] = None
        report["camera_sync"] = {
            "header_skew_s": _summary_stats(
                np.asarray(audit.get("point_cloud_header_skew", []), dtype=np.float64)
            ),
            "receive_skew_s": _summary_stats(
                np.asarray(audit.get("point_cloud_receive_skew", []), dtype=np.float64)
            ),
            "received_age_s": {
                name: _summary_stats(
                    np.asarray(audit.get(f"{name}_depth_age", []), dtype=np.float64)
                )
                for name in ("head", "left", "right")
            },
            "valid_points": _summary_stats(
                np.asarray(audit.get("point_cloud_valid_points", []), dtype=np.float64)
            ),
        }
    return report


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
        # KuavoBaseRosEnv adds timestamp metadata to reset()/get_obs(), while
        # its legacy observation_space only declares observation.state.  The
        # RL100 adapter normalizes and validates the final observation contract
        # below, so the legacy passive checker must not reject reset first.
        disable_env_checker=True,
        config=deploy_cfg,
    )


def run_live_collect_session(config: RL100CollectConfig) -> dict[str, Any]:
    """Long-running VR session: RESET → RECORD → B success/fail → save NPZ."""
    from kuavo_rl.ros_msg_compat import ensure_foot_pose_6d_msgs

    ensure_foot_pose_6d_msgs()

    config.validate_contract()
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
    allowed["record_all_arm_commands"] = True
    # Arm and hand command streams are independent.  An arm-only phase must
    # still be recordable before the first hand command arrives.
    allowed["require_hand_command"] = False
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

    state_hub = TopicStateHub(
        sensors_topic=config.sensors_topic,
        dexhand_state_topic=config.dexhand_state_topic,
    )
    state_hub.start()
    state_pf = state_hub.preflight(timeout_s=8.0)
    _say(f"raw state preflight: {state_pf}")
    if not state_pf.get("ok"):
        state_hub.close()
        pc_hub.close()
        teleop.close()
        env.close()
        raise RuntimeError(f"raw state preflight failed: {state_pf}")

    # /kuavo_arm_traj and /control_robot_hand_position are demand-driven:
    # they only publish while the operator is actively teleoping. A hard
    # preflight would block collection before any motion occurs. Treat as
    # a soft warning; the per-step any-command gate still discards episodes
    # that never receive a command stream.
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
                state_hub=state_hub,
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
        state_hub.close()
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
    state_hub: TopicStateHub,
    attempt: int = 1,
) -> LiveEpisodeResult:
    """Record raw state32 and command-cache action26 at a common sample cutoff."""
    eid = uuid.uuid4().hex[:12]
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    pcs: list[np.ndarray] = []
    audit: dict[str, list[Any]] = {
        "sample_stamp": [],
        "sample_cutoff_received_at": [],
        "joint_state_stamp": [],
        "dexhand_state_stamp": [],
        "arm_command_stamp": [],
        "hand_command_stamp": [],
        "joint_state_received_at": [],
        "dexhand_state_received_at": [],
        "arm_command_received_at": [],
        "hand_command_received_at": [],
        "joint_sensor_time": [],
        "joint_hand_skew": [],
        "joint_age": [],
        "hand_age": [],
        "point_cloud_stamp": [],
        "point_cloud_received_at": [],
        "point_cloud_age": [],
        "point_cloud_camera_skew": [],
        "point_cloud_header_skew": [],
        "point_cloud_receive_skew": [],
        "point_cloud_reference_stamp": [],
        "point_cloud_reference_camera": [],
        "head_depth_stamp": [],
        "left_depth_stamp": [],
        "right_depth_stamp": [],
        "head_depth_received_at": [],
        "left_depth_received_at": [],
        "right_depth_received_at": [],
        "head_depth_age": [],
        "left_depth_age": [],
        "right_depth_age": [],
        "point_cloud_valid_points": [],
        "arm_command_changed": [],
        "hand_command_changed": [],
        "arm_command_seen": [],
        "hand_command_seen": [],
        "arm_command_source": [],
        "hand_command_source": [],
    }
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

    try:
        initial = state_hub.snapshot(
            config.joint_state_max_age_s,
            config.dexhand_state_max_age_s,
            config.state_max_skew_s,
        )
    except Exception as exc:  # noqa: BLE001
        _say(f"{tag}  raw state unavailable at record start: {exc}")
        return LiveEpisodeResult("discarded", eid, 0, result_type, None, "state_preflight")

    default_hand = np.asarray(config.hand_default, dtype=np.float32)
    default_error = float(np.max(np.abs(initial.dexhand_position12 - default_hand)))
    if default_error > float(config.hand_default_tolerance):
        _say(
            f"{tag}  hand is not at configured default: max error={default_error:.3f} "
            f"> tolerance={config.hand_default_tolerance:.3f}; restore hand and retry"
        )
        return LiveEpisodeResult("discarded", eid, 0, result_type, None, "hand_default_mismatch")
    teleop.begin_topic_native_episode(initial.raw_joint_q20[RL100_ARM_SLICE_RAW20])

    dt = 1.0 / max(config.fps, 1.0)
    t0 = time.monotonic()
    max_steps = max(int(config.live_max_steps), 1)
    max_duration_s = float(config.live_max_duration_s)
    source_failures = 0
    camera_sync_failures = 0
    camera_sync_attempts = 0
    camera_sync_successes = 0
    camera_sync_failure_codes: Counter[str] = Counter()
    invalid_counts_before = Counter(pc_hub.invalid_counts)

    try:
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

            cutoff = time.monotonic()
            try:
                command = teleop.topic_native_snapshot(cutoff)
            except Exception as exc:  # noqa: BLE001
                _say(f"{tag}  command cache failed: {exc}")
                result_type, stop_reason = "abort", "command_cache_error"
                break

            has_command = bool(command.arm_seen or command.hand_seen)
            state_sample = None
            cloud_sample = None
            if has_command or not config.start_on_any_command:
                try:
                    state_sample = state_hub.snapshot(
                        config.joint_state_max_age_s,
                        config.dexhand_state_max_age_s,
                        config.state_max_skew_s,
                        cutoff_monotonic_s=cutoff,
                    )
                    camera_sync_attempts += 1
                    cloud_sample = pc_hub.get_point_cloud_sample(
                        cutoff_monotonic_s=cutoff,
                        max_received_age_s=min(
                            config.depth_max_age_s,
                            config.camera_max_received_age_s,
                        ),
                        max_header_skew_s=config.camera_max_header_skew_s,
                        max_receive_skew_s=config.camera_max_receive_skew_s,
                    )
                    camera_sync_successes += 1
                    source_failures = 0
                    camera_sync_failures = 0
                except Exception as exc:  # noqa: BLE001
                    is_camera_sync = isinstance(exc, CameraSyncError)
                    if is_camera_sync:
                        camera_sync_failure_codes[exc.code] += 1
                        camera_sync_failures += 1
                        failure_count = camera_sync_failures
                        failure_limit = config.camera_max_consecutive_sync_failures
                        failure_label = "camera sync"
                    else:
                        source_failures += 1
                        failure_count = source_failures
                        failure_limit = config.max_consecutive_source_failures
                        failure_label = "source"
                    if failure_count >= failure_limit:
                        _say(f"{tag}  {failure_label} failed {failure_count} times: {exc}")
                        result_type, stop_reason = "abort", (
                            "camera_sync_error" if is_camera_sync else "source_error"
                        )
                        break
                    _say(
                        f"{tag}  {failure_label} frame skipped ({failure_count}/"
                        f"{failure_limit}): {exc}"
                    )
                    # Still run one env step so B/Y labels remain responsive;
                    # no state/action sample is appended for this stale frame.
                    state_sample = None
                    cloud_sample = None

            # env.step still polls teleop for B labels, but its 16-D hold action
            # is unrelated to the RL-100 dataset action.
            hold = np.asarray(runner.select_action(obs).action, dtype=np.float32)
            obs, _, term, trunc, info = env.step(hold)
            teleop.set_reference_action(obs["observation.state"])
            labeled = _label_from_teleop_info(info)

            if has_command and state_sample is not None and cloud_sample is not None:
                states.append(state_sample.state32.copy())
                actions.append(command.action26.copy())
                pcs.append(cloud_sample.points.copy())
                audit["sample_stamp"].append(time.time())
                audit["sample_cutoff_received_at"].append(cutoff)
                audit["joint_state_stamp"].append(state_sample.joint_stamp_s)
                audit["dexhand_state_stamp"].append(state_sample.hand_stamp_s)
                audit["arm_command_stamp"].append(command.arm_header_stamp_s)
                audit["hand_command_stamp"].append(command.hand_header_stamp_s)
                audit["joint_state_received_at"].append(state_sample.joint_received_at_s)
                audit["dexhand_state_received_at"].append(state_sample.hand_received_at_s)
                audit["arm_command_received_at"].append(command.arm_received_at_s)
                audit["hand_command_received_at"].append(command.hand_received_at_s)
                audit["joint_sensor_time"].append(state_sample.joint_sensor_time_s)
                audit["joint_hand_skew"].append(state_sample.joint_hand_skew_s)
                audit["joint_age"].append(state_sample.joint_age_s)
                audit["hand_age"].append(state_sample.hand_age_s)
                audit["point_cloud_stamp"].append(cloud_sample.fused_stamp_s)
                audit["point_cloud_received_at"].append(cloud_sample.received_at_s)
                audit["point_cloud_age"].append(cloud_sample.oldest_age_s)
                audit["point_cloud_camera_skew"].append(cloud_sample.max_camera_skew_s)
                audit["point_cloud_header_skew"].append(cloud_sample.max_header_skew_s)
                audit["point_cloud_receive_skew"].append(cloud_sample.max_receive_skew_s)
                audit["point_cloud_reference_stamp"].append(cloud_sample.reference_stamp_s)
                audit["point_cloud_reference_camera"].append(cloud_sample.reference_camera)
                camera_stamps = cloud_sample.camera_stamps
                camera_received = cloud_sample.camera_received_wall_s or {}
                camera_ages = cloud_sample.camera_received_ages_s or {}
                for camera_name, key in (
                    ("head_cam_h", "head"),
                    ("wrist_cam_l", "left"),
                    ("wrist_cam_r", "right"),
                ):
                    audit[f"{key}_depth_stamp"].append(camera_stamps.get(camera_name, 0.0))
                    audit[f"{key}_depth_received_at"].append(camera_received.get(camera_name, 0.0))
                    audit[f"{key}_depth_age"].append(camera_ages.get(camera_name, float("inf")))
                audit["point_cloud_valid_points"].append(cloud_sample.valid_points)
                audit["arm_command_changed"].append(command.arm_changed)
                audit["hand_command_changed"].append(command.hand_changed)
                audit["arm_command_seen"].append(command.arm_seen)
                audit["hand_command_seen"].append(command.hand_seen)
                audit["arm_command_source"].append(command.arm_source)
                audit["hand_command_source"].append(command.hand_source)

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
    finally:
        teleop.end_topic_native_episode()

    # Only discard on explicit rerecord or empty episodes. Everything else
    # (B success, B failure, env terminate, etc.) is kept.
    if stop_reason == "rerecord" or len(states) == 0:
        return LiveEpisodeResult(
            status="discarded",
            episode_id=eid,
            steps=len(states),
            result_type=result_type,
            path=None,
            stop_reason=stop_reason,
        )

    audit_arrays = {key: np.asarray(values) for key, values in audit.items()}
    quality_errors = episode_action_quality_errors(
        states,
        actions,
        require_gripper_motion=config.require_hand_motion,
        min_gripper_action_range=config.min_hand_action_range,
        audit=audit_arrays,
    )
    if quality_errors:
        _say(f"{tag}  quality gate rejected episode: {quality_errors}")
        return LiveEpisodeResult(
            "discarded", eid, len(states), result_type, None, "quality_gate"
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
            "action_source": "topic_command_cache",
            "contract": config.contract,
            "state_dim": config.state_dim,
            "action_dim": config.action_dim,
            "state_order": "raw_joint_q20 + dexhand_position12",
            "action_order": "arm14_deg + left_hand6_raw + right_hand6_raw",
            "camera_sync": {
                "label": (
                    "UNSYNCHRONIZED_LEGACY"
                    if config.camera_sync_mode == "latest_legacy"
                    else "BUFFERED_HEADER"
                ),
                "mode": config.camera_sync_mode,
                "reference_camera": config.camera_reference_camera,
                "buffer_size": config.camera_buffer_size,
                "max_header_skew_s": config.camera_max_header_skew_s,
                "max_receive_skew_s": config.camera_max_receive_skew_s,
                "max_received_age_s": min(
                    config.depth_max_age_s,
                    config.camera_max_received_age_s,
                ),
                "tf_at_image_stamp": config.camera_tf_at_image_stamp,
                "tf_timeout_s": config.camera_tf_timeout_s,
                "sync_attempts": camera_sync_attempts,
                "sync_successes": camera_sync_successes,
                "sync_success_rate": (
                    camera_sync_successes / camera_sync_attempts
                    if camera_sync_attempts
                    else 0.0
                ),
                "failure_codes": dict(camera_sync_failure_codes),
                "invalid_counts": {
                    key: int(value - invalid_counts_before.get(key, 0))
                    for key, value in pc_hub.invalid_counts.items()
                    if value > invalid_counts_before.get(key, 0)
                },
            },
        },
        audit=audit_arrays,
    )
    return LiveEpisodeResult(
        status="saved",
        episode_id=eid,
        steps=len(states),
        result_type=result_type,
        path=str(path),
        stop_reason=stop_reason,
    )
