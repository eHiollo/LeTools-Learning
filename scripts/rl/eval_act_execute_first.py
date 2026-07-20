#!/usr/bin/env python3
"""Load Stage-A ACT checkpoint and verify execute-first (chunk=10 -> take step 0).

Default path uses MockBackend env (no ROS). For live Kuavo-Sim:
  - start Docker infer: bash scripts/rl/run_act_infer_server.sh
  - host: bash scripts/rl/run_act_kuavo_sim_eval.sh
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _mock_obs_from_dataset(root: Path) -> dict:
    """Build one observation matching dataset feature keys (for dry-run)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="lerobot_merged", root=str(root), video_backend="pyav")
    sample = ds[0]
    obs = {}
    for k, v in sample.items():
        if k.startswith("observation."):
            if torch.is_tensor(v):
                obs[k] = v.detach().cpu().numpy()
            else:
                obs[k] = v
    return obs


def _make_policy(args):
    if args.policy == "remote":
        from kuavo_rl.act_policy import RemoteActChunkPolicy

        policy = RemoteActChunkPolicy(host=args.infer_host, port=args.infer_port)
        policy.connect()
        return policy, "remote"

    from kuavo_rl.act_policy import LerobotActChunkPolicy

    ckpt = args.checkpoint
    if not ckpt.exists():
        alt = Path("data/rl_runs/act_stage_a_latest")
        candidates = list(alt.glob("checkpoints/*/pretrained_model")) if alt.exists() else []
        if not candidates:
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        ckpt = candidates[-1]
    print(f"[eval] loading ACT from {ckpt}")
    return LerobotActChunkPolicy.from_checkpoint(str(ckpt), device=args.device), str(ckpt)


def _make_kuavo_env(deploy_config: Path):
    import gymnasium as gym
    import kuavo_deploy.kuavo_env  # noqa: F401
    from kuavo_deploy.config import load_kuavo_config

    deploy_cfg = load_kuavo_config(str(deploy_config))
    if deploy_cfg.env.env_name != "Kuavo-Sim":
        raise RuntimeError(
            f"deploy config env_name={deploy_cfg.env.env_name!r}; expected Kuavo-Sim "
            f"(set inference_env: sim in {deploy_config})"
        )
    return gym.make(
        deploy_cfg.env.env_name,
        max_episode_steps=int(deploy_cfg.inference.max_episode_steps),
        config=deploy_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/rl_runs/checkpoints/005000/pretrained_model"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot/lerobot_merged"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kuavo-env", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("local", "remote"),
        default="local",
        help="local=load ACT in-process (needs Py>=3.11 lerobot); remote=TCP infer server",
    )
    parser.add_argument("--infer-host", default="127.0.0.1")
    parser.add_argument("--infer-port", type=int, default=8765)
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=Path("configs/deploy/total/deploy_sim_smoke_cams_total.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/kuavo_hilserl_sim_act.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rl_runs/act_execute_first_eval/manifest.json"),
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Predict + safety only; never publish policy actions",
    )
    parser.add_argument(
        "--ros-teleop",
        action="store_true",
        help="Consume existing Quest3 IK output as human intervention (ROS required)",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path("data/rl_runs/hilserl_episodes"),
        help="Write per-step HIL audit JSONL here for Kuavo runs.",
    )
    parser.add_argument(
        "--record-experiment",
        default="hilserl_vr",
        help="Subdirectory name under --record-dir.",
    )
    parser.add_argument(
        "--hil-recording",
        action="store_true",
        help="Use HILRecordingSession (staging→accepted_replay, SQLite, dry-run rosbag).",
    )
    parser.add_argument(
        "--hil-recording-live-rosbag",
        action="store_true",
        help="With --hil-recording, start real rosbag (requires ROS).",
    )
    parser.add_argument(
        "--hil-topics-profile",
        type=Path,
        default=None,
        help="Topic profile yaml for HILRecordingSession (default: hil_topics_v002; sim live: hil_topics_sim_v002).",
    )
    args = parser.parse_args()

    from kuavo_rl.act_runner import ActExecuteFirstRunner
    from kuavo_rl.adapter import make_kuavo_hilserl_env
    from kuavo_rl.config import ActRunnerConfig, build_env_config_from_dict, load_yaml
    from kuavo_rl.recording import (
        EpisodeRecorder,
        HILReplayWriter,
        TransitionRecord,
        build_manifest,
        write_manifest,
    )

    policy, policy_id = _make_policy(args)

    # Dry-run chunk shape when local + dataset available; remote uses zero obs.
    chunk_shape = None
    if args.policy == "local":
        obs_ds = _mock_obs_from_dataset(args.dataset_root)
        chunk = policy.predict_action_chunk(obs_ds)
        chunk_shape = list(chunk.shape)
        print(f"[eval] dataset-obs chunk shape={chunk.shape} (expect (10, 16))")
        if chunk.ndim != 2 or chunk.shape[1] != 16:
            raise RuntimeError(f"unexpected chunk shape {chunk.shape}")
    else:
        from kuavo_rl.contracts import IMAGE_KEYS, IMAGE_SHAPE_CHW

        warm = {"observation.state": np.zeros(16, dtype=np.float32)}
        for k in IMAGE_KEYS:
            warm[k] = np.zeros(IMAGE_SHAPE_CHW, dtype=np.float32)
        chunk = policy.predict_action_chunk(warm)
        chunk_shape = list(chunk.shape)
        print(f"[eval] remote warmup chunk shape={chunk.shape}")

    raw = load_yaml(args.config) if args.config.exists() else {"env": {}}
    cfg = build_env_config_from_dict(raw)
    if args.shadow:
        cfg.shadow_mode = True

    kuavo_gym_env = None
    mode = "mock"
    if args.kuavo_env:
        try:
            kuavo_gym_env = _make_kuavo_env(args.deploy_config)
            mode = "kuavo_sim"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Kuavo-Sim unavailable ({exc}); using mock")
            mode = "mock_fallback"

    teleop = None
    if args.ros_teleop:
        from kuavo_rl.ros_teleop import RosTeleopAdapter, RosTeleopConfig

        teleop_raw = raw.get("teleop", {}) if isinstance(raw, dict) else {}
        allowed = {k: teleop_raw[k] for k in RosTeleopConfig.__dataclass_fields__ if k in teleop_raw}
        teleop = RosTeleopAdapter(RosTeleopConfig(**allowed))
        teleop.start()
    env = make_kuavo_hilserl_env(
        cfg, kuavo_gym_env=kuavo_gym_env, use_stub_robometer=True, teleop=teleop
    )
    runner = ActExecuteFirstRunner(policy, ActRunnerConfig(chunk_size=10, execute_steps=1))

    summary = {
        "mode": mode,
        "policy": args.policy,
        "policy_id": policy_id,
        "checkpoint": str(args.checkpoint),
        "chunk_shape": chunk_shape,
        "discarded_tail_len": int(chunk.shape[0] - 1),
        "execute_first_ok": True,
        "shadow_mode": bool(cfg.shadow_mode),
        "ros_teleop": bool(args.ros_teleop),
    }

    recorder = None
    replay_writer = None
    hil_session = None
    if mode.startswith("kuavo"):
        if args.hil_recording:
            import os

            from kuavo_rl.hil_recording import (
                FINALIZED_OK,
                HILRecordingSession,
                RecordingConfig,
                RecordRequest,
                ResultEvent,
                now_stamps,
            )

            root = args.record_dir / args.record_experiment
            live = bool(args.hil_recording_live_rosbag)
            profile = args.hil_topics_profile
            if profile is None and live:
                sim_profile = Path("configs/rl/hil_topics_sim_v002.yaml")
                if sim_profile.exists():
                    profile = sim_profile
            hil_cfg = RecordingConfig(
                root_dir=root,
                dry_run_recorder=not live,
                skip_gate_ros=not live,
                post_roll_s=0.5,
                bag_stall_timeout_s=30.0 if live else 5.0,
            )
            hil_session = HILRecordingSession(hil_cfg, profile_path=profile)
            hil_session.recover_interrupted()
            episode_id = f"ep_{int(time.time())}"
            hil_session.create(
                RecordRequest(
                    episode_id=episode_id,
                    task_id=args.record_experiment,
                    control_profile="act_vr" if args.ros_teleop else "act",
                    dry_run=not live,
                    skip_gate_ros=not live,
                )
            )
            hil_session.register_producer("act_runner", os.getpid(), kind="policy")
            hil_session.start(episode_id)
            replay_writer = HILReplayWriter(
                args.record_dir,
                args.record_experiment,
                staging_dir=hil_cfg.staging_dir(episode_id),
            )

            def record_step(step: dict) -> None:
                info = step["info"]
                # Force episode_id from session for staging consistency
                info = {**info, "episode_id": episode_id}
                audit = info.get("action_audit", {})
                executed = step.get("executed_action") or audit.get("raw_action")
                policy_action = np.asarray(step["policy_action"], dtype=np.float32).tolist()
                stamps = now_stamps()
                transition = TransitionRecord(
                    experiment_id=args.record_experiment,
                    episode_id=episode_id,
                    step_id=int(step["step_id"]),
                    timestamp=float(stamps.wall_time_ns) / 1e9,
                    action=list(executed) if executed is not None else policy_action,
                    reward=float(step["reward"]),
                    reward_source=str(info.get("reward_source", "unknown")),
                    terminated=bool(step["terminated"]),
                    truncated=bool(step["truncated"]),
                    fault_code=str(info.get("fault_code", "UNKNOWN")),
                    is_intervention=bool(info.get("is_intervention", False)),
                    action_clipped=bool(info.get("action_clipped", False)),
                    extras={
                        "policy_action": policy_action,
                        "executed_action": executed,
                        "chunk_len": int(step["chunk_len"]),
                        "discarded": int(step["discarded"]),
                        "teleop_source": info.get("teleop_source", "none"),
                        "teleop_age_s": info.get("teleop_age_s"),
                        "teleop_raw_action": info.get("teleop_raw_action"),
                        "teleop_replay_action": info.get("teleop_replay_action"),
                        "intervention_mask": info.get("intervention_mask"),
                        "intervention_segment_id": info.get("intervention_segment_id", 0),
                        "intervention_segment_step": info.get("intervention_segment_step", 0),
                        "teleop_events": info.get("teleop_events", {}),
                        "stamps": stamps.to_dict(),
                    },
                )
                replay_writer.log_transition(
                    transition,
                    observation=step["observation"],
                    next_observation=step["next_observation"],
                )
                hil_session.update_transition(
                    {
                        "step_id": int(step["step_id"]),
                        "stamps": stamps,
                        "policy_action": policy_action,
                        "executed_action": executed,
                        "teleop_raw_action": info.get("teleop_raw_action"),
                        "teleop_replay_action": info.get("teleop_replay_action"),
                        "intervention_mask": info.get("intervention_mask"),
                        "intervention_segment_id": info.get("intervention_segment_id", 0),
                        "intervention_segment_step": info.get("intervention_segment_step", 0),
                        "reward": float(step["reward"]),
                        "fault_code": str(info.get("fault_code", "UNKNOWN")),
                    }
                )
                stop_req = hil_session.poll_stop_request()
                if stop_req is not None:
                    raise RuntimeError(f"watchdog_stop:{stop_req.reason}")

            result = {"n": 0, "steps": []}
            reason = "success"
            try:
                result = runner.run_episode(env, max_steps=args.steps, on_step=record_step)
                if result["steps"]:
                    last = result["steps"][-1]["info"]
                    fault = str(last.get("fault_code", "NONE"))
                    if "ESTOP" in fault.upper():
                        reason = "estop"
                    elif last.get("terminated") and float(result["steps"][-1]["reward"]) < 0:
                        reason = "failure"
                hil_session.record_event(
                    ResultEvent(
                        episode_id=episode_id,
                        event_type=reason,
                        source="eval_act_execute_first",
                        stamps=now_stamps(),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                reason = "estop" if "estop" in str(exc).lower() else "fault"
                hil_session.record_event(
                    ResultEvent(
                        episode_id=episode_id,
                        event_type=reason,
                        source="eval_act_execute_first",
                        stamps=now_stamps(),
                        payload={"error": str(exc)},
                    )
                )
                raise
            finally:
                replay_writer.close()
                hil_session.request_stop(episode_id, reason=reason)
                final = hil_session.wait_finalized(episode_id, timeout_s=60.0)
                summary["hil_session_state"] = final.session_state
                summary["hil_export_status"] = final.replay_export_status
                summary["hil_result_type"] = final.result_type
                if final.session_state in FINALIZED_OK:
                    export = hil_session.publish_replay(episode_id)
                    summary["hil_publish"] = export.to_dict()
                    summary["replay_path"] = export.path
                else:
                    summary["replay_path"] = str(hil_cfg.quarantine_dir / episode_id)
                hil_session.close()
                hil_session = None
                replay_writer = None

            summary["steps"] = result["n"]
            summary["recording_path"] = str(root / "hil_recording.db")
            summary["discarded_per_step"] = (
                result["steps"][0]["discarded"] if result["steps"] else None
            )
        else:
            recorder = EpisodeRecorder(args.record_dir, args.record_experiment)
            replay_writer = HILReplayWriter(args.record_dir, args.record_experiment)

            def record_step(step: dict) -> None:
                info = step["info"]
                audit = info.get("action_audit", {})
                executed = step.get("executed_action") or audit.get("raw_action")
                policy_action = np.asarray(step["policy_action"], dtype=np.float32).tolist()
                transition = TransitionRecord(
                    experiment_id=args.record_experiment,
                    episode_id=str(info["episode_id"]),
                    step_id=int(step["step_id"]),
                    timestamp=float(info["timestamp"]),
                    action=list(executed) if executed is not None else policy_action,
                    reward=float(step["reward"]),
                    reward_source=str(info.get("reward_source", "unknown")),
                    terminated=bool(step["terminated"]),
                    truncated=bool(step["truncated"]),
                    fault_code=str(info.get("fault_code", "UNKNOWN")),
                    is_intervention=bool(info.get("is_intervention", False)),
                    action_clipped=bool(info.get("action_clipped", False)),
                    extras={
                        "policy_action": policy_action,
                        "executed_action": executed,
                        "chunk_len": int(step["chunk_len"]),
                        "discarded": int(step["discarded"]),
                        "teleop_source": info.get("teleop_source", "none"),
                        "teleop_age_s": info.get("teleop_age_s"),
                        "teleop_raw_action": info.get("teleop_raw_action"),
                        "teleop_replay_action": info.get("teleop_replay_action"),
                        "intervention_mask": info.get("intervention_mask"),
                        "intervention_segment_id": info.get("intervention_segment_id", 0),
                        "intervention_segment_step": info.get("intervention_segment_step", 0),
                        "teleop_events": info.get("teleop_events", {}),
                    },
                )
                recorder.log(transition)
                replay_writer.log_transition(
                    transition,
                    observation=step["observation"],
                    next_observation=step["next_observation"],
                )

            result = runner.run_episode(env, max_steps=args.steps, on_step=record_step)
            recorder.close()
            replay_writer.close()
            recorder = None
            replay_writer = None
            summary["steps"] = result["n"]
            summary["recording_path"] = str(args.record_dir / args.record_experiment / "transitions.jsonl")
            summary["replay_path"] = str(args.record_dir / args.record_experiment / "replay")
            summary["discarded_per_step"] = result["steps"][0]["discarded"] if result["steps"] else None
            if result["steps"]:
                summary["last_fault"] = result["steps"][-1]["info"].get("fault_code")
                summary["last_reward"] = result["steps"][-1]["reward"]
    else:
        # Mock: validate select_action contract with synthetic/training-like obs
        from kuavo_rl.contracts import IMAGE_KEYS, IMAGE_SHAPE_CHW

        if args.policy == "local":
            step_obs = obs_ds  # noqa: F821 — set in local branch above
        else:
            step_obs = {"observation.state": np.zeros(16, dtype=np.float32)}
            for k in IMAGE_KEYS:
                step_obs[k] = np.zeros(IMAGE_SHAPE_CHW, dtype=np.float32)
        step = runner.select_action(step_obs)
        summary["executed_index"] = step.executed_index
        summary["discarded_tail_len"] = int(step.discarded_tail.shape[0])
        summary["action_dim"] = int(step.action.shape[0])
        assert step.executed_index == 0
        assert step.discarded_tail.shape[0] == chunk.shape[0] - 1
        assert len(runner.pending_queue) == 0

    if recorder is not None:
        recorder.close()
    if replay_writer is not None:
        replay_writer.close()
    env.close()
    if hasattr(policy, "close"):
        policy.close()
    man = build_manifest("act_execute_first_eval", extra=summary)
    write_manifest(args.manifest, man)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
