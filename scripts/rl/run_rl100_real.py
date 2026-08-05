#!/usr/bin/env python3
"""RL-100 BC deployment CLI for Kuavo real robot.

All motion is opt-in: the default config is shadow-only and the ``live`` command
requires CLI confirmation, a one-time token, a declared physical E-stop check,
field-approved limits and a field-approved startup pose.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


class JsonlAudit:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def __call__(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _write_manifest(path: Path, cfg, policy, *, live: bool) -> None:
    point_cfg_path = Path(cfg.raw["observation"]["pointcloud_config"])
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        git_commit = "unknown"
    manifest = {
        "mode": "live" if live else "shadow",
        "checkpoint": policy.info.__dict__,
        "contract": getattr(policy.info, "contract", ""),
        "deploy_config_path": str(cfg.path),
        "deploy_config": cfg.raw,
        "pointcloud_config_path": str(point_cfg_path),
        "pointcloud_config_text": point_cfg_path.read_text(encoding="utf-8"),
        "control_hz": cfg.control_hz,
        "ros_master_uri": os.environ.get("ROS_MASTER_URI", ""),
        "ros_ip": os.environ.get("ROS_IP", ""),
        "git_commit": git_commit,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _policy_from_config(cfg):
    from kuavo_rl.rl100_policy import RL100Policy

    infer = cfg.raw.get("inference", {})
    use_cm = infer.get("use_cm", "auto")
    policy = RL100Policy.from_checkpoint(
        cfg.checkpoint_path,
        device=cfg.device,
        model_source=cfg.model_source,
        deterministic=bool(infer.get("deterministic", False)),
        distill2mean=bool(infer.get("distill2mean", False)),
        use_cm=None if use_cm == "auto" else bool(use_cm),
    )
    if use_cm != "auto" and bool(use_cm) != policy.info.use_cm:
        raise RuntimeError(
            "deployment inference.use_cm conflicts with checkpoint cfg.policy.use_cm; "
            "use auto or retrain/choose a compatible checkpoint"
        )
    return policy


def _validate_camera_sync_contract(deploy_cfg, point_cfg) -> None:
    """Reject deployment/point-cloud YAML drift before subscribing for control."""

    deploy_obs = deploy_cfg.raw.get("observation", {})
    deploy_sync = deploy_obs.get("camera_sync", {}) or {}
    expected = {
        "label": (
            "UNSYNCHRONIZED_LEGACY"
            if str(deploy_sync.get("mode", "buffered_header")) == "latest_legacy"
            else "BUFFERED_HEADER"
        ),
        "mode": str(deploy_sync.get("mode", "buffered_header")),
        "reference_camera": str(deploy_sync.get("reference_camera", "head_cam_h")),
        "buffer_size": int(deploy_sync.get("buffer_size", 32)),
        "max_header_skew_s": float(
            deploy_sync.get("max_header_skew_s", deploy_obs.get("max_camera_skew_s", 0.10))
        ),
        "warn_header_skew_s": float(deploy_sync.get("warn_header_skew_s", 0.05)),
        "max_received_age_config_s": float(deploy_sync.get("max_received_age_s", 0.20)),
        "max_receive_skew_s": float(deploy_sync.get("max_receive_skew_s", 0.20)),
        "require_monotonic_header": bool(deploy_sync.get("require_monotonic_header", True)),
        "require_same_time_epoch": bool(deploy_sync.get("require_same_time_epoch", True)),
        "tf_at_image_stamp": bool(deploy_sync.get("tf_at_image_stamp", True)),
        "tf_timeout_s": float(deploy_sync.get("tf_timeout_s", 0.05)),
        "max_consecutive_sync_failures": int(
            deploy_sync.get("max_consecutive_sync_failures", 10)
        ),
        "max_received_age_s": float(deploy_obs.get("depth_max_age_s", 0.15)),
    }
    actual = {
        "label": (
            "UNSYNCHRONIZED_LEGACY"
            if point_cfg.camera_sync_mode == "latest_legacy"
            else "BUFFERED_HEADER"
        ),
        "mode": point_cfg.camera_sync_mode,
        "reference_camera": point_cfg.camera_reference_camera,
        "buffer_size": point_cfg.camera_buffer_size,
        "max_header_skew_s": point_cfg.camera_max_header_skew_s,
        "warn_header_skew_s": point_cfg.camera_warn_header_skew_s,
        "max_received_age_config_s": point_cfg.camera_max_received_age_s,
        "max_receive_skew_s": point_cfg.camera_max_receive_skew_s,
        "require_monotonic_header": point_cfg.camera_require_monotonic_header,
        "require_same_time_epoch": point_cfg.camera_require_same_time_epoch,
        "tf_at_image_stamp": point_cfg.camera_tf_at_image_stamp,
        "tf_timeout_s": point_cfg.camera_tf_timeout_s,
        "max_consecutive_sync_failures": point_cfg.camera_max_consecutive_sync_failures,
        "max_received_age_s": min(
            point_cfg.depth_max_age_s,
            point_cfg.camera_max_received_age_s,
        ),
    }
    mismatches: list[str] = []
    for key, expected_value in expected.items():
        actual_value = actual[key]
        if isinstance(expected_value, float):
            equal = bool(np.isclose(expected_value, actual_value))
        else:
            equal = expected_value == actual_value
        if not equal:
            mismatches.append(
                f"{key}: deploy={expected_value!r}, pointcloud={actual_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            "deployment and point-cloud camera_sync contracts differ: "
            + "; ".join(mismatches)
        )


def cmd_inspect_checkpoint(args: argparse.Namespace) -> int:
    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config

    cfg = load_rl100_deploy_config(args.config)
    policy = _policy_from_config(cfg)
    out = {"checkpoint": policy.info.__dict__, "warmup": policy.warmup(int(cfg.raw["inference"].get("warmup_runs", 3)))}
    _print(out)
    return 0


def _make_ros_components(cfg):
    from kuavo_rl.rl100_zarr.config import load_rl100_collect_config
    from kuavo_rl.rl100_zarr.ros_depth import DepthPointCloudHub
    from kuavo_rl.rl100_zarr.ros_state import TopicStateHub
    from kuavo_rl.rl100_real_runner import RL100TopicCommandPublisher

    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node("rl100_real_deploy", anonymous=True)
    point_cfg = load_rl100_collect_config(cfg.raw["observation"]["pointcloud_config"])
    _validate_camera_sync_contract(cfg, point_cfg)
    if point_cfg.num_points != 1024 or point_cfg.state_dim != 32 or point_cfg.action_dim != 26:
        raise RuntimeError("pointcloud collection config must preserve the 1024/32/26 training contract")
    hub = DepthPointCloudHub(point_cfg)
    hub.start()
    state_hub = TopicStateHub(
        sensors_topic=str(cfg.raw["observation"].get("raw_joint_topic", "/sensors_data_raw")),
        dexhand_state_topic=str(cfg.raw["observation"].get("dexhand_state_topic", "/dexhand/state")),
    )
    state_hub.start()
    publisher = RL100TopicCommandPublisher(
        arm_topic=str(cfg.raw.get("publish", {}).get("arm_topic", "/kuavo_arm_traj")),
        hand_topic=str(cfg.raw.get("publish", {}).get("hand_topic", "/control_robot_hand_position")),
    )
    publisher.start()
    return hub, state_hub, publisher


def _ros_preflight(
    cfg, *, timeout_s: float, profile_s: float = 0.0
) -> tuple[dict[str, Any], tuple[Any, Any, Any]]:
    hub, state_hub, publisher = _make_ros_components(cfg)
    limits = cfg.runner_limits()
    point_report = hub.preflight(timeout_s=timeout_s, profile_s=profile_s)
    state_report = state_hub.preflight(timeout_s=min(timeout_s, 5.0), profile_s=profile_s)
    connections = publisher.connection_counts()
    publishers_ok = all(value > 0 for value in connections.values())
    if not point_report.get("ok", False) or not state_report.get("ok", False):
        return {
            "ok": False,
            "pointcloud": point_report,
            "state": state_report,
            "publishers": {
                "connections": connections,
                "ok": publishers_ok,
                "expected_types": {
                    "arm": "sensor_msgs/JointState",
                    "hand": "kuavo_msgs/robotHandPosition",
                },
            },
        }, (hub, state_hub, publisher)
    try:
        cloud = hub.get_point_cloud_sample(
            max_received_age_s=limits.depth_max_age_s,
            max_header_skew_s=limits.max_camera_skew_s,
            max_receive_skew_s=limits.max_camera_receive_skew_s,
        )
        state = state_hub.snapshot(
            limits.joint_state_max_age_s,
            limits.dexhand_state_max_age_s,
            limits.state_max_skew_s,
        )
        report = {
            "ok": bool(publishers_ok or not cfg.require_publish_subscribers),
            "pointcloud": point_report,
            "state": state_report,
            "sample": {
                "shape": list(cloud.points.shape),
                "oldest_age_s": cloud.oldest_age_s,
                "max_camera_skew_s": cloud.max_camera_skew_s,
                "max_receive_skew_s": cloud.max_receive_skew_s,
                "reference_camera": cloud.reference_camera,
                "camera_stamps": cloud.camera_stamps,
            },
            "robot_state": {
                "state_shape": list(state.state32.shape),
                "raw_joint_dim": list(state.raw_joint_q20.shape)[0],
                "dexhand_dim": list(state.dexhand_position12.shape)[0],
                "timing": {
                    "joint_stamp_s": state.joint_stamp_s,
                    "hand_stamp_s": state.hand_stamp_s,
                    "joint_age_s": state.joint_age_s,
                    "hand_age_s": state.hand_age_s,
                },
            },
            "publishers": {
                "connections": connections,
                "ok": publishers_ok,
                "required": cfg.require_publish_subscribers,
                "expected_types": {
                    "arm": "sensor_msgs/JointState",
                    "hand": "kuavo_msgs/robotHandPosition",
                },
            },
        }
    except Exception as exc:  # noqa: BLE001
        report = {
            "ok": False,
            "pointcloud": point_report,
            "state": state_report,
            "publishers": {"connections": connections, "ok": publishers_ok},
            "error": str(exc),
        }
    return report, (hub, state_hub, publisher)


def cmd_ros_preflight(args: argparse.Namespace) -> int:
    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config

    cfg = load_rl100_deploy_config(args.config)
    components = None
    try:
        report, components = _ros_preflight(
            cfg,
            timeout_s=min(float(args.duration_s), 10.0),
            profile_s=float(args.duration_s),
        )
        _print(report)
        return 0 if report.get("ok") else 2
    finally:
        if components is not None:
            hub, state_hub, publisher = components
            hub.close()
            state_hub.close()
            publisher.close()


def _check_live_arm(cfg, state_hub, publisher, args: argparse.Namespace) -> None:
    if not args.confirm_live or not args.physical_estop_ready or not args.live_token:
        raise RuntimeError("live requires --confirm-live --physical-estop-ready and a non-empty --live-token")
    configured_token = str(cfg.raw.get("mode", {}).get("live_confirmation_token", ""))
    if configured_token and args.live_token != configured_token:
        raise RuntimeError("live confirmation token does not match deployment config")
    cfg.safety_gate(require_approved=True)
    connections = publisher.connection_counts()
    if cfg.require_publish_subscribers and not all(value > 0 for value in connections.values()):
        raise RuntimeError(f"live requires one subscriber on each command topic, got {connections}")
    limits = cfg.runner_limits()
    state = state_hub.snapshot(
        limits.joint_state_max_age_s,
        limits.dexhand_state_max_age_s,
        limits.state_max_skew_s,
    )
    startup = cfg.startup_requirements(require_approved=True)
    if startup["require_physical_estop_ready"] and not args.physical_estop_ready:
        raise RuntimeError("physical E-stop readiness was not confirmed")
    default_hand = np.asarray(cfg.raw.get("startup", {}).get("hand_default", [0, 99, 0, 0, 0, 0, 0, 99, 0, 0, 0, 0]), dtype=np.float32)
    if default_hand.shape != (12,):
        raise RuntimeError("startup.hand_default must be 12-D")
    hand_error = float(np.max(np.abs(state.dexhand_position12 - default_hand)))
    if hand_error > float(startup["hand_default_tolerance"]):
        raise RuntimeError(f"startup hand default mismatch: max error={hand_error:.3f}")


def _run_realtime(args: argparse.Namespace, *, live: bool) -> int:
    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config
    from kuavo_rl.rl100_real_runner import RL100TopicRealRunner

    cfg = load_rl100_deploy_config(args.config)
    components = None
    runner = None
    try:
        preflight, components = _ros_preflight(cfg, timeout_s=float(args.preflight_timeout_s))
        if not preflight.get("ok"):
            _print(preflight)
            return 2
        hub, state_hub, publisher = components
        if live:
            _check_live_arm(cfg, state_hub, publisher, args)
        policy = _policy_from_config(cfg)
        log_dir = Path(cfg.raw.get("logging", {}).get("output_dir", "logs/rl100_real"))
        run_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{'live' if live else 'shadow'}"
        audit = JsonlAudit(log_dir / f"{run_stem}.jsonl")
        _write_manifest(log_dir / f"{run_stem}.manifest.json", cfg, policy, live=live)
        limits = cfg.runner_limits()
        runner = RL100TopicRealRunner(
            policy=policy,
            state_hub=state_hub,
            point_cloud_source=lambda cutoff_monotonic_s=None: hub.get_point_cloud_sample(
                cutoff_monotonic_s=cutoff_monotonic_s,
                max_received_age_s=limits.depth_max_age_s,
                max_header_skew_s=limits.max_camera_skew_s,
                max_receive_skew_s=limits.max_camera_receive_skew_s,
            ),
            publisher=publisher,
            safety=cfg.safety_gate(require_approved=live),
            limits=limits,
            shadow_mode=not live,
            audit_sink=audit,
            stop_source=lambda: (False, False, bool(__import__("rospy").is_shutdown())),
        )
        runner.preflight()
        if live:
            runner.arm_live()
        max_steps = min(int(args.max_steps), int(cfg.raw["safety"].get("max_episode_steps", 200)))
        duration_s = min(float(args.duration_s), float(cfg.raw["safety"].get("max_episode_duration_s", 20.0)))
        deadline = time.monotonic() + duration_s
        results = []
        for _ in range(max_steps):
            if time.monotonic() >= deadline:
                break
            tick_started = time.monotonic()
            result = runner.tick()
            results.append(result)
            if result.state.value == "FAULT":
                break
            time.sleep(max(0.0, (1.0 / cfg.control_hz) - (time.monotonic() - tick_started)))
        _print({
            "ok": all(r.fault_code.value == "NONE" for r in results),
            "mode": "live" if live else "shadow",
            "steps": len(results),
            "published": sum(r.published for r in results),
            "last": results[-1].record if results else None,
            "audit_log": str(audit.path),
            "manifest": str(audit.path.with_suffix(".manifest.json")),
        })
        return 0 if results and results[-1].state.value != "FAULT" else 2
    finally:
        if runner is not None:
            runner.close()
        if components is not None:
            hub, state_hub, publisher = components
            hub.close()
            state_hub.close()
            publisher.close()


def cmd_offline_replay(args: argparse.Namespace) -> int:
    import numpy as np
    import zarr

    from kuavo_rl.contracts import RL100_ACTION_DIM, RL100_STATE_DIM, RL100_TOPIC_NATIVE_CONTRACT
    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config
    from kuavo_rl.rl100_real_runner import RL100TopicObservationHistory

    cfg = load_rl100_deploy_config(args.config)
    if "grasp_8_4.zarr" in str(args.zarr_path) and "grasp_8_4_v2" not in str(args.zarr_path):
        raise RuntimeError("refusing known-invalid old grasp_8_4.zarr")
    policy = _policy_from_config(cfg)
    root = zarr.open(str(args.zarr_path), mode="r")
    attrs = root.attrs
    if attrs.get("contract") != RL100_TOPIC_NATIVE_CONTRACT:
        raise RuntimeError(
            "offline zarr contract must be "
            f"{RL100_TOPIC_NATIVE_CONTRACT!r}, got {attrs.get('contract')!r}"
        )
    if int(attrs.get("state_dim", -1)) != RL100_STATE_DIM or int(attrs.get("action_dim", -1)) != RL100_ACTION_DIM:
        raise RuntimeError(
            "offline zarr attrs must declare state_dim=32/action_dim=26, "
            f"got {attrs.get('state_dim')!r}/{attrs.get('action_dim')!r}"
        )
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    points = np.asarray(root["data"]["point_cloud"][:], dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != 32 or actions.ndim != 2 or actions.shape[1] != 26:
        raise RuntimeError(f"offline zarr must contain state(T,32)/action(T,26), got {states.shape}/{actions.shape}")
    history = RL100TopicObservationHistory(policy.info.n_obs_steps)
    predictions = []
    count = min(len(states), int(args.max_steps))
    for i in range(count):
        history.append(points[i], states[i])
        pc_hist, state_hist = history.arrays()
        predictions.append(policy.predict(pc_hist, state_hist)[0])
    pred = np.asarray(predictions, dtype=np.float32)
    target = actions[:count]
    err = abs(pred - target)
    _print({
        "zarr_path": str(args.zarr_path),
        "steps": count,
        "all_finite": bool(np.isfinite(pred).all()),
        "mae": float(err.mean()),
        "arm_deg_mae": float(err[:, :14].mean()),
        "hand_raw_mae": float(err[:, 14:].mean()),
        "prediction_min": pred.min(axis=0).tolist(),
        "prediction_max": pred.max(axis=0).tolist(),
    })
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/rl/rl100_real_deploy.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RL-100 Kuavo real-robot deploy")
    sub = parser.add_subparsers(dest="cmd", required=True)
    inspect = sub.add_parser("inspect-checkpoint", help="load and warm up workspace .ckpt")
    _common(inspect); inspect.set_defaults(func=cmd_inspect_checkpoint)
    replay = sub.add_parser("offline-replay", help="replay valid RL-100 zarr without ROS")
    _common(replay); replay.add_argument("--zarr-path", required=True); replay.add_argument("--max-steps", type=int, default=1000); replay.set_defaults(func=cmd_offline_replay)
    preflight = sub.add_parser("ros-preflight", help="check live sources without publish")
    _common(preflight); preflight.add_argument("--duration-s", type=float, default=60.0); preflight.set_defaults(func=cmd_ros_preflight)
    shadow = sub.add_parser("shadow", help="real-time predict only; never publish")
    _common(shadow); shadow.add_argument("--max-steps", type=int, default=500); shadow.add_argument("--duration-s", type=float, default=60.0); shadow.add_argument("--preflight-timeout-s", type=float, default=10.0); shadow.set_defaults(func=lambda a: _run_realtime(a, live=False))
    live = sub.add_parser("live", help="explicitly confirmed finite real-robot publish")
    _common(live); live.add_argument("--max-steps", type=int, default=1); live.add_argument("--duration-s", type=float, default=5.0); live.add_argument("--preflight-timeout-s", type=float, default=10.0); live.add_argument("--confirm-live", action="store_true"); live.add_argument("--physical-estop-ready", action="store_true"); live.add_argument("--live-token", default=""); live.set_defaults(func=lambda a: _run_realtime(a, live=True))
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
