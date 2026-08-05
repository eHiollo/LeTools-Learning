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


def cmd_inspect_checkpoint(args: argparse.Namespace) -> int:
    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config

    cfg = load_rl100_deploy_config(args.config)
    policy = _policy_from_config(cfg)
    out = {"checkpoint": policy.info.__dict__, "warmup": policy.warmup(int(cfg.raw["inference"].get("warmup_runs", 3)))}
    _print(out)
    return 0


def _make_ros_components(cfg):
    from kuavo_rl.backend import ROSBackend
    from kuavo_rl.rl100_zarr.config import load_rl100_collect_config
    from kuavo_rl.rl100_zarr.live_collect import _make_kuavo_gym
    from kuavo_rl.rl100_zarr.ros_depth import DepthPointCloudHub

    import rospy

    if not rospy.core.is_initialized():
        rospy.init_node("rl100_real_deploy", anonymous=True)
    point_cfg = load_rl100_collect_config(cfg.raw["observation"]["pointcloud_config"])
    if point_cfg.num_points != 1024 or point_cfg.state_dim != 16 or point_cfg.action_dim != 16:
        raise RuntimeError("pointcloud collection config must preserve the 1024/16/16 training contract")
    hub = DepthPointCloudHub(point_cfg)
    hub.start()
    robot_cfg = cfg.raw["robot"]
    kuavo_gym = _make_kuavo_gym(
        Path(robot_cfg["deploy_config"]),
        max_episode_steps=int(cfg.raw["safety"].get("max_episode_steps", 200)),
    )
    backend = ROSBackend(kuavo_gym, publish_unit=str(robot_cfg.get("publish_unit", "rad_to_sdk")))
    raw_env = getattr(kuavo_gym, "unwrapped", kuavo_gym)
    expected_eef = str(robot_cfg.get("expected_eef_type", "qiangnao"))
    if str(getattr(raw_env, "eef_type", "")) != expected_eef:
        hub.close()
        backend.close()
        raise RuntimeError(f"deploy env eef_type != expected {expected_eef!r}")
    if bool(getattr(raw_env, "frame_alignment", False)):
        hub.close()
        backend.close()
        raise RuntimeError("RL-100 real deployment requires frame_alignment=false for timestamp-aware state")
    return hub, backend


def _ros_preflight(cfg, *, timeout_s: float) -> tuple[dict[str, Any], Any, Any]:
    hub, backend = _make_ros_components(cfg)
    limits = cfg.runner_limits()
    report = hub.preflight(timeout_s=timeout_s)
    if not report.get("ok", False):
        return {"ok": False, "pointcloud": report}, hub, backend
    try:
        cloud = hub.get_point_cloud_sample(
            max_depth_age_s=limits.depth_max_age_s,
            max_camera_skew_s=limits.max_camera_skew_s,
        )
        robot = backend.get_observation()
        required = ("state_stamp_s", "hand_stamp_s", "state_age_s", "hand_age_s")
        missing = [key for key in required if key not in robot.extras]
        expected_dims = [int(v) for v in cfg.raw["robot"].get("expected_raw_joint_dims", [20])]
        report = {
            "ok": not missing and robot.raw_joint_dim in expected_dims,
            "pointcloud": report,
            "sample": {
                "shape": list(cloud.points.shape),
                "oldest_age_s": cloud.oldest_age_s,
                "max_camera_skew_s": cloud.max_camera_skew_s,
                "camera_stamps": cloud.camera_stamps,
            },
            "robot": {
                "state_shape": list(robot.state.shape),
                "raw_joint_dim": robot.raw_joint_dim,
                "expected_raw_joint_dims": expected_dims,
                "timing": robot.extras,
                "missing_timing": missing,
            },
        }
    except Exception as exc:  # noqa: BLE001
        report = {"ok": False, "pointcloud": report, "error": str(exc)}
    return report, hub, backend


def cmd_ros_preflight(args: argparse.Namespace) -> int:
    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config

    cfg = load_rl100_deploy_config(args.config)
    hub = backend = None
    try:
        report, hub, backend = _ros_preflight(cfg, timeout_s=float(args.duration_s))
        _print(report)
        return 0 if report.get("ok") else 2
    finally:
        if hub is not None:
            hub.close()
        if backend is not None:
            backend.close()


def _check_live_arm(cfg, backend, args: argparse.Namespace) -> None:
    if not args.confirm_live or not args.physical_estop_ready or not args.live_token:
        raise RuntimeError("live requires --confirm-live --physical-estop-ready and a non-empty --live-token")
    configured_token = str(cfg.raw.get("mode", {}).get("live_confirmation_token", ""))
    if configured_token and args.live_token != configured_token:
        raise RuntimeError("live confirmation token does not match deployment config")
    expected = cfg.expected_pose16(require_approved=True)
    cfg.safety_config(require_approved=True)
    current = backend.get_observation().state
    arm_error = float(max(abs(current[i] - expected[i]) for i in [*range(7), *range(8, 15)]))
    grip_error = float(max(abs(current[7] - expected[7]), abs(current[15] - expected[15])))
    startup = cfg.raw["startup"]
    if arm_error > float(startup["max_arm_error_rad"]) or grip_error > float(startup["max_gripper_error"]):
        raise RuntimeError(f"startup pose mismatch: arm={arm_error:.4f}, gripper={grip_error:.4f}")


def _run_realtime(args: argparse.Namespace, *, live: bool) -> int:
    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config
    from kuavo_rl.rl100_real_runner import RL100RealRunner
    from kuavo_rl.safety import SafetyGate

    cfg = load_rl100_deploy_config(args.config)
    hub = backend = None
    try:
        preflight, hub, backend = _ros_preflight(cfg, timeout_s=float(args.preflight_timeout_s))
        if not preflight.get("ok"):
            _print(preflight)
            return 2
        if live:
            _check_live_arm(cfg, backend, args)
        policy = _policy_from_config(cfg)
        log_dir = Path(cfg.raw.get("logging", {}).get("output_dir", "logs/rl100_real"))
        run_stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{'live' if live else 'shadow'}"
        audit = JsonlAudit(log_dir / f"{run_stem}.jsonl")
        _write_manifest(log_dir / f"{run_stem}.manifest.json", cfg, policy, live=live)
        limits = cfg.runner_limits()
        runner = RL100RealRunner(
            policy=policy,
            backend=backend,
            point_cloud_source=lambda: hub.get_point_cloud_sample(
                max_depth_age_s=limits.depth_max_age_s,
                max_camera_skew_s=limits.max_camera_skew_s,
            ),
            safety=SafetyGate(cfg.safety_config(require_approved=live)),
            limits=limits,
            shadow_mode=not live,
            audit_sink=audit,
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
        if hub is not None:
            hub.close()
        if backend is not None:
            backend.close()


def cmd_offline_replay(args: argparse.Namespace) -> int:
    import numpy as np
    import zarr

    from kuavo_rl.rl100_deploy_config import load_rl100_deploy_config
    from kuavo_rl.rl100_real_runner import ObservationHistory

    cfg = load_rl100_deploy_config(args.config)
    if "grasp_8_4.zarr" in str(args.zarr_path) and "grasp_8_4_v2" not in str(args.zarr_path):
        raise RuntimeError("refusing known-invalid old grasp_8_4.zarr")
    policy = _policy_from_config(cfg)
    root = zarr.open(str(args.zarr_path), mode="r")
    states = np.asarray(root["data"]["state"][:], dtype=np.float32)
    actions = np.asarray(root["data"]["action"][:], dtype=np.float32)
    points = np.asarray(root["data"]["point_cloud"][:], dtype=np.float32)
    history = ObservationHistory(policy.info.n_obs_steps)
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
        "arm_mae": float(err[:, [*range(7), *range(8, 15)]].mean()),
        "gripper_mae": float(err[:, [7, 15]].mean()),
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
