#!/usr/bin/env python3
"""RL-100 zarr collection CLI (parallel feature; does not change HIL defaults).

Examples::

  # Offline smoke: write a tiny zarr from synthetic episodes
  python scripts/rl/collect_rl100_zarr.py smoke --task box_to_chest_v1

  # Merge staged episode NPZs into data/rl100/<task>/demo.zarr
  python scripts/rl/collect_rl100_zarr.py build --config configs/rl/rl100_zarr_collect.yaml

  # Live VR collect on real robot (requires ROS + depth/tf + --confirm-live)
  python scripts/rl/collect_rl100_zarr.py collect --config configs/rl/rl100_zarr_collect.yaml --confirm-live
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _config_sha256(path: str | None) -> str:
    if not path:
        return "unknown"
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return "unknown"
    return hashlib.sha256(cfg_path.read_bytes()).hexdigest()


def _zarr_attrs(cfg, *, smoke: bool = False) -> dict[str, object]:
    return {
        "task": cfg.task,
        "contract": cfg.contract,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "smooth_penalty": cfg.smooth_penalty,
        "collection_config_sha256": _config_sha256(getattr(cfg, "config_path", None)),
        "smoke": smoke,
    }


def _load_cfg(args: argparse.Namespace):
    from kuavo_rl.rl100_zarr.config import load_rl100_collect_config

    cfg = load_rl100_collect_config(getattr(args, "config", None))
    if getattr(args, "task", None):
        cfg.task = args.task
    if getattr(args, "output_root", None):
        cfg.output_root = args.output_root
    if getattr(args, "zarr_name", None):
        cfg.zarr_name = args.zarr_name
    if getattr(args, "overwrite", False):
        cfg.overwrite = True
    if getattr(args, "only_success", False):
        cfg.only_success = True
    if getattr(args, "confirm_live", False):
        cfg.confirm_live = True
    cfg.validate_contract()
    cfg.config_path = getattr(args, "config", None)
    return cfg


def cmd_preflight(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    out = {
        "task": cfg.task,
        "zarr_path": str(cfg.zarr_path()),
        "episode_dir": str(cfg.output_dir() / "episodes"),
        "num_points": cfg.num_points,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "only_success": cfg.only_success,
        "deploy_config": cfg.deploy_config,
        "env_config": cfg.env_config,
        "require_all_cameras": cfg.require_all_cameras,
        "fail_on_empty_pointcloud": cfg.fail_on_empty_pointcloud,
        "command_topics": {
            "arm": cfg.arm_traj_topic,
            "hand": cfg.hand_command_topic,
        },
        "quality_gates": {
            "start_on_any_command": cfg.start_on_any_command,
            "command_hold_last": cfg.command_hold_last,
            "require_hand_motion": cfg.require_hand_motion,
            "min_hand_action_range": cfg.min_hand_action_range,
            "max_consecutive_source_failures": cfg.max_consecutive_source_failures,
        },
        "cameras": [
            {
                "name": c.name,
                "enabled": c.enabled,
                "depth_topic": c.depth_topic,
                "camera_info_topic": c.camera_info_topic,
                "frame_id": c.frame_id,
                "depth_msg_type": c.depth_msg_type,
            }
            for c in cfg.cameras
        ],
    }
    if args.check_ros:
        try:
            import rospy
            from kuavo_rl.rl100_zarr.ros_depth import DepthPointCloudHub

            if not rospy.core.is_initialized():
                rospy.init_node("rl100_zarr_preflight", anonymous=True)
            hub = DepthPointCloudHub(cfg)
            hub.start()
            out["ros_pointcloud"] = hub.preflight(timeout_s=float(args.timeout_s))
            hub.close()
        except Exception as exc:  # noqa: BLE001
            out["ros_pointcloud"] = {"ok": False, "error": str(exc)}
            _print(out)
            return 2
    _print(out)
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Write a tiny valid zarr without ROS (CI / local sanity)."""
    import numpy as np

    from kuavo_rl.rl100_zarr.pointcloud import downsample_fps
    from kuavo_rl.rl100_zarr.schema import ACTION_DIM, NUM_POINTS, STATE_DIM
    from kuavo_rl.rl100_zarr.staging import save_episode_npz, build_zarr_from_episode_dir

    cfg = _load_cfg(args)
    ep_dir = cfg.output_dir() / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i, result in enumerate(["success", "failure", "success"]):
        t = 8 + i
        actions = []
        for _ in range(t):
            action = np.zeros((ACTION_DIM,), dtype=np.float32)
            action[:14] = rng.normal(size=(14,)).astype(np.float32)
            action[14:] = rng.uniform(0.0, 100.0, size=(12,)).astype(np.float32)
            actions.append(action)
        states = []
        for _ in range(t):
            state = rng.normal(size=(STATE_DIM,)).astype(np.float32)
            state[20:] = rng.uniform(0.0, 100.0, size=(12,)).astype(np.float32)
            states.append(state)
        pcs = [
            downsample_fps(rng.normal(size=(5000, 3)).astype(np.float32), NUM_POINTS)
            for _ in range(t)
        ]
        save_episode_npz(
            ep_dir / f"smoke_{i}_{result}.npz",
            states=states,
            actions=actions,
            point_clouds=pcs,
            result_type=result,
            meta={"smoke": True},
        )

    report = build_zarr_from_episode_dir(
        ep_dir,
        cfg.zarr_path(),
        only_success=cfg.only_success,
        overwrite=True,
        lambda_penalty=cfg.lambda_penalty,
        smooth_penalty=cfg.smooth_penalty,
        max_episode_len=cfg.max_episode_len,
        attrs=_zarr_attrs(cfg, smoke=True),
    )
    _print(report)
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from kuavo_rl.rl100_zarr.staging import build_zarr_from_episode_dir

    cfg = _load_cfg(args)
    ep_dir = Path(args.episode_dir) if args.episode_dir else (cfg.output_dir() / "episodes")
    report = build_zarr_from_episode_dir(
        ep_dir,
        cfg.zarr_path(),
        only_success=cfg.only_success,
        overwrite=cfg.overwrite or bool(args.overwrite),
        lambda_penalty=cfg.lambda_penalty,
        smooth_penalty=cfg.smooth_penalty,
        max_episode_len=cfg.max_episode_len,
        attrs=_zarr_attrs(cfg),
    )
    _print(report)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    import numpy as np
    import zarr

    from kuavo_rl.rl100_zarr.live_collect import (
        episode_action_quality_errors,
        episode_action_quality_report,
    )

    cfg = _load_cfg(args)
    path = Path(args.zarr_path) if args.zarr_path else cfg.zarr_path()
    root = zarr.open(str(path), mode="r")
    data = root["data"]
    meta = root["meta"]
    states = np.asarray(data["state"][:], dtype=np.float32)
    actions = np.asarray(data["action"][:], dtype=np.float32)
    audit = {
        name: np.asarray(meta[name][:])
        for name in meta.array_keys()
        if name != "episode_ends"
    }
    quality_errors = episode_action_quality_errors(
        [row for row in states],
        [row for row in actions],
        require_gripper_motion=cfg.require_hand_motion,
        min_gripper_action_range=cfg.min_hand_action_range,
        audit=audit or None,
    )
    quality_report = episode_action_quality_report(
        [row for row in states],
        [row for row in actions],
        audit=audit or None,
    )
    out = {
        "path": str(path),
        "attrs": dict(root.attrs),
        "shapes": {k: list(data[k].shape) for k in data.array_keys()},
        "episode_ends": np.asarray(meta["episode_ends"][:]).tolist(),
        "n_transitions": int(data["action"].shape[0]),
        "reward_sum": float(np.asarray(data["reward"][:]).sum()),
        "action_quality": {
            "ok": not quality_errors,
            "errors": quality_errors,
            "report": quality_report,
        },
    }
    _print(out)
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    cfg = _load_cfg(args)
    from kuavo_rl.rl100_zarr.live_collect import run_live_collect_session

    report = run_live_collect_session(cfg)
    _print(report)
    if args.build_after and report.get("episode_dir"):
        from kuavo_rl.rl100_zarr.staging import build_zarr_from_episode_dir

        built = build_zarr_from_episode_dir(
            report["episode_dir"],
            cfg.zarr_path(),
            only_success=cfg.only_success,
            overwrite=True,
            lambda_penalty=cfg.lambda_penalty,
            smooth_penalty=cfg.smooth_penalty,
            max_episode_len=cfg.max_episode_len,
            attrs=_zarr_attrs(cfg),
        )
        _print({"build": built})
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kuavo → RL-100 zarr collector")
    p.add_argument("--config", default="configs/rl/rl100_zarr_collect.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight", help="Print resolved config / optional ROS depth check")
    sp.add_argument("--task")
    sp.add_argument("--check-ros", action="store_true")
    sp.add_argument("--timeout-s", type=float, default=5.0)
    sp.set_defaults(func=cmd_preflight)

    ss = sub.add_parser("smoke", help="Synthetic episodes → zarr (no ROS)")
    ss.add_argument("--task")
    ss.add_argument("--output-root")
    ss.add_argument("--zarr-name")
    ss.add_argument("--only-success", action="store_true")
    ss.set_defaults(func=cmd_smoke)

    sb = sub.add_parser("build", help="Merge episode NPZs into RL-100 zarr")
    sb.add_argument("--task")
    sb.add_argument("--episode-dir")
    sb.add_argument("--output-root")
    sb.add_argument("--zarr-name")
    sb.add_argument("--overwrite", action="store_true")
    sb.add_argument("--only-success", action="store_true")
    sb.set_defaults(func=cmd_build)

    si = sub.add_parser("inspect", help="Inspect a zarr store")
    si.add_argument("--task")
    si.add_argument("--zarr-path")
    si.set_defaults(func=cmd_inspect)

    sc = sub.add_parser("collect", help="Live VR collect → episode NPZ (+ optional build)")
    sc.add_argument("--task")
    sc.add_argument("--confirm-live", action="store_true")
    sc.add_argument("--only-success", action="store_true")
    sc.add_argument("--build-after", action="store_true")
    sc.add_argument("--overwrite", action="store_true")
    sc.set_defaults(func=cmd_collect)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
