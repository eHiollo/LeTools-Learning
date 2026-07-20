#!/usr/bin/env python3
"""Local HIL dataset collection CLI (C0–C3; C1 Quest gate; C2 dry-run collect).

True robot motion requires --confirm-live and later phases (C2 live / C6).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print(obj: object) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _load_orch(args: argparse.Namespace):
    from kuavo_rl.hil_collection import (
        CollectionConfig,
        HILCollectionOrchestrator,
        load_collection_config,
    )

    raw = load_collection_config(args.config)
    cfg = CollectionConfig.from_dict(raw, config_path=Path(args.config))
    if getattr(args, "task_id", None):
        cfg.task_id = args.task_id
    if getattr(args, "mode", None):
        cfg.mode = args.mode
    if getattr(args, "root", None):
        cfg.root = Path(args.root)
    if getattr(args, "allow_ros_gate", False):
        cfg.skip_gate_ros = False
    return HILCollectionOrchestrator(cfg)


def cmd_preflight(args: argparse.Namespace) -> int:
    orch = _load_orch(args)
    try:
        out = orch.preflight(
            task_id=args.task_id,
            for_live_collect=bool(args.for_live_collect),
        )
        _print(out)
        return 0 if out["status"] != "Block" else 2
    finally:
        orch.close()


def cmd_recover(args: argparse.Namespace) -> int:
    orch = _load_orch(args)
    try:
        report = orch.recover()
        _print(report.to_dict())
        return 0
    finally:
        orch.close()


def cmd_inspect(args: argparse.Namespace) -> int:
    orch = _load_orch(args)
    try:
        if args.report:
            _print(orch.collection_report())
            return 0
        out = orch.inspect(
            episode_id=args.episode_id,
            pending_review=bool(args.pending_review),
        )
        _print(out)
        return 0
    finally:
        orch.close()


def cmd_label(args: argparse.Namespace) -> int:
    orch = _load_orch(args)
    try:
        label = orch.label_episode(
            args.episode_id,
            final_label=args.label,
            reason=args.reason,
            labeler=args.labeler,
        )
        _print(label.to_dict())
        return 0
    finally:
        orch.close()


def cmd_export_lerobot(args: argparse.Namespace) -> int:
    """Batch rosbag → one LeRobot v3 via KuavoBrain CvtRosbag2Lerobot."""
    from kuavo_rl.brain_lerobot_export import (
        assert_brain_converter,
        export_batch_to_lerobot,
        stage_bag_for_batch,
    )

    check = assert_brain_converter()
    if args.check_only:
        _print(check)
        return 0 if check["ok"] else 2
    if not check["ok"]:
        _print(check)
        return 2

    orch = _load_orch(args)
    try:
        root = Path(args.root or orch.config.root)
        profile = args.topic_profile or getattr(
            orch.config, "lerobot_topic_profile", "sim"
        )
        task = (
            args.task_description
            or orch.config.task_text
            or orch.config.task_id
        )
        # Optional: stage one extra bag (by eid or --bag) into the batch dir first.
        if args.bag:
            eid = args.episode_id or Path(args.bag).stem
            dest, why = stage_bag_for_batch(
                root=root, episode_id=eid, bag_path=Path(args.bag)
            )
            if dest is None:
                _print({"status": "Rejected", "reasons": why, "episode_id": eid})
                return 2
        elif args.episode_id:
            dest, why = stage_bag_for_batch(root=root, episode_id=args.episode_id)
            if dest is None:
                _print(
                    {
                        "status": "Rejected",
                        "reasons": why,
                        "episode_id": args.episode_id,
                    }
                )
                return 2

        report = export_batch_to_lerobot(
            root=root,
            episode_ids=[args.episode_id] if args.only_listed and args.episode_id else None,
            resync_from_accepted=not bool(args.staged_only),
            lerobot_dir_name=args.lerobot_dir_name,
            topic_profile=profile,
            task_description=task,
        )
        _print(report.to_dict())
        return 0 if report.status == "ok" else 2
    finally:
        orch.close()


def cmd_review(args: argparse.Namespace) -> int:
    orch = _load_orch(args)
    try:
        label = orch.review_episode(
            args.episode_id,
            approve=bool(args.approve),
            reviewer=args.reviewer,
            reason=args.reason,
            require_distinct_reviewer=bool(args.require_distinct_reviewer),
        )
        if args.approve and args.publish_accepted:
            pub = orch.publish_train_ready(args.episode_id)
            _print({"label": label.to_dict(), "publish": pub})
        else:
            _print(label.to_dict())
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        orch.close()


def cmd_calibrate_stick(args: argparse.Namespace) -> int:
    """Write machine-local stick axis override (C1). No robot action publish."""
    from kuavo_rl.hil_collection import load_collection_config, CollectionConfig
    from kuavo_rl.quest_episode_control import (
        StickAxisCalibration,
        infer_calibration_from_samples,
        save_stick_calibration,
    )

    raw = load_collection_config(args.config)
    cfg = CollectionConfig.from_dict(raw, config_path=Path(args.config))
    path = Path(args.output) if args.output else cfg.stick_calibration_path

    if args.manual:
        cal = StickAxisCalibration(
            invert_right_x=bool(args.invert_x),
            invert_right_y=bool(args.invert_y),
            swap_xy=bool(args.swap_xy),
            calibrated=True,
            calibrated_by=args.operator,
            notes="manual_cli",
        )
        from kuavo_rl.hil_recording.timebase import now_stamps

        cal.calibrated_at_wall_ns = now_stamps().wall_time_ns
    else:
        # tip samples: "x,y"
        def _parse(s: str) -> tuple[float, float]:
            a, b = s.split(",")
            return float(a), float(b)

        try:
            cal = infer_calibration_from_samples(
                tip_right=_parse(args.tip_right),
                tip_down=_parse(args.tip_down),
                calibrated_by=args.operator,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    out = save_stick_calibration(cal, path)
    _print({"path": str(out), "calibration": cal.to_dict()})
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    """Collect one episode: dry-run | vr_only sim | (ACT live later)."""
    if args.dry_run:
        orch = _load_orch(args)
        try:
            out = orch.collect_episode_dry_run(
                episode_id=args.episode_id,
                operator=args.operator or "anonymous",
                scene_id=args.scene_id,
                max_steps=int(args.max_steps or orch.config.default_max_steps),
                end_event=args.end_event,
            )
            _print(out)
            if out.get("status") == "Block":
                return 2
            return 0
        finally:
            orch.close()

    want_live = bool(args.confirm_live or args.vr_sim)
    if not want_live and not args.shadow:
        print(
            "ERROR: collect refuses live control without --confirm-live "
            "(use --dry-run, or --vr-sim for Kuavo-Sim VR teaching).",
            file=sys.stderr,
        )
        return 2

    orch = _load_orch(args)
    mode = args.mode or orch.config.mode
    vr_sim = bool(args.vr_sim or mode == "vr_only")
    if want_live and vr_sim:
        try:
            from kuavo_rl.hil_collect_live import collect_vr_session

            if args.deploy_config:
                orch.config.deploy_config = Path(args.deploy_config)
            if args.env_config:
                orch.config.env_config = Path(args.env_config)
            orch.config.mode = "vr_only"
            # LeRobot-style: stay up across episodes unless --single-episode
            summary = collect_vr_session(
                orch,
                deploy_config=orch.config.deploy_config,
                env_config=orch.config.env_config,
                operator=args.operator or "anonymous",
                scene_id=args.scene_id,
                max_steps=args.max_steps,
                num_episodes=int(args.episodes or 50),
                reset_time_s=args.reset_time_s,
                live_rosbag=True if args.live_rosbag else None,
                enable_quest_episode_control=not bool(args.no_quest_episode_control),
                single_episode=bool(args.single_episode),
            )
            # Short exit summary (per-episode lines already printed during session)
            last = (summary.get("results") or [{}])[-1] or {}
            batch_lr = summary.get("lerobot_export") or {}
            print(
                json.dumps(
                    {
                        "status": summary.get("status"),
                        "completed_episodes": summary.get("completed_episodes"),
                        "attempts": summary.get("attempts"),
                        "root": summary.get("root"),
                        "last_episode_id": last.get("episode_id"),
                        "last_status": last.get("status"),
                        "last_steps": last.get("steps"),
                        "last_accepted_replay": last.get("accepted_replay"),
                        "last_train_ready": last.get("train_ready"),
                        "lerobot_dir": summary.get("lerobot_dir"),
                        "lerobot_bag_count": batch_lr.get("bag_count"),
                        "lerobot_episode_ids": batch_lr.get("episode_ids"),
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
            if summary.get("status") == "Block":
                return 2
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        finally:
            orch.close()

    orch.close()
    if want_live:
        print(
            "ERROR: live ACT collect not wired yet. "
            "For VR teaching in sim: --mode vr_only --confirm-live --vr-sim "
            "(or collect --vr-sim).",
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {"status": "not_implemented", "message": "use --vr-sim or --dry-run"},
            indent=2,
        )
    )
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    if args.dry_run:
        orch = _load_orch(args)
        try:
            out = orch.batch_dry_run(
                episodes=int(args.episodes),
                operator=args.operator or "anonymous",
                scene_id=args.scene_id,
                max_steps=int(args.max_steps or 4),
                collection_id=args.collection_id,
            )
            _print(out)
            return 0 if out.get("status") == "ok" else 2
        finally:
            orch.close()
    if not args.confirm_live:
        print(
            "ERROR: batch refuses live control without --confirm-live "
            "(use --dry-run for C4 scaffolding).",
            file=sys.stderr,
        )
        return 2
    print(
        "ERROR: live batch not wired yet (C4/C6). Use batch --dry-run.",
        file=sys.stderr,
    )
    return 3


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="collect_hil_dataset.py",
        description="Local HIL dataset collection (no KuavoBrain platform).",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/hil_collection_sim_v001.yaml"),
    )
    p.add_argument("--root", type=Path, default=None)
    p.add_argument("--task-id", default=None)
    p.add_argument("--mode", default=None, choices=["act", "act_vr", "vr_only", "shadow"])

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight", help="Recover + gate checks; never publishes control")
    sp.add_argument("--allow-ros-gate", action="store_true")
    sp.add_argument(
        "--for-live-collect",
        action="store_true",
        help="Enforce stick calibration + collection_mode_ack (C1 live gate)",
    )
    sp.set_defaults(func=cmd_preflight)

    sr = sub.add_parser("recover", help="Recover interrupted sessions")
    sr.set_defaults(func=cmd_recover)

    si = sub.add_parser("inspect", help="Inspect sessions / pending_review / index / report")
    si.add_argument("episode_id", nargs="?", default=None)
    si.add_argument("--pending-review", action="store_true")
    si.add_argument("--report", action="store_true", help="C3 label/export distribution report")
    si.set_defaults(func=cmd_inspect)

    sl = sub.add_parser("label", help="Offline label an episode (no robot)")
    sl.add_argument("episode_id")
    sl.add_argument("--label", required=True, choices=["success", "failure", "abort", "unsafe", "invalid"])
    sl.add_argument("--reason", default=None)
    sl.add_argument("--labeler", default="anonymous")
    sl.set_defaults(func=cmd_label)

    sv = sub.add_parser("review", help="Offline review / approve into TRAIN_READY path")
    sv.add_argument("episode_id")
    g = sv.add_mutually_exclusive_group(required=True)
    g.add_argument("--approve", action="store_true")
    g.add_argument("--reject", action="store_true")
    sv.add_argument("--reviewer", default="anonymous")
    sv.add_argument("--reason", default=None)
    sv.add_argument("--require-distinct-reviewer", action="store_true")
    sv.add_argument(
        "--publish-accepted",
        action="store_true",
        help="After approve, run publish_accepted → accepted_replay/TRAIN_READY",
    )
    sv.set_defaults(func=cmd_review)

    se = sub.add_parser(
        "export-lerobot",
        help="Batch bags → one LeRobot v3 via KuavoBrain CvtRosbag2Lerobot",
    )
    se.add_argument(
        "episode_id",
        nargs="?",
        default=None,
        help="Optional: stage this eid into batch_bags/ before convert",
    )
    se.add_argument(
        "--check-only",
        action="store_true",
        help="Verify Brain converter + challenge lerobot v3.0 only",
    )
    se.add_argument("--bag", type=Path, default=None, help="Stage this bag into batch")
    se.add_argument("--lerobot-dir-name", default="lerobot_v3")
    se.add_argument(
        "--topic-profile",
        choices=["sim", "brain"],
        default=None,
        help="sim: alias /camera|/wrist_cam_* → Brain reader keys; brain: /cam_* compressed",
    )
    se.add_argument("--task-description", default=None)
    se.add_argument(
        "--staged-only",
        action="store_true",
        help="Convert existing batch_bags/ only (do not resync from accepted_replay)",
    )
    se.add_argument(
        "--only-listed",
        action="store_true",
        help="With episode_id: convert only that eid (not full TRAIN_READY set)",
    )
    se.set_defaults(func=cmd_export_lerobot)

    sca = sub.add_parser("calibrate-stick", help="Write local stick axis override (C1)")
    sca.add_argument("--operator", default="anonymous")
    sca.add_argument("--output", type=Path, default=None)
    sca.add_argument("--manual", action="store_true")
    sca.add_argument("--invert-x", action="store_true")
    sca.add_argument("--invert-y", action="store_true")
    sca.add_argument("--swap-xy", action="store_true")
    sca.add_argument(
        "--tip-right",
        default="0.9,0.0",
        help="Raw sample when operator tips stick RIGHT (x,y)",
    )
    sca.add_argument(
        "--tip-down",
        default="0.0,-0.9",
        help="Raw sample when operator tips stick DOWN (x,y)",
    )
    sca.set_defaults(func=cmd_calibrate_stick)

    sc = sub.add_parser("collect", help="Collect one episode")
    sc.add_argument("--confirm-live", action="store_true")
    sc.add_argument("--shadow", action="store_true")
    sc.add_argument(
        "--vr-sim",
        action="store_true",
        help="Kuavo-Sim VR teaching session (LeRobot-style multi-episode loop)",
    )
    sc.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthetic transitions → pending_review (no ACT/env)",
    )
    sc.add_argument("--live-rosbag", action="store_true", help="Real rosbag (sim/real ROS)")
    sc.add_argument("--no-quest-episode-control", action="store_true")
    sc.add_argument(
        "--single-episode",
        action="store_true",
        help="Exit after one episode (default: stay up like lerobot-record)",
    )
    sc.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Max episodes for VR session (default 50)",
    )
    sc.add_argument("--reset-time-s", type=float, default=None, help="RESET phase timeout")
    sc.add_argument(
        "--deploy-config",
        type=Path,
        default=None,
        help="Kuavo deploy yaml (default: sim native cams)",
    )
    sc.add_argument(
        "--env-config",
        type=Path,
        default=None,
        help="HIL env yaml (safety/episode); shadow forced for vr_only",
    )
    sc.add_argument("--scene-id", default=None)
    sc.add_argument("--operator", default=None)
    sc.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="VR collect: ignored for episode cutoff (B ends episodes). "
        "Only raises the safety ceiling if larger than default.",
    )
    sc.add_argument("--episode-id", default=None)
    sc.add_argument(
        "--end-event",
        default="right_stick_right",
        choices=[
            "right_stick_right",
            "right_stick_left",
            "right_stick_down",
            "ctrl_c_abort",
        ],
    )
    sc.set_defaults(func=cmd_collect)

    sb = sub.add_parser("batch", help="Batch collect (C4)")
    sb.add_argument("--confirm-live", action="store_true")
    sb.add_argument("--dry-run", action="store_true", help="C4 dry-run batch loop")
    sb.add_argument("--episodes", type=int, default=20)
    sb.add_argument("--max-steps", type=int, default=4)
    sb.add_argument("--operator", default=None)
    sb.add_argument("--scene-id", default=None)
    sb.add_argument("--collection-id", default=None)
    sb.set_defaults(func=cmd_batch)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "reject", False):
        args.approve = False
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
