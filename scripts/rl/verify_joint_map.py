#!/usr/bin/env python3
"""
Static joint-map preflight for Kuavo v62.

Default mode is OFFLINE (prints expected map, no robot motion).
Use --live only on the robot PC with ROS up; it must NOT publish policy actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from kuavo_rl.contracts import ARM_SLICE_BY_RAW_DIM
from kuavo_rl.ros_adapter import default_v62_joint_map, slice_arm_state


def offline_report(map_path: Path | None) -> dict:
    expected = default_v62_joint_map()
    if map_path and map_path.exists():
        loaded = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    else:
        loaded = {}
    return {
        "mode": "offline",
        "expected": expected,
        "loaded_config": loaded,
        "slice_table": {str(k): [v.start, v.stop] for k, v in ARM_SLICE_BY_RAW_DIM.items()},
        "warnings": [
            "platform_config.yaml 5w=[4:18] is for 20-D only; v62 28-D uses [12:26].",
            "SDK control_arm_joint_positions is assumed to take radians (deploy path).",
            "If publishing /kuavo_arm_traj directly, convert with arm_rad_to_traj_deg.",
        ],
    }


def live_report() -> dict:
    """Best-effort live read; never publishes actions."""
    try:
        import rospy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"mode": "live", "ok": False, "error": f"rospy unavailable: {exc}"}
    return {
        "mode": "live",
        "ok": False,
        "error": (
            "Live joint dump must be wired to the robot's /sensors_data_raw subscriber "
            "on the Jetson ROS environment. Refusing to invent topic reads here."
        ),
        "required_prints": [
            "raw joint_q length",
            "joint names if available",
            "slice [12:26] values",
            "claw state",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("configs/rl/kuavo_v62_joint_map.yaml"),
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = live_report() if args.live else offline_report(args.map)
    # sanity: slicing example
    import numpy as np

    demo = np.arange(28, dtype=np.float32)
    report["demo_slice_12_26"] = slice_arm_state(demo).tolist()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
