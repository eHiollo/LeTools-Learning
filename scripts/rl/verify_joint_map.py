#!/usr/bin/env python3
"""
Static joint-map preflight for Kuavo v62.

Default mode is OFFLINE (prints expected map, no robot motion).
Use --live with ROS up; it only reads /sensors_data_raw (+ optional claw) and
never publishes policy actions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from kuavo_rl.contracts import ARM_SLICE_BY_RAW_DIM, RAW_STATE_DIM_V62
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


def _read_one_sensors(timeout_s: float = 5.0) -> dict:
    import rospy
    from kuavo_msgs.msg import sensorsData

    rospy.init_node("kuavo_verify_joint_map", anonymous=True, disable_signals=True)
    msg = rospy.wait_for_message("/sensors_data_raw", sensorsData, timeout=timeout_s)
    joint_q = np.asarray(msg.joint_data.joint_q, dtype=np.float32)
    out: dict = {
        "raw_joint_dim": int(joint_q.shape[0]),
        "joint_q": joint_q.tolist(),
    }
    try:
        from kuavo_msgs.msg import lejuClawState

        claw = rospy.wait_for_message("/leju_claw_state", lejuClawState, timeout=2.0)
        # Best-effort fields; message layout may vary by firmware.
        data = getattr(claw, "data", None)
        if data is not None and hasattr(data, "position"):
            out["claw_position"] = list(data.position)
        elif hasattr(claw, "position"):
            out["claw_position"] = list(claw.position)
        else:
            out["claw_raw_fields"] = [a for a in dir(claw) if not a.startswith("_")]
    except Exception as exc:  # noqa: BLE001
        out["claw_error"] = str(exc)
    return out


def live_report(map_path: Path | None, timeout_s: float = 5.0) -> dict:
    """Live read of sensors; never publishes actions."""
    try:
        import rospy  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        return {"mode": "live", "ok": False, "error": f"rospy unavailable: {exc}"}

    expected = default_v62_joint_map()
    loaded = {}
    if map_path and map_path.exists():
        loaded = yaml.safe_load(map_path.read_text(encoding="utf-8")) or {}

    try:
        raw = _read_one_sensors(timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001
        return {"mode": "live", "ok": False, "error": str(exc), "expected": expected}

    dim = int(raw["raw_joint_dim"])
    joint_q = np.asarray(raw["joint_q"], dtype=np.float32)
    arm14 = slice_arm_state(joint_q)
    expected_slice = ARM_SLICE_BY_RAW_DIM.get(dim)
    # 28 = classic biped v62; 20 = wheeled / platform 5w layout (still [L7,R7] via table).
    ok = (
        dim in ARM_SLICE_BY_RAW_DIM
        and arm14.shape == (14,)
        and bool(np.isfinite(arm14).all())
        and float(np.max(np.abs(arm14))) < 3.2
    )

    report = {
        "mode": "live",
        "ok": bool(ok),
        "expected": expected,
        "loaded_config": loaded,
        "raw_joint_dim": dim,
        "arm_slice_used": [expected_slice.start, expected_slice.stop] if expected_slice else None,
        "arm14_rad": arm14.tolist(),
        "arm14_abs_max": float(np.max(np.abs(arm14))),
        "arm14_mean": float(np.mean(arm14)),
        "joint_q_head8": joint_q[:8].tolist() if dim >= 8 else joint_q.tolist(),
        "claw": {
            k: raw[k]
            for k in ("claw_position", "claw_error", "claw_raw_fields")
            if k in raw
        },
        "checks": {
            "dim_known_in_slice_table": dim in ARM_SLICE_BY_RAW_DIM,
            "preferred_raw_dim_28": dim == RAW_STATE_DIM_V62,
            "arm14_finite": bool(np.isfinite(arm14).all()),
            "arm14_abs_max_lt_pi": bool(float(np.max(np.abs(arm14))) < 3.2),
        },
        "warnings": [],
    }
    if dim != RAW_STATE_DIM_V62:
        sl = (
            [expected_slice.start, expected_slice.stop]
            if expected_slice is not None
            else None
        )
        report["warnings"].append(
            f"raw_joint_dim={dim} != 28; using ARM_SLICE_BY_RAW_DIM[{dim}]={sl}. "
            "Confirm before real publish."
        )
    if not report["checks"]["arm14_abs_max_lt_pi"]:
        report["warnings"].append("arm14 abs max >= pi; units may be deg not rad.")
        report["ok"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("configs/rl/kuavo_v62_joint_map.yaml"),
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    report = (
        live_report(args.map, timeout_s=args.timeout)
        if args.live
        else offline_report(args.map)
    )
    demo = np.arange(28, dtype=np.float32)
    report["demo_slice_12_26"] = slice_arm_state(demo).tolist()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    if args.live and not report.get("ok", False):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
