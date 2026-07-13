#!/usr/bin/env python3
"""Dataset / contract preflight for Kuavo HIL-SERL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kuavo_rl.contracts import ACTION_DIM, IMAGE_KEYS, STATE_DIM
from kuavo_rl.ros_adapter import default_v62_joint_map


def audit_info_json(info_path: Path) -> dict:
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    report = {
        "path": str(info_path),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "checks": [],
        "ok": True,
    }

    def check(name: str, cond: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": bool(cond), "detail": detail})
        if not cond:
            report["ok"] = False

    state = features.get("observation.state", {})
    action = features.get("action", {})
    check("state_dim", state.get("shape") == [STATE_DIM], str(state.get("shape")))
    check("action_dim", action.get("shape") == [ACTION_DIM], str(action.get("shape")))
    check("fps_10", info.get("fps") == 10, str(info.get("fps")))
    for key in IMAGE_KEYS:
        feat = features.get(key, {})
        check(f"image_{key}", feat.get("shape") == [3, 480, 848], str(feat.get("shape")))
    report["joint_map_default"] = default_v62_joint_map()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/lerobot/lerobot_merged"),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    info_path = args.dataset / "meta" / "info.json"
    if not info_path.exists():
        raise SystemExit(f"missing {info_path}")
    report = audit_info_json(info_path)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["ok"] else 2)


if __name__ == "__main__":
    main()
