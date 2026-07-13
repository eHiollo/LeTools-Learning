#!/usr/bin/env python3
"""Offline Robometer scoring scaffold (does not block control loop)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/lerobot/lerobot_merged"))
    parser.add_argument("--out", type=Path, default=Path("data/reward_calibration/offline_scores.json"))
    parser.add_argument("--stub", action="store_true", help="Write placeholder report without loading 4B model")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.stub or True:
        # Default to stub until GPU budget verified; keep interface stable.
        report = {
            "dataset": str(args.dataset),
            "model": "lerobot/Robometer-4B",
            "mode": "stub_placeholder",
            "status": "NOT_SCORED",
            "gates": {
                "spearman_progress_min": 0.7,
                "success_auc_min": 0.85,
                "passed": False,
            },
            "next_steps": [
                "Install lerobot robometer extras and download weights on 5060 Ti.",
                "Score demos + intentional failures.",
                "Fill calibration metrics then flip robometer_mode to episode_end online.",
            ],
        }
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
        return


if __name__ == "__main__":
    main()
