#!/usr/bin/env python3
"""Prefetch Robometer + Qwen3-VL backbone from ModelScope (HF unreachable)."""

from __future__ import annotations

import argparse
from pathlib import Path


MODELS = (
    ("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct"),
    ("lerobot/Robometer-4B", "Robometer-4B"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/models"),
        help="Local directory for ModelScope snapshots",
    )
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    from modelscope.hub.snapshot_download import snapshot_download

    for mid, sub in MODELS:
        out = args.out_root / sub
        print(f"DOWNLOAD_START {mid} -> {out}", flush=True)
        path = snapshot_download(mid, local_dir=str(out))
        print(f"DOWNLOAD_OK {mid} {path}", flush=True)
    print("ALL_DONE", flush=True)
    print(
        "Set (optional):\n"
        f"  export KUAVO_ROBOMETER_BASE_MODEL={args.out_root / 'Qwen3-VL-4B-Instruct'}\n"
        f"  export KUAVO_ROBOMETER_MODEL={args.out_root / 'Robometer-4B'}"
    )


if __name__ == "__main__":
    main()
