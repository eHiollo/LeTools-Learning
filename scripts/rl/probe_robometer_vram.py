#!/usr/bin/env python3
"""Probe Robometer VRAM vs empty CUDA baseline (handbook 1.8)."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _cuda_mem_gb() -> dict[str, float]:
    import torch

    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "available": False}
    torch.cuda.synchronize()
    return {
        "allocated_gb": float(torch.cuda.memory_allocated() / (1024**3)),
        "reserved_gb": float(torch.cuda.memory_reserved() / (1024**3)),
        "max_allocated_gb": float(torch.cuda.max_memory_allocated() / (1024**3)),
        "available": True,
    }


def _recommend(peak_gb: float, gpu_total_gb: float | None) -> str:
    # Heuristic from handbook: Robometer ~8–10GB; leave room for actor+learner.
    if gpu_total_gb is None:
        return "unknown_gpu_capacity; prefer serial episode_end or offline-only until measured with nvidia-smi"
    headroom = gpu_total_gb - peak_gb
    if peak_gb >= 9.0 and headroom < 5.0:
        return "serial_or_offline_only"
    if peak_gb >= 6.0 and headroom < 4.0:
        return "prefer_serial_episode_end"
    return "co_resident_possible_if_actor_learner_fit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/reward_calibration/vram_budget.json"))
    parser.add_argument("--model-id", default="lerobot/Robometer-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--stub", action="store_true")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "model": args.model_id,
        "device": args.device,
        "status": "NOT_RUN",
    }

    if args.stub:
        report.update(
            {
                "status": "STUB",
                "baseline": {"allocated_gb": 0.0},
                "after_load": {"allocated_gb": None},
                "after_forward": {"allocated_gb": None, "latency_s": None},
                "peak_allocated_gb": None,
                "recommendation": "serial_or_offline_only",
                "note": "stub probe; re-run without --stub on hilserl GPU image",
            }
        )
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    try:
        import torch

        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        baseline = _cuda_mem_gb()
        gpu_total = None
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu_total = float(props.total_memory / (1024**3))
        report["baseline"] = baseline
        report["gpu_total_gb"] = gpu_total
        # Always record a numeric peak field (baseline until model loads).
        report["peak_allocated_gb"] = float(baseline.get("max_allocated_gb") or baseline.get("allocated_gb") or 0.0)

        from kuavo_rl.robometer_scorer import RobometerScorer

        scorer = RobometerScorer(model_id=args.model_id, device=args.device, max_frames=args.max_frames)
        scorer.ensure_loaded()
        after_load = _cuda_mem_gb()

        # Dummy CHW frames
        frames = np.zeros((args.max_frames, 3, 224, 224), dtype=np.uint8)
        frames[:] = np.random.RandomState(0).randint(0, 255, size=frames.shape, dtype=np.uint8)
        _, _, latency = scorer.score_chw_frames(frames, "probe task")
        after_fwd = _cuda_mem_gb()
        after_fwd["latency_s"] = latency

        peak = float(after_fwd.get("max_allocated_gb") or after_fwd.get("allocated_gb") or 0.0)
        report.update(
            {
                "status": "OK",
                "after_load": after_load,
                "after_forward": after_fwd,
                "peak_allocated_gb": peak,
                "recommendation": _recommend(peak, gpu_total),
            }
        )
        scorer.unload()
    except Exception as exc:  # noqa: BLE001
        report.update(
            {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "peak_allocated_gb": float(report.get("peak_allocated_gb") or 0.0),
                "recommendation": "serial_or_offline_only",
            }
        )

    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
