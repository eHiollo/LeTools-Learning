#!/usr/bin/env python3
"""Offline Robometer scoring + handbook 3.4 calibration gates.

Does not block the 10 Hz control loop. Default tries the real model; use --stub
for a placeholder report without GPU/HF.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from kuavo_rl.calibration_metrics import evaluate_calibration_gates
from kuavo_rl.contracts import DEFAULT_TASK_TEXT, IMAGE_KEYS


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = path.with_name("summary.md")
    gates = report.get("gates", {})
    lines = [
        "# Robometer offline calibration",
        "",
        f"- status: `{report.get('status')}`",
        f"- mode: `{report.get('mode')}`",
        f"- model: `{report.get('model')}`",
        f"- gates.passed: **{gates.get('passed')}**",
        f"- gates.passed_synthetic_sanity: `{gates.get('passed_synthetic_sanity')}`",
        f"- failure_source: `{gates.get('failure_source')}`",
        f"- spearman_progress: `{gates.get('spearman_progress')}` (min {gates.get('spearman_progress_min')})",
        f"- success_vs_fail_auc: `{gates.get('success_vs_fail_auc')}` (min {gates.get('success_auc_min')})",
        "",
    ]
    if report.get("error"):
        lines.extend(["## Error", "", f"```\n{report['error']}\n```", ""])
    if report.get("recommendation"):
        lines.extend(["## Recommendation", "", report["recommendation"], ""])
    summary.write_text("\n".join(lines), encoding="utf-8")


def _stub_report(dataset: Path, model: str) -> dict[str, Any]:
    return {
        "dataset": str(dataset),
        "model": model,
        "mode": "stub_placeholder",
        "status": "NOT_SCORED",
        "gates": {
            "spearman_progress": None,
            "success_vs_fail_auc": None,
            "spearman_progress_min": 0.7,
            "success_auc_min": 0.85,
            "n_success": 0,
            "n_failure": 0,
            "passed": False,
        },
        "recommendation": (
            "Install robometer extras, download weights on 5060 Ti, then re-run without --stub. "
            "Do not enable online episode_end until gates.passed is true."
        ),
    }


def _load_episode_frames(
    dataset_root: Path,
    *,
    episode_idx: int,
    image_key: str,
    max_frames: int,
) -> tuple[np.ndarray, str]:
    """Load one episode as (T,C,H,W).

    Prefer PyAV (handles AV1 in Docker). OpenCV often fails on AV1 without
    matching hardware decode libs.
    """
    import pyarrow.parquet as pq

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = float(info.get("fps", 10))
    ep_files = sorted((dataset_root / "meta" / "episodes").glob("**/file-*.parquet"))
    if not ep_files:
        raise RuntimeError(f"no episode parquet under {dataset_root}/meta/episodes")
    df = pq.read_table(ep_files[0]).to_pandas()
    if episode_idx < 0 or episode_idx >= len(df):
        raise RuntimeError(f"episode_idx={episode_idx} out of range (n={len(df)})")
    row = df.iloc[episode_idx]

    chunk = int(row[f"videos/{image_key}/chunk_index"])
    file_idx = int(row[f"videos/{image_key}/file_index"])
    t0 = float(row[f"videos/{image_key}/from_timestamp"])
    t1 = float(row[f"videos/{image_key}/to_timestamp"])
    video_path = dataset_root / "videos" / image_key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(f"missing video {video_path}")

    frames = _decode_mp4_pyav(video_path, t0=t0, t1=t1, fps=fps)
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path} [{t0},{t1}]")

    stacked = np.stack(frames, axis=0)
    if stacked.shape[0] > max_frames * 4:
        sel = np.linspace(0, stacked.shape[0] - 1, max_frames * 4).round().astype(np.int64)
        stacked = stacked[sel]

    tasks = row.get("tasks")
    if isinstance(tasks, (list, np.ndarray)) and len(tasks) > 0:
        task = str(tasks[0])
    elif isinstance(tasks, str) and tasks:
        task = tasks
    else:
        task = DEFAULT_TASK_TEXT
    return stacked.astype(np.uint8), task


def _decode_mp4_pyav(video_path: Path, *, t0: float, t1: float, fps: float) -> list[np.ndarray]:
    import av

    frames: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        # Seek near start (keyframe); then filter by pts time
        offset = max(0.0, t0 - 1.0 / max(fps, 1.0))
        try:
            container.seek(int(offset * av.time_base), any_frame=False, backward=True)
        except Exception:  # noqa: BLE001
            pass
        for frame in container.decode(stream):
            ts = float(frame.time) if frame.time is not None else None
            if ts is not None and ts + 1e-3 < t0:
                continue
            if ts is not None and ts > t1 + 1e-3:
                break
            rgb = frame.to_ndarray(format="rgb24")
            frames.append(np.transpose(rgb, (2, 0, 1)))
    return frames


def _synthetic_failure(frames: np.ndarray) -> np.ndarray:
    """Hard-negative proxy when no human-labeled failures exist.

    Reverse+truncate looked too much like a short demo (AUC inverted under
    max_frames=4). Freeze on the first frame, black out the second half, and
    add light noise so Robometer sees a stuck / occluded non-progressing traj.
    """
    t = int(frames.shape[0])
    fail = np.repeat(frames[:1], t, axis=0).copy()
    mid = max(1, t // 2)
    fail[mid:] = 0
    noise = np.random.RandomState(0).randint(0, 24, size=fail.shape, dtype=np.int16)
    return np.clip(fail.astype(np.int16) + noise, 0, 255).astype(frames.dtype)


def _run_real(
    *,
    dataset: Path,
    model_id: str,
    device: str,
    image_key: str,
    max_frames: int,
    max_success_eps: int,
) -> dict[str, Any]:
    from kuavo_rl.robometer_scorer import RobometerScorer

    scorer = RobometerScorer(
        model_id=model_id,
        device=device,
        max_frames=max_frames,
        image_key=image_key,
        default_task=DEFAULT_TASK_TEXT,
    )
    scorer.ensure_loaded()

    # Discover episode count via info.json to avoid full dataset scan on failure
    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    n_eps = int(info.get("total_episodes", 0))
    if n_eps <= 0:
        raise RuntimeError(f"no episodes in {dataset}")

    success_curves: list[list[float]] = []
    success_finals: list[float] = []
    failure_finals: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    latencies: list[float] = []

    for ep_idx in range(min(n_eps, max_success_eps)):
        frames, task = _load_episode_frames(
            dataset, episode_idx=ep_idx, image_key=image_key, max_frames=max_frames
        )
        progress, success, latency = scorer.score_chw_frames(frames, task)
        latencies.append(latency)
        success_curves.append(progress)
        success_finals.append(float(progress[-1]) if progress else 0.0)
        episode_rows.append(
            {
                "episode": ep_idx,
                "label": "success_demo",
                "final_progress": success_finals[-1],
                "final_success": float(success[-1]) if success else 0.0,
                "progress_curve": progress,
                "latency_s": latency,
            }
        )

        fail_frames = _synthetic_failure(frames)
        f_progress, f_success, f_lat = scorer.score_chw_frames(fail_frames, task)
        latencies.append(f_lat)
        failure_finals.append(float(f_progress[-1]) if f_progress else 0.0)
        episode_rows.append(
            {
                "episode": ep_idx,
                "label": "synthetic_failure",
                "final_progress": failure_finals[-1],
                "final_success": float(f_success[-1]) if f_success else 0.0,
                "progress_curve": f_progress,
                "latency_s": f_lat,
            }
        )

    metrics = evaluate_calibration_gates(
        success_progress_curves=success_curves,
        success_final_scores=success_finals,
        failure_final_scores=failure_finals,
    )
    # Handbook 3.4 needs human-judged failures; synthetic is smoke/sanity only.
    gates = {
        **metrics,
        "failure_source": "synthetic_hard_negative",
        "passed_handbook": False,
        "passed_synthetic_sanity": bool(metrics["passed"]),
        "passed": False,
    }
    if gates["passed_synthetic_sanity"]:
        recommendation = (
            "Synthetic sanity OK (Spearman + hard-neg AUC), but handbook gate still open: "
            "need human-labeled success/failure episodes before enabling episode_end online. "
            "Keep robometer_mode=disabled / deterministic reward; VRAM probe may proceed."
        )
    else:
        recommendation = (
            "Gates failed or incomplete. Keep robometer_mode=disabled / deterministic reward online; "
            "use Robometer only for offline/旁路 logging until recalibrated."
        )
    return {
        "dataset": str(dataset),
        "model": scorer.model_id,
        "mode": "real_inference",
        "status": "SCORED",
        "device": device,
        "image_key": image_key,
        "max_frames": max_frames,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "gates": gates,
        "episodes": episode_rows,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/lerobot/lerobot_merged"))
    parser.add_argument("--out", type=Path, default=Path("data/reward_calibration/offline_scores.json"))
    parser.add_argument("--model-id", default="lerobot/Robometer-4B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-key", default=IMAGE_KEYS[0])
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--max-success-eps", type=int, default=3)
    # 8 frames: enough temporal signal; 4 was too short and inverted AUC on weak negatives.
    parser.add_argument("--stub", action="store_true", help="Write placeholder report without loading 4B")
    args = parser.parse_args()

    if args.stub:
        report = _stub_report(args.dataset, args.model_id)
        _write_report(args.out, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"wrote {args.out}")
        return

    try:
        report = _run_real(
            dataset=args.dataset,
            model_id=args.model_id,
            device=args.device,
            image_key=args.image_key,
            max_frames=args.max_frames,
            max_success_eps=args.max_success_eps,
        )
    except Exception as exc:  # noqa: BLE001
        report = {
            "dataset": str(args.dataset),
            "model": args.model_id,
            "mode": "real_inference",
            "status": "FAILED",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "gates": {
                "spearman_progress": None,
                "success_vs_fail_auc": None,
                "spearman_progress_min": 0.7,
                "success_auc_min": 0.85,
                "n_success": 0,
                "n_failure": 0,
                "passed": False,
            },
            "recommendation": (
                f"Calibration failed ({type(exc).__name__}). "
                "If error mentions torchcodec/FFmpeg: use OpenCV loader (already preferred) or install pyav. "
                "If model load fails: ensure data/models/Robometer-4B + Qwen3-VL-4B-Instruct exist "
                "(python scripts/rl/prefetch_robometer_modelscope.py). "
                "Keep online reward deterministic until gates.passed."
            ),
        }

    _write_report(args.out, report)
    print(json.dumps({k: report[k] for k in report if k != "episodes"}, indent=2, ensure_ascii=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
