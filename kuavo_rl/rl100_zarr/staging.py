"""Per-episode NPZ staging + merge into RL-100 zarr."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from kuavo_rl.rl100_zarr.episode import append_labeled_episode
from kuavo_rl.rl100_zarr.schema import ZarrEpisodeBuffers
from kuavo_rl.rl100_zarr.writer import write_rl100_zarr


def save_episode_npz(
    path: str | Path,
    *,
    states: list[np.ndarray],
    actions: list[np.ndarray],
    point_clouds: list[np.ndarray],
    result_type: str,
    meta: dict[str, Any] | None = None,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "states": np.stack([np.asarray(s, dtype=np.float32) for s in states], axis=0),
        "actions": np.stack([np.asarray(a, dtype=np.float32) for a in actions], axis=0),
        "point_clouds": np.stack(
            [np.asarray(p, dtype=np.float32) for p in point_clouds], axis=0
        ),
        "result_type": np.asarray(result_type),
        "meta_json": np.asarray(json.dumps(meta or {}, ensure_ascii=False)),
    }
    np.savez_compressed(out, **payload)
    return out


def load_episode_npz(path: str | Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    meta_raw = data["meta_json"].item() if "meta_json" in data.files else "{}"
    if isinstance(meta_raw, bytes):
        meta_raw = meta_raw.decode("utf-8")
    result = data["result_type"].item() if "result_type" in data.files else "unknown"
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return {
        "states": [data["states"][i] for i in range(data["states"].shape[0])],
        "actions": [data["actions"][i] for i in range(data["actions"].shape[0])],
        "point_clouds": [
            data["point_clouds"][i] for i in range(data["point_clouds"].shape[0])
        ],
        "result_type": str(result),
        "meta": json.loads(str(meta_raw) or "{}"),
    }


def build_zarr_from_episode_dir(
    episode_dir: str | Path,
    zarr_path: str | Path,
    *,
    only_success: bool = False,
    overwrite: bool = False,
    lambda_penalty: float = 0.05,
    smooth_penalty: float = 0.01,
    max_episode_len: int = 2000,
    attrs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(episode_dir)
    files = sorted(root.glob("*.npz"))
    buffers = ZarrEpisodeBuffers()
    kept = 0
    skipped = 0
    for fp in files:
        ep = load_episode_npz(fp)
        ok = append_labeled_episode(
            buffers,
            states=ep["states"],
            actions=ep["actions"],
            point_clouds=ep["point_clouds"],
            result_type=ep["result_type"],
            lambda_penalty=lambda_penalty,
            smooth_penalty=smooth_penalty,
            max_episode_len=max_episode_len,
            only_success=only_success,
            source={"episode_file": str(fp), **(ep.get("meta") or {})},
        )
        if ok:
            kept += 1
        else:
            skipped += 1
    if kept == 0:
        raise RuntimeError(
            f"no episodes kept from {root} (files={len(files)}, skipped={skipped})"
        )
    out = write_rl100_zarr(buffers, zarr_path, overwrite=overwrite, attrs=attrs)
    return {
        "zarr_path": str(out),
        "episodes_kept": kept,
        "episodes_skipped": skipped,
        "transitions": len(buffers),
        "n_episodes": buffers.n_episodes(),
    }
