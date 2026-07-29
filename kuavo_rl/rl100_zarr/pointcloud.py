"""Depth → point cloud, multi-camera fuse, FPS downsample to RL-100 (1024, 3)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from kuavo_rl.rl100_zarr.schema import NUM_POINTS, POINT_DIM


def depth_to_point_cloud(
    depth_m: np.ndarray,
    intrinsics: Sequence[float],
    camera_pose: np.ndarray | None = None,
    *,
    min_depth: float = 0.05,
    max_depth: float = 3.0,
) -> np.ndarray:
    """Back-project depth (meters) to world/base frame points ``(M, 3)``.

    ``intrinsics`` is ``(fx, fy, cx, cy)``.
    ``camera_pose`` is 4x4 ``T_base_cam`` (camera → base). Identity keeps camera frame.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth expected (H, W), got {depth.shape}")
    fx, fy, cx, cy = (float(x) for x in intrinsics)
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.reshape(-1)
    valid = np.isfinite(z) & (z > float(min_depth)) & (z < float(max_depth))
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    u = u.reshape(-1)[valid]
    v = v.reshape(-1)[valid]
    z = z[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=1)

    if camera_pose is None:
        return pts_cam.astype(np.float32)
    T = np.asarray(camera_pose, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"camera_pose expected (4, 4), got {T.shape}")
    ones = np.ones((pts_cam.shape[0], 1), dtype=np.float32)
    pts_h = np.concatenate([pts_cam, ones], axis=1)
    pts_w = (T @ pts_h.T).T[:, :3]
    return pts_w.astype(np.float32)


def crop_workspace(
    points: np.ndarray,
    *,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 3)
    mask = np.ones((pts.shape[0],), dtype=bool)
    if x_range is not None:
        mask &= (pts[:, 0] > x_range[0]) & (pts[:, 0] < x_range[1])
    if y_range is not None:
        mask &= (pts[:, 1] > y_range[0]) & (pts[:, 1] < y_range[1])
    if z_range is not None:
        mask &= (pts[:, 2] > z_range[0]) & (pts[:, 2] < z_range[1])
    return pts[mask]


def fuse_point_clouds(clouds: Sequence[np.ndarray]) -> np.ndarray:
    parts = [np.asarray(c, dtype=np.float32).reshape(-1, 3) for c in clouds if c is not None]
    parts = [p for p in parts if p.size > 0]
    if not parts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def downsample_fps(points: np.ndarray, num_points: int = NUM_POINTS) -> np.ndarray:
    """Farthest-point style downsample to fixed ``num_points``.

    Prefers ``fpsample`` when installed; otherwise uses a deterministic
    random subsample / pad so unit tests and dry-runs still work.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, POINT_DIM)
    n = int(num_points)
    if n <= 0:
        raise ValueError("num_points must be positive")
    if pts.shape[0] == 0:
        return np.zeros((n, POINT_DIM), dtype=np.float32)

    try:
        import fpsample  # type: ignore

        sample = pts
        if sample.shape[0] < n:
            reps = int(np.ceil(n / sample.shape[0]))
            sample = np.concatenate([sample] * reps, axis=0)
        idx = fpsample.bucket_fps_kdtree_sampling(sample, n)
        return sample[idx].astype(np.float32)
    except Exception:
        return _fallback_resample(pts, n)


def _fallback_resample(pts: np.ndarray, n: int) -> np.ndarray:
    if pts.shape[0] >= n:
        # Even stride for stability without fpsample.
        idx = np.linspace(0, pts.shape[0] - 1, n).astype(np.int64)
        return pts[idx].astype(np.float32)
    out = np.zeros((n, POINT_DIM), dtype=np.float32)
    out[: pts.shape[0]] = pts
    need = n - pts.shape[0]
    pad_idx = np.arange(need) % max(pts.shape[0], 1)
    out[pts.shape[0] :] = pts[pad_idx]
    return out


def build_rl100_point_cloud(
    clouds: Sequence[np.ndarray],
    *,
    num_points: int = NUM_POINTS,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Fuse → workspace crop → FPS. Always returns ``(num_points, 3)``."""
    fused = fuse_point_clouds(clouds)
    cropped = crop_workspace(fused, x_range=x_range, y_range=y_range, z_range=z_range)
    return downsample_fps(cropped, num_points=num_points)
