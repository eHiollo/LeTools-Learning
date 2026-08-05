"""Write RL-100-compatible zarr datasets."""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from kuavo_rl.rl100_zarr.schema import (
    RETURN_GAMMA,
    ZarrEpisodeBuffers,
    compute_return,
)
from kuavo_rl.contracts import (
    RL100_ACTION_DIM,
    RL100_ARM_JOINT_NAMES,
    RL100_DEXHAND_JOINT_NAMES,
    RL100_HAND_DEFAULT,
    RL100_STATE_DIM,
    RL100_TOPIC_NATIVE_CONTRACT,
)

ZARR_CHUNK_LEAD = 100


def write_rl100_zarr(
    buffers: ZarrEpisodeBuffers,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    attrs: dict[str, Any] | None = None,
) -> Path:
    """Persist buffers to ``output_path`` (directory zarr store)."""
    try:
        import zarr
    except ImportError as exc:  # noqa: BLE001
        raise RuntimeError(
            "zarr is required for RL-100 export; pip install 'zarr>=2.12,<3'"
        ) from exc

    if len(buffers) == 0:
        raise RuntimeError("no transitions collected, refusing to write empty zarr")
    if buffers.n_episodes() == 0:
        raise RuntimeError("episode_ends is empty; call close_episode() per episode")

    required_attrs = {
        "contract": RL100_TOPIC_NATIVE_CONTRACT,
        "state_dim": RL100_STATE_DIM,
        "action_dim": RL100_ACTION_DIM,
        "state_order": "raw_joint_q20 + dexhand_position12",
        "action_order": "arm14_deg + left_hand6_raw + right_hand6_raw",
        "state_units": {"raw_joint_q20": "rad", "dexhand_position12": "0..100"},
        "action_units": {"arm14_deg": "degree", "hand12_raw": "0..100"},
        "dexhand_joint_names": list(RL100_DEXHAND_JOINT_NAMES),
        "arm_joint_names": list(RL100_ARM_JOINT_NAMES),
        "hand_default": RL100_HAND_DEFAULT.tolist(),
        "schema": "rl100-zarr-v1",
    }
    requested_attrs = attrs or {}
    if float(requested_attrs.get("smooth_penalty", 0.0)) != 0.0:
        raise ValueError("topic-native zarr requires smooth_penalty=0.0")
    for key, expected in required_attrs.items():
        if key in requested_attrs and requested_attrs[key] != expected:
            raise ValueError(f"zarr required attr {key} must be {expected!r}")

    out = Path(output_path)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"zarr already exists: {out} (pass overwrite=True)")
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.group(str(out))
    data = root.create_group("data")
    meta = root.create_group("meta")

    manifest = list(buffers.source_manifest)
    root.attrs["source_manifest"] = manifest
    root.attrs["contract"] = RL100_TOPIC_NATIVE_CONTRACT
    root.attrs["state_dim"] = RL100_STATE_DIM
    root.attrs["action_dim"] = RL100_ACTION_DIM
    root.attrs["state_order"] = "raw_joint_q20 + dexhand_position12"
    root.attrs["action_order"] = "arm14_deg + left_hand6_raw + right_hand6_raw"
    root.attrs["state_units"] = {"raw_joint_q20": "rad", "dexhand_position12": "0..100"}
    root.attrs["action_units"] = {"arm14_deg": "degree", "hand12_raw": "0..100"}
    root.attrs["dexhand_joint_names"] = list(RL100_DEXHAND_JOINT_NAMES)
    root.attrs["arm_joint_names"] = list(RL100_ARM_JOINT_NAMES)
    root.attrs["hand_default"] = RL100_HAND_DEFAULT.tolist()
    root.attrs["smooth_penalty"] = float((attrs or {}).get("smooth_penalty", 0.0))
    root.attrs["collection_config_sha256"] = str((attrs or {}).get("collection_config_sha256", "unknown"))
    root.attrs["schema"] = "rl100-zarr-v1"
    if attrs:
        for k, v in attrs.items():
            if k in required_attrs or k in {"smooth_penalty", "collection_config_sha256"}:
                continue
            root.attrs[k] = v

    try:
        from numcodecs import Blosc

        compressor = Blosc(cname="zstd", clevel=3, shuffle=1)
    except Exception:  # noqa: BLE001
        compressor = None

    def _create(group, name: str, array: np.ndarray, chunks=None) -> None:
        if hasattr(group, "create_dataset"):
            kwargs: dict[str, Any] = {
                "data": array,
                "overwrite": True,
                "dtype": array.dtype,
            }
            if chunks is not None:
                kwargs["chunks"] = chunks
            if compressor is not None:
                kwargs["compressor"] = compressor
            group.create_dataset(name, **kwargs)
            return
        kwargs = {"data": array, "overwrite": True}
        if chunks is not None:
            kwargs["chunks"] = chunks
        group.create_array(name, **kwargs)

    point_cloud = np.stack(buffers.point_cloud, axis=0).astype(np.float32)
    next_point_cloud = np.stack(buffers.next_point_cloud, axis=0).astype(np.float32)
    state = np.stack(buffers.state, axis=0).astype(np.float32)
    next_state = np.stack(buffers.next_state, axis=0).astype(np.float32)
    action = np.stack(buffers.action, axis=0).astype(np.float32)
    next_action = np.stack(buffers.next_action, axis=0).astype(np.float32)
    reward = np.asarray(buffers.reward, dtype=np.float32).reshape(-1, 1)
    done = np.asarray(buffers.done, dtype=bool).reshape(-1, 1)
    timeout = np.asarray(buffers.timeout, dtype=bool).reshape(-1, 1)
    episode_ends = np.asarray(buffers.episode_ends, dtype=np.int64)
    not_done = 1.0 - (done | timeout).astype(np.float32)
    returns = compute_return(reward, not_done, gamma=RETURN_GAMMA)

    _create(data, "point_cloud", point_cloud, chunks=(ZARR_CHUNK_LEAD, *point_cloud.shape[1:]))
    del point_cloud
    gc.collect()
    _create(
        data,
        "next_point_cloud",
        next_point_cloud,
        chunks=(ZARR_CHUNK_LEAD, *next_point_cloud.shape[1:]),
    )
    del next_point_cloud
    gc.collect()

    _create(data, "state", state, chunks=(ZARR_CHUNK_LEAD, state.shape[1]))
    _create(data, "next_state", next_state, chunks=(ZARR_CHUNK_LEAD, next_state.shape[1]))
    _create(data, "action", action, chunks=(ZARR_CHUNK_LEAD, action.shape[1]))
    _create(data, "next_action", next_action, chunks=(ZARR_CHUNK_LEAD, next_action.shape[1]))
    _create(data, "reward", reward, chunks=(ZARR_CHUNK_LEAD, 1))
    _create(data, "return", returns, chunks=(ZARR_CHUNK_LEAD, 1))
    _create(data, "done", done, chunks=(ZARR_CHUNK_LEAD, 1))
    _create(data, "timeout", timeout, chunks=(ZARR_CHUNK_LEAD, 1))
    _create(meta, "episode_ends", episode_ends)

    root.attrs["audit_fields"] = []
    if buffers.audit:
        written_audit: list[str] = []
        skipped_audit: list[str] = []
        for name, values in sorted(buffers.audit.items()):
            array = np.asarray(values)
            if array.shape[0] != len(buffers):
                raise ValueError(f"audit field {name!r} length mismatch")
            # zarr v2 does not reliably support object arrays.  Store strings
            # as fixed-width bytes and keep numeric/bool fields unchanged.
            if array.dtype.kind == "O":
                array = np.asarray([str(v) for v in values], dtype="S64")
            elif array.dtype.kind == "U":
                array = array.astype("S64")
            if array.dtype.kind not in "biufSc":
                skipped_audit.append(name)
                continue
            _create(meta, name, array, chunks=(ZARR_CHUNK_LEAD, *array.shape[1:]))
            written_audit.append(name)
        root.attrs["audit_fields"] = written_audit
        if skipped_audit:
            root.attrs["audit_fields_skipped"] = skipped_audit

    return out
