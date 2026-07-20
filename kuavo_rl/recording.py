"""Episode recording and experiment manifest helpers."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TransitionRecord:
    experiment_id: str
    episode_id: str
    step_id: int
    timestamp: float
    action: list[float]
    reward: float
    reward_source: str
    terminated: bool
    truncated: bool
    fault_code: str
    is_intervention: bool
    action_clipped: bool
    weight_version: str = "none"
    contract_version: str = "v62-16d-r1"
    extras: dict[str, Any] = field(default_factory=dict)


class EpisodeRecorder:
    def __init__(self, root: str | Path, experiment_id: str):
        self.root = Path(root)
        self.experiment_id = experiment_id
        self.dir = self.root / experiment_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.dir / "transitions.jsonl", "a", encoding="utf-8")

    def log(self, record: TransitionRecord) -> None:
        self._fp.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


class HILReplayWriter:
    """Durable, episode-oriented replay storage for HIL-SERL collection.

    Legacy layout (default):
      ``<root>/<experiment_id>/replay/episodes/<episode_id>/``

    Staging layout (``staging_dir`` set — preferred by hil_recording):
      ``<staging_dir>/{transitions.jsonl, frames/}``
      Training loaders must only read ``accepted_replay/`` after publish.
    """

    def __init__(
        self,
        root: str | Path,
        experiment_id: str,
        *,
        jpeg_quality: int = 90,
        staging_dir: str | Path | None = None,
    ):
        self.experiment_id = experiment_id
        self.staging_dir = Path(staging_dir) if staging_dir is not None else None
        if self.staging_dir is not None:
            self.root = self.staging_dir
            self.root.mkdir(parents=True, exist_ok=True)
            (self.root / "frames").mkdir(exist_ok=True)
            self.episodes_root = self.root
        else:
            self.root = Path(root) / experiment_id / "replay"
            self.root.mkdir(parents=True, exist_ok=True)
            self.episodes_root = self.root / "episodes"
            self.episodes_root.mkdir(exist_ok=True)
        self.jpeg_quality = int(np.clip(jpeg_quality, 1, 100))
        self._files: dict[str, Any] = {}
        self._write_schema()

    def _write_schema(self) -> None:
        path = self.root / "schema.json"
        if not path.exists():
            fmt = "kuavo-hil-replay-v2-staging" if self.staging_dir is not None else "kuavo-hil-replay-v1"
            path.write_text(
                json.dumps(
                    {
                        "format": fmt,
                        "transition": "obs, action, reward, next_obs, done, intervention",
                        "state_key": "observation.state",
                        "images": "JPEG when OpenCV is available, otherwise .npy",
                        "note": "Derived artifact; H.265 rosbag is image source of truth.",
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def _episode_dir(self, episode_id: str) -> Path:
        if self.staging_dir is not None:
            directory = self.root
        else:
            directory = self.episodes_root / episode_id
        (directory / "frames").mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _write_image(self, frame_path: Path, image: Any) -> Path:
        arr = self._array(image)
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3:
            raise ValueError(f"image must be HWC/CHW, got {arr.shape}")
        if arr.dtype != np.uint8:
            if arr.size and float(np.max(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        try:
            import cv2

            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.shape[-1] == 3 else arr
            output = frame_path.with_suffix(".jpg")
            if not cv2.imwrite(str(output), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]):
                raise RuntimeError("cv2.imwrite returned false")
            return output
        except Exception:
            output = frame_path.with_suffix(".npy")
            np.save(output, arr)
            return output

    def _write_observation(self, episode_dir: Path, step_id: int, prefix: str, obs: dict) -> dict[str, Any]:
        state = self._array(obs["observation.state"]).astype(np.float32).reshape(-1)
        state_path = episode_dir / "frames" / f"{step_id:06d}_{prefix}_state.npy"
        np.save(state_path, state)
        images: dict[str, str] = {}
        for key, value in obs.items():
            if not key.startswith("observation.images."):
                continue
            safe_key = key.rsplit(".", 1)[-1]
            path = self._write_image(episode_dir / "frames" / f"{step_id:06d}_{prefix}_{safe_key}", value)
            images[key] = str(path.relative_to(self.root))
        return {"state": str(state_path.relative_to(self.root)), "images": images}

    def log_transition(
        self,
        record: TransitionRecord,
        *,
        observation: dict,
        next_observation: dict,
    ) -> None:
        episode_dir = self._episode_dir(record.episode_id)
        fp = self._files.get(record.episode_id)
        if fp is None:
            fp = open(episode_dir / "transitions.jsonl", "a", encoding="utf-8")
            self._files[record.episode_id] = fp
        row = asdict(record)
        row["observation"] = self._write_observation(episode_dir, record.step_id, "obs", observation)
        row["next_observation"] = self._write_observation(episode_dir, record.step_id, "next", next_observation)
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        fp.flush()

    def close(self) -> None:
        for fp in self._files.values():
            fp.close()
        self._files = {}


def _safe_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def build_manifest(
    experiment_id: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "experiment_id": experiment_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_head": _safe_cmd(["git", "rev-parse", "HEAD"]),
        "lerobot_git_head": _safe_cmd(
            ["git", "-C", "third_party/lerobot", "rev-parse", "HEAD"]
        ),
        "contract_version": "v62-16d-r1",
        "notes": "Do not record secrets/tokens in manifests.",
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
