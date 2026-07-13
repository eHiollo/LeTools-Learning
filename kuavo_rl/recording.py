"""Episode recording and experiment manifest helpers."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


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
