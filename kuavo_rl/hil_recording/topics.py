"""Topic profile resolver (role / mode / required_for_*)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kuavo_rl.hil_recording.models import TOPICS_VERSION

_DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "configs" / "rl" / "hil_topics_v002.yaml"


@dataclass
class TopicSpec:
    name: str
    role: str = "training"  # training | audit | calibration
    mode: str = "streaming"  # streaming | latched
    required_for_start: bool = False
    required_for_export: bool = False
    min_hz: float | None = None
    freshness_s: float | None = None
    profiles: list[str] | None = None

    def applies_to(self, control_profile: str) -> bool:
        if not self.profiles:
            return True
        return control_profile in self.profiles


@dataclass
class ResolvedTopics:
    version: str
    robot_type: str
    eef_type: str
    control_profile: str
    topics: list[TopicSpec] = field(default_factory=list)

    def for_start(self) -> list[TopicSpec]:
        return [t for t in self.topics if t.required_for_start and t.applies_to(self.control_profile)]

    def for_export(self) -> list[TopicSpec]:
        return [t for t in self.topics if t.required_for_export and t.applies_to(self.control_profile)]

    def record_topic_names(self) -> list[str]:
        """Topics to pass to rosbag record (training + audit + hil)."""
        names: list[str] = []
        for t in self.topics:
            if not t.applies_to(self.control_profile):
                continue
            if t.role in ("training", "audit", "calibration") or t.name.startswith("/hil/"):
                names.append(t.name)
        # de-dup preserve order
        seen: set[str] = set()
        out: list[str] = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def to_json(self) -> str:
        payload = {
            "version": self.version,
            "robot_type": self.robot_type,
            "eef_type": self.eef_type,
            "control_profile": self.control_profile,
            "topics": [asdict(t) for t in self.topics],
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, text: str) -> "ResolvedTopics":
        data = json.loads(text)
        topics = [TopicSpec(**t) for t in data.get("topics", [])]
        return cls(
            version=data.get("version", TOPICS_VERSION),
            robot_type=data.get("robot_type", "Kuavo"),
            eef_type=data.get("eef_type", "leju_claw"),
            control_profile=data.get("control_profile", "act_vr"),
            topics=topics,
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML required to load topic profiles") from exc
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_topics(
    *,
    control_profile: str = "act_vr",
    robot_type: str = "Kuavo",
    eef_type: str = "leju_claw",
    profile_path: str | Path | None = None,
) -> ResolvedTopics:
    path = Path(profile_path) if profile_path else _DEFAULT_PROFILE
    if not path.exists():
        # Minimal embedded fallback for tests without config tree.
        return ResolvedTopics(
            version=TOPICS_VERSION,
            robot_type=robot_type,
            eef_type=eef_type,
            control_profile=control_profile,
            topics=[
                TopicSpec(
                    name="/sensors_data_raw",
                    role="training",
                    mode="streaming",
                    required_for_start=True,
                    required_for_export=True,
                    min_hz=25,
                    freshness_s=1.0,
                ),
                TopicSpec(
                    name="/tf_static",
                    role="calibration",
                    mode="latched",
                    required_for_start=True,
                    required_for_export=True,
                ),
                TopicSpec(
                    name="/kuavo_arm_traj",
                    role="training",
                    mode="streaming",
                    required_for_start=False,
                    required_for_export=True,
                    profiles=["act_vr", "vr_only"],
                    min_hz=25,
                    freshness_s=1.0,
                ),
                TopicSpec(
                    name="/hil/transition_audit",
                    role="audit",
                    mode="streaming",
                    required_for_start=False,
                    required_for_export=True,
                ),
                TopicSpec(
                    name="/hil/result_event",
                    role="audit",
                    mode="streaming",
                    required_for_start=False,
                    required_for_export=True,
                ),
            ],
        )

    data = _load_yaml(path)
    topics = [TopicSpec(**t) for t in data.get("topics", [])]
    return ResolvedTopics(
        version=str(data.get("version", TOPICS_VERSION)),
        robot_type=str(data.get("robot_type", robot_type)),
        eef_type=str(data.get("eef_type", eef_type)),
        control_profile=control_profile,
        topics=topics,
    )
