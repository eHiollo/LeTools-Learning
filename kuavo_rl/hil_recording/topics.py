"""Topic profile resolver (role / mode / required_for_*)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from kuavo_rl.hil_recording.models import TOPICS_VERSION

_DEFAULT_PROFILE = Path(__file__).resolve().parents[2] / "configs" / "rl" / "hil_topics_v002.yaml"


@dataclass
class TopicSpec:
    """One HIL topic.

    ``name`` is the canonical name written into the bag / export schema.
    Optional ``source`` is the live ROS bus topic (lower-machine / driver name).
    When ``source`` differs from ``name``, the recorder relays ``source`` → ``name``
    so bags match VLA reference layouts without changing the robot publishers.
    """

    name: str
    role: str = "training"  # training | audit | calibration
    mode: str = "streaming"  # streaming | latched
    required_for_start: bool = False
    required_for_export: bool = False
    min_hz: float | None = None
    freshness_s: float | None = None
    profiles: list[str] | None = None
    source: str | None = None

    def applies_to(self, control_profile: str) -> bool:
        if not self.profiles:
            return True
        return control_profile in self.profiles

    @property
    def bus_name(self) -> str:
        """Topic to probe on the live ROS bus (gate / freshness)."""
        return self.source or self.name

    def needs_relay(self) -> bool:
        return bool(self.source) and self.source != self.name


def _topic_spec_from_dict(raw: dict[str, Any]) -> TopicSpec:
    allowed = {f.name for f in fields(TopicSpec)}
    return TopicSpec(**{k: v for k, v in raw.items() if k in allowed})


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
        """Canonical topic names passed to rosbag record (after optional relay)."""
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

    def relay_pairs(self) -> list[tuple[str, str]]:
        """``(source, name)`` pairs that must be relayed before rosbag record."""
        pairs: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for t in self.topics:
            if not t.applies_to(self.control_profile) or not t.needs_relay():
                continue
            pair = (t.bus_name, t.name)
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
        return pairs

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
        topics = [_topic_spec_from_dict(t) for t in data.get("topics", [])]
        return cls(
            version=data.get("version", TOPICS_VERSION),
            robot_type=data.get("robot_type", "Kuavo"),
            eef_type=data.get("eef_type", "leju_claw"),
            control_profile=data.get("control_profile", "act_vr"),
            topics=topics,
        )


def _resolve_env_templates(value: Any) -> Any:
    """Resolve ``${ENV:NAME}`` / ``${ENV:NAME:default}`` in topic profile strings."""
    import os

    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        expr = value[2:-1]
        if expr.startswith("ENV:"):
            rest = expr[4:]
            if ":" in rest:
                name, default = rest.split(":", 1)
                return os.environ.get(name, default)
            name = rest
            if name not in os.environ:
                raise KeyError(
                    f"Environment variable {name!r} is required for topic template {value}"
                )
            return os.environ[name]
        return value
    if isinstance(value, list):
        return [_resolve_env_templates(v) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_env_templates(v) for k, v in value.items()}
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML required to load topic profiles") from exc
    with path.open("r", encoding="utf-8") as f:
        return _resolve_env_templates(yaml.safe_load(f) or {})


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
    topics = [_topic_spec_from_dict(t) for t in data.get("topics", [])]
    return ResolvedTopics(
        version=str(data.get("version", TOPICS_VERSION)),
        robot_type=str(data.get("robot_type", robot_type)),
        eef_type=str(data.get("eef_type", eef_type)),
        control_profile=control_profile,
        topics=topics,
    )
