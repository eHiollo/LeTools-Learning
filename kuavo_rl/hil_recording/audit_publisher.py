"""Publish /hil/transition_audit and /hil/result_event into ROS (or local sidecar mirror)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from kuavo_rl.hil_recording.models import ResultEvent, TimeStamps
from kuavo_rl.hil_recording.timebase import now_stamps


TRANSITION_TOPIC = "/hil/transition_audit"
RESULT_TOPIC = "/hil/result_event"


class AuditPublisher:
    """Best-effort ROS String publishers + local JSONL mirror for bag-less tests."""

    def __init__(self, session_dir: Path, *, enable_ros: bool = True):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._transition_path = self.session_dir / "hil_transition_audit.jsonl"
        self._result_path = self.session_dir / "hil_result_event.jsonl"
        self._ros_ok = False
        self._pub_transition = None
        self._pub_result = None
        if enable_ros:
            self._try_init_ros()

    def _try_init_ros(self) -> None:
        try:
            import rospy  # type: ignore
            from std_msgs.msg import String  # type: ignore

            if not rospy.core.is_initialized():
                rospy.init_node("hil_audit_publisher", anonymous=True, disable_signals=True)
            self._pub_transition = rospy.Publisher(TRANSITION_TOPIC, String, queue_size=50)
            self._pub_result = rospy.Publisher(RESULT_TOPIC, String, queue_size=20)
            self._ros_ok = True
        except Exception:
            self._ros_ok = False

    def publish_transition(self, payload: dict[str, Any], stamps: TimeStamps | None = None) -> None:
        stamps = stamps or now_stamps()
        row = {
            "topic": TRANSITION_TOPIC,
            "stamps": stamps.to_dict(),
            **payload,
        }
        text = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with self._transition_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
            if self._ros_ok and self._pub_transition is not None:
                try:
                    from std_msgs.msg import String  # type: ignore

                    self._pub_transition.publish(String(data=text))
                except Exception:
                    pass

    def publish_result(self, event: ResultEvent) -> None:
        row = {
            "topic": RESULT_TOPIC,
            "episode_id": event.episode_id,
            "event_type": event.event_type,
            "source": event.source,
            "stamps": event.stamps.to_dict(),
            "payload": event.payload,
        }
        text = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with self._result_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")
            if self._ros_ok and self._pub_result is not None:
                try:
                    from std_msgs.msg import String  # type: ignore

                    self._pub_result.publish(String(data=text))
                except Exception:
                    pass

    def transition_count(self) -> int:
        if not self._transition_path.exists():
            return 0
        return sum(1 for _ in self._transition_path.open("r", encoding="utf-8"))

    def close(self) -> None:
        pass
