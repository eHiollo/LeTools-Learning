"""Result event priority / dedup helpers."""

from __future__ import annotations

from kuavo_rl.hil_recording.models import RESULT_PRIORITY, ResultEvent


def prefer_result(current: str | None, incoming: str) -> str:
    """Higher priority wins; estop/fault > abort > failure > success."""
    if current is None:
        return incoming
    if RESULT_PRIORITY.get(incoming, 0) >= RESULT_PRIORITY.get(current, 0):
        return incoming
    return current


def is_terminal_result(event_type: str) -> bool:
    return event_type in RESULT_PRIORITY


def should_quarantine_result(event_type: str | None) -> bool:
    return event_type in {"estop", "fault"}


def validate_event(event: ResultEvent) -> None:
    if event.event_type not in RESULT_PRIORITY:
        raise ValueError(f"unknown result event_type={event.event_type!r}")
