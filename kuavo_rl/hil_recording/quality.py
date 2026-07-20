"""Post-stop bag / sidecar quality checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kuavo_rl.hil_recording.models import QualityReport
from kuavo_rl.hil_recording.topics import ResolvedTopics


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def check_bag_readable(bag_path: Path) -> tuple[bool, str | None]:
    if not bag_path.exists():
        # dry-run may leave partial file; also accept non-empty sibling
        if bag_path.with_suffix(".bag.active").exists():
            return True, None
        return False, f"bag_missing:{bag_path}"
    if bag_path.stat().st_size <= 0:
        return False, "bag_empty"
    # Prefer rosbag info when available; otherwise size>0 is enough for dry-run.
    try:
        import rosbag  # type: ignore

        with rosbag.Bag(str(bag_path), "r") as bag:
            _ = bag.get_type_and_topic_info()
        return True, None
    except ImportError:
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"bag_unreadable:{exc}"


def compare_sidecar_and_audit(
    staging_dir: Path,
    session_dir: Path,
    *,
    max_step_delta: int = 0,
) -> tuple[bool, int, int | None, list[str]]:
    reasons: list[str] = []
    sidecar = staging_dir / "transitions.jsonl"
    # Prefer local audit mirror (always present); bag topic count optional.
    audit = session_dir / "hil_transition_audit.jsonl"
    sidecar_n = count_jsonl_lines(sidecar)
    audit_n = count_jsonl_lines(audit) if audit.exists() else None
    if audit_n is None:
        reasons.append("audit_mirror_missing")
        return False, sidecar_n, None, reasons
    if abs(sidecar_n - audit_n) > max_step_delta:
        reasons.append(f"sidecar_audit_mismatch sidecar={sidecar_n} audit={audit_n}")
        return False, sidecar_n, audit_n, reasons
    return True, sidecar_n, audit_n, reasons


def run_quality_check(
    *,
    bag_path: Path,
    staging_dir: Path,
    session_dir: Path,
    resolved: ResolvedTopics,
    require_mask_fields: bool = True,
    dry_run: bool = False,
) -> QualityReport:
    reasons: list[str] = []
    topic_issues: list[str] = []

    readable, err = check_bag_readable(bag_path)
    if not readable and not dry_run:
        reasons.append(err or "bag_unreadable")
    elif not readable and dry_run:
        # dry-run fake bag may still be partial; accept if staging has data
        if count_jsonl_lines(staging_dir / "transitions.jsonl") == 0:
            reasons.append(err or "bag_unreadable")

    match, sidecar_n, audit_n, cmp_reasons = compare_sidecar_and_audit(
        staging_dir, session_dir
    )
    reasons.extend(cmp_reasons)

    if require_mask_fields and sidecar_n > 0:
        path = staging_dir / "transitions.jsonl"
        with path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                extras = row.get("extras") or {}
                if "intervention_mask" not in extras and "intervention_mask" not in row:
                    reasons.append(f"missing_intervention_mask_at_step_{i}")
                    break
                if (
                    "intervention_segment_step" not in extras
                    and "intervention_segment_step" not in row
                ):
                    reasons.append(f"missing_intervention_segment_step_at_step_{i}")
                    break

    # Export-required topics: soft warn only. Schema stays fixed; missing streams
    # are treated as zeros downstream (do not fail the episode).
    if not dry_run:
        try:
            import rosbag  # type: ignore

            if bag_path.exists():
                with rosbag.Bag(str(bag_path), "r") as bag:
                    info = bag.get_type_and_topic_info()[1]
                    present = set(info.keys())
                for spec in resolved.for_export():
                    if spec.name.startswith("/hil/"):
                        continue  # checked via audit mirror
                    if spec.name not in present:
                        topic_issues.append(
                            f"missing_export_topic_filled_zero:{spec.name}"
                        )
        except Exception:
            pass

    # Hard failures only (not missing optional/export topics).
    if reasons:
        status = "Failed"
    else:
        status = "Healthy"

    return QualityReport(
        status=status,
        bag_readable=readable or dry_run,
        sidecar_step_count=sidecar_n,
        bag_audit_step_count=audit_n,
        sidecar_bag_match=match,
        topic_issues=topic_issues,
        reasons=reasons,
        details={
            "dry_run": dry_run,
            "missing_topics_policy": "fill_zero_keep_schema",
            "topic_issues_soft": topic_issues,
        },
    )


def write_quality_report(session_dir: Path, report: QualityReport) -> Path:
    path = session_dir / "quality_report.json"
    path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path
