"""Export HIL bags → LeRobot v3 via KuavoBrain ``CvtRosbag2Lerobot`` (no shortcut).

Authority chain (must not skip):
  finished rosbag
    → KuavoRosbagReader.process_rosbag_chunked (Brain time-align)
    → CvtRosbag2Lerobot.populate_dataset_chunked / LeRobotDataset
    → codebase_version v3.0 (challenge vendored lerobot)

HIL default: **batch** — stage many ``*.bag`` into one dir, convert once into
one LeRobot dataset (``total_episodes`` = number of bags).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_ROOT = (
    ROOT
    / "third_party"
    / "kuavo_brain"
    / "kuavobrain-v2.x"
    / "src"
    / "kuavo_websocket_service_opensource"
    / "src"
    / "kuavo_data_challenge"
)
CVT_MODULE = "kuavo_data.CvtRosbag2Lerobot"
CVT_SCRIPT = CHALLENGE_ROOT / "kuavo_data" / "CvtRosbag2Lerobot.py"
CHALLENGE_LEROBOT_SRC = CHALLENGE_ROOT / "third_party" / "lerobot" / "src"
DEFAULT_HIL_CFG = ROOT / "configs" / "rl" / "KuavoRosbag2Lerobot_hil.yaml"

# Sim / HIL native cams → names CvtRosbag2Lerobot's KuavoRosbagReader expects.
SIM_CAMERA_TOPIC_ALIASES = {
    "head_cam_h": "/camera/color/image_raw",
    "wrist_cam_l": "/wrist_cam_l/color/image_raw",
    "wrist_cam_r": "/wrist_cam_r/color/image_raw",
}

# KuavoBrain v3 production color topics (H.265).
BRAIN_CAMERA_TOPICS = {
    "head_cam_h": "/cam_h/color/h265_stream",
    "wrist_cam_l": "/cam_l/color/h265_stream",
    "wrist_cam_r": "/cam_r/color/h265_stream",
}
# Older JPEG path still accepted in preflight soft-check.
BRAIN_CAMERA_TOPICS_JPEG_LEGACY = {
    "head_cam_h": "/cam_h/color/image_raw/compressed",
    "wrist_cam_l": "/cam_l/color/image_raw/compressed",
    "wrist_cam_r": "/cam_r/color/image_raw/compressed",
}

MIN_LIVE_BAG_BYTES = 200_000
BATCH_BAGS_SUBDIR = "batch_bags"
DEFAULT_LEROBOT_DIR_NAME = "lerobot_v3"


@dataclass
class BrainExportReport:
    status: str
    episode_id: str
    bag_path: str | None
    lerobot_dir: str | None
    reasons: list[str]
    episode_ids: list[str] = field(default_factory=list)
    bag_count: int = 0
    converter: str = str(CVT_SCRIPT)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "episode_id": self.episode_id,
            "episode_ids": self.episode_ids,
            "bag_count": self.bag_count,
            "bag_path": self.bag_path,
            "lerobot_dir": self.lerobot_dir,
            "reasons": self.reasons,
            "converter": self.converter,
            "lerobot_codebase_target": "v3.0",
            "note": "Batch CvtRosbag2Lerobot: many bags → one LeRobot v3 dataset",
        }


def assert_brain_converter() -> dict[str, Any]:
    """Sanity-check that the Brain tool exists and targets LeRobot v3.0."""
    reasons: list[str] = []
    if not CVT_SCRIPT.is_file():
        reasons.append(f"missing_converter:{CVT_SCRIPT}")
    ver_file = (
        CHALLENGE_LEROBOT_SRC / "lerobot" / "datasets" / "lerobot_dataset.py"
    )
    codebase = None
    if ver_file.is_file():
        text = ver_file.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if "CODEBASE_VERSION" in line and "=" in line and "v3" in line:
                codebase = line.strip()
                break
        if codebase is None:
            reasons.append("challenge_lerobot_codebase_version_not_v3")
    else:
        reasons.append(f"missing_challenge_lerobot:{ver_file}")
    ok = not reasons
    return {
        "ok": ok,
        "converter": str(CVT_SCRIPT),
        "challenge_root": str(CHALLENGE_ROOT),
        "codebase_version_line": codebase,
        "reasons": reasons,
    }


def session_bag_path(root: Path, episode_id: str) -> Path:
    return Path(root) / "sessions" / episode_id / "bags" / "original.bag"


def export_work_dir(root: Path) -> Path:
    return Path(root).resolve() / "lerobot_export_work"


def batch_bags_dir(root: Path) -> Path:
    return export_work_dir(root) / BATCH_BAGS_SUBDIR


def batch_lerobot_dir(root: Path, lerobot_dir_name: str = DEFAULT_LEROBOT_DIR_NAME) -> Path:
    # Driver writes: <batch_bags>/../<name>/lerobot
    return export_work_dir(root) / lerobot_dir_name / "lerobot"


def bag_export_preflight(bag_path: Path) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bag_path.is_file():
        return False, [f"bag_missing:{bag_path}"]
    size = bag_path.stat().st_size
    if size < MIN_LIVE_BAG_BYTES:
        reasons.append(
            f"bag_too_small:{size}B (need >={MIN_LIVE_BAG_BYTES}B; "
            "dry_run placeholder bags are not Brain-convertible — use --live-rosbag)"
        )
        return False, reasons
    try:
        import rosbag  # type: ignore

        with rosbag.Bag(str(bag_path), "r") as bag:
            topics = set(bag.get_type_and_topic_info()[1].keys())
        need_any_cam = (
            set(SIM_CAMERA_TOPIC_ALIASES.values())
            | set(BRAIN_CAMERA_TOPICS.values())
            | set(BRAIN_CAMERA_TOPICS_JPEG_LEGACY.values())
        )
        if not (topics & need_any_cam):
            reasons.append("missing_camera_topics_filled_zero")
        if "/sensors_data_raw" not in topics:
            reasons.append("bag_missing:/sensors_data_raw")
        if "/kuavo_arm_traj" not in topics and "/kuavo_arm_traj_synced" not in topics:
            reasons.append("missing_kuavo_arm_traj_filled_zero")
        hard = [r for r in reasons if r.startswith("bag_missing:/sensors")]
        return len(hard) == 0, reasons
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"bag_topic_probe_skipped:{exc}")
        return True, reasons


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def stage_bag_for_batch(
    *,
    root: Path,
    episode_id: str,
    bag_path: Path | None = None,
) -> tuple[Path | None, list[str]]:
    """Hardlink/copy one session bag into the flat batch_bags/ dir (no convert)."""
    root = Path(root).resolve()
    bag = Path(bag_path or session_bag_path(root, episode_id)).resolve()
    ok, reasons = bag_export_preflight(bag)
    if not ok:
        return None, reasons
    out_dir = batch_bags_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{episode_id}.bag"
    _link_or_copy(bag, dest)
    return dest, reasons


def list_train_ready_episode_ids(root: Path) -> list[str]:
    accepted = Path(root) / "accepted_replay"
    if not accepted.is_dir():
        return []
    eids: list[str] = []
    for d in sorted(accepted.iterdir()):
        if d.is_dir() and (d / "TRAIN_READY").exists():
            eids.append(d.name)
    return eids


def sync_batch_bags_from_accepted(
    root: Path,
    *,
    episode_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Stage all (or selected) TRAIN_READY session bags into batch_bags/.

    Returns (staged_episode_ids, soft_or_hard_reasons).
    """
    root = Path(root).resolve()
    eids = episode_ids if episode_ids is not None else list_train_ready_episode_ids(root)
    out_dir = batch_bags_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(eids)
    for old in out_dir.glob("*.bag"):
        if old.stem not in wanted:
            old.unlink()
    staged: list[str] = []
    reasons: list[str] = []
    for eid in eids:
        dest, why = stage_bag_for_batch(root=root, episode_id=eid)
        if dest is None:
            reasons.append(f"{eid}:{';'.join(why) or 'stage_failed'}")
            continue
        staged.append(eid)
        reasons.extend(f"{eid}:{r}" for r in why if r)
    return staged, reasons


def list_staged_batch_bags(root: Path) -> list[Path]:
    d = batch_bags_dir(root)
    if not d.is_dir():
        return []
    return sorted(d.glob("*.bag"))


def _challenge_pythonpath() -> str:
    parts = [
        str(CHALLENGE_ROOT),
        str(CHALLENGE_LEROBOT_SRC),
        str(ROOT),
    ]
    cur = os.environ.get("PYTHONPATH", "")
    if cur:
        parts.append(cur)
    return os.pathsep.join(parts)


def run_cvt_rosbag2lerobot(
    *,
    bag_dir: Path,
    lerobot_out_name: str,
    config_name: str = "KuavoRosbag2Lerobot_hil",
    config_dir: Path | None = None,
    topic_profile: str = "sim",
    extra_overrides: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Invoke Brain converter. ``config_dir`` defaults to repo configs/rl."""
    cfg_dir = config_dir or (ROOT / "configs" / "rl")
    driver = ROOT / "scripts" / "rl" / "_run_brain_cvt_rosbag2lerobot.py"
    cmd = [
        sys.executable,
        "-u",
        str(driver),
        "--bag-dir",
        str(Path(bag_dir).resolve()),
        "--lerobot-dir-name",
        lerobot_out_name,
        "--config-dir",
        str(Path(cfg_dir).resolve()),
        "--config-name",
        config_name,
        "--topic-profile",
        topic_profile,
    ]
    if extra_overrides:
        cmd.extend(["--override", *extra_overrides])
    env = os.environ.copy()
    env["PYTHONPATH"] = _challenge_pythonpath()
    return subprocess.run(
        cmd,
        cwd=str(CHALLENGE_ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def export_batch_to_lerobot(
    *,
    root: Path,
    episode_ids: list[str] | None = None,
    resync_from_accepted: bool = True,
    lerobot_dir_name: str = DEFAULT_LEROBOT_DIR_NAME,
    topic_profile: str = "sim",
    task_description: str | None = None,
) -> BrainExportReport:
    """Convert all staged (or selected TRAIN_READY) bags → one LeRobot v3 dataset."""
    check = assert_brain_converter()
    if not check["ok"]:
        return BrainExportReport(
            status="Block",
            episode_id="batch",
            bag_path=None,
            lerobot_dir=None,
            reasons=check["reasons"],
        )

    root = Path(root).resolve()
    soft: list[str] = []
    if resync_from_accepted or episode_ids is not None:
        staged, sync_reasons = sync_batch_bags_from_accepted(
            root, episode_ids=episode_ids
        )
        soft.extend(sync_reasons)
        if episode_ids is not None and not staged:
            return BrainExportReport(
                status="Rejected",
                episode_id="batch",
                bag_path=str(batch_bags_dir(root)),
                lerobot_dir=None,
                reasons=["no_bags_staged", *soft],
                episode_ids=[],
                bag_count=0,
            )
    else:
        staged = [p.stem for p in list_staged_batch_bags(root)]

    bag_dir = batch_bags_dir(root)
    bags = list_staged_batch_bags(root)
    if not bags:
        return BrainExportReport(
            status="Rejected",
            episode_id="batch",
            bag_path=str(bag_dir),
            lerobot_dir=None,
            reasons=["batch_bags_empty", *soft],
            episode_ids=[],
            bag_count=0,
        )

    eids = [p.stem for p in bags]
    overrides: list[str] = []
    if task_description:
        overrides.append(f"dataset.task_description={task_description}")

    proc = run_cvt_rosbag2lerobot(
        bag_dir=bag_dir,
        lerobot_out_name=lerobot_dir_name,
        topic_profile=topic_profile,
        extra_overrides=overrides or None,
    )
    out = batch_lerobot_dir(root, lerobot_dir_name)
    info = out / "meta" / "info.json"
    if proc.returncode != 0 or not info.exists():
        return BrainExportReport(
            status="Failed",
            episode_id="batch",
            bag_path=str(bag_dir),
            lerobot_dir=str(out) if out.exists() else None,
            reasons=[
                f"cvt_exit={proc.returncode}",
                (proc.stderr or "")[-2000:],
                (proc.stdout or "")[-1000:],
                *soft[:8],
            ],
            episode_ids=eids,
            bag_count=len(bags),
        )

    meta = {}
    try:
        meta = json.loads(info.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass
    prov = {
        "mode": "batch",
        "episode_ids": eids,
        "bag_count": len(bags),
        "source_bags": [str(p) for p in bags],
        "total_episodes_meta": meta.get("total_episodes"),
        "total_frames_meta": meta.get("total_frames"),
        "converter": str(CVT_SCRIPT),
        "brain_check": check,
        "topic_profile": topic_profile,
    }
    (out / "hil_export_provenance.json").write_text(
        json.dumps(prov, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return BrainExportReport(
        status="ok",
        episode_id="batch",
        bag_path=str(bag_dir),
        lerobot_dir=str(out),
        reasons=soft,
        episode_ids=eids,
        bag_count=len(bags),
    )


def export_episode_bag_to_lerobot(
    *,
    root: Path,
    episode_id: str,
    bag_path: Path | None = None,
    work_dir: Path | None = None,  # noqa: ARG001 — kept for call-site compat
    lerobot_dir_name: str = DEFAULT_LEROBOT_DIR_NAME,
    topic_profile: str = "sim",
    task_description: str | None = None,
) -> BrainExportReport:
    """Stage one episode into the batch dir, then convert the full batch."""
    dest, reasons = stage_bag_for_batch(
        root=root, episode_id=episode_id, bag_path=bag_path
    )
    if dest is None:
        return BrainExportReport(
            status="Rejected",
            episode_id=episode_id,
            bag_path=str(bag_path or session_bag_path(root, episode_id)),
            lerobot_dir=None,
            reasons=reasons,
            episode_ids=[episode_id],
            bag_count=0,
        )
    report = export_batch_to_lerobot(
        root=root,
        resync_from_accepted=True,
        lerobot_dir_name=lerobot_dir_name,
        topic_profile=topic_profile,
        task_description=task_description,
    )
    # Keep the triggering eid visible in the report.
    if report.status == "ok":
        report.episode_id = episode_id
    return report
