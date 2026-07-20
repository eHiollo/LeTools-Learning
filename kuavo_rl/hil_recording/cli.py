"""Local diagnostics for HIL recording sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kuavo_rl.hil_recording.config import RecordingConfig
from kuavo_rl.hil_recording.database import HILDatabase
from kuavo_rl.hil_recording.session import HILRecordingSession


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m kuavo_rl.hil_recording.cli")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/rl_runs/hilserl_episodes/hilserl_vr"),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("recover", help="Recover interrupted sessions")
    p_show = sub.add_parser("show", help="Show session snapshot from SQLite")
    p_show.add_argument("episode_id")
    p_list = sub.add_parser("list-active", help="List active sessions")
    _ = p_list

    args = parser.parse_args(argv)
    cfg = RecordingConfig(root_dir=args.root)
    if args.cmd == "recover":
        session = HILRecordingSession(cfg)
        report = session.recover_interrupted()
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
        session.close()
        return
    db = HILDatabase(cfg.db_path)
    if args.cmd == "show":
        print(json.dumps(db.snapshot(args.episode_id).to_dict(), indent=2, ensure_ascii=False))
    elif args.cmd == "list-active":
        rows = db.list_active_sessions()
        print(json.dumps(rows, indent=2, default=str, ensure_ascii=False))
    db.close()


if __name__ == "__main__":
    main()
