#!/usr/bin/env python3
"""Simulate HIL recording closed loop with MockBackend (no Kuavo-Sim required).

Validates: create → register ACT producer → start → step transitions →
request_stop → wait_finalized → publish_pending_review → pending_review layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _ZeroChunkPolicy:
    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        return np.zeros((10, 16), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/rl_runs/hilserl_sim_verify/hilserl_vr"),
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--episode-id", default="sim_verify_ep1")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete root dir before run (default: keep DB for multi-episode).",
    )
    args = parser.parse_args()

    from kuavo_rl.act_runner import ActExecuteFirstRunner
    from kuavo_rl.adapter import make_kuavo_hilserl_env
    from kuavo_rl.config import ActRunnerConfig, EnvConfig
    from kuavo_rl.hil_recording import (
        FINALIZED_OK,
        HILRecordingSession,
        RecordingConfig,
        RecordRequest,
        ResultEvent,
        now_stamps,
    )
    from kuavo_rl.recording import HILReplayWriter, TransitionRecord

    out = {
        "ok": False,
        "episode_id": args.episode_id,
        "root": str(args.root),
    }

    cfg = RecordingConfig(
        root_dir=args.root,
        dry_run_recorder=True,
        skip_gate_ros=True,
        post_roll_s=0.1,
        bag_stall_timeout_s=60.0,
    )
    if args.fresh and args.root.exists():
        import shutil

        shutil.rmtree(args.root)

    session = HILRecordingSession(cfg)
    recover = session.recover_interrupted()
    out["recover"] = recover.to_dict()

    session.create(
        RecordRequest(
            episode_id=args.episode_id,
            task_id="sim_verify",
            control_profile="act",
            dry_run=True,
            skip_gate_ros=True,
            post_roll_s=0.1,
        )
    )
    session.register_producer("act_runner", os.getpid(), kind="policy")
    snap = session.start(args.episode_id)
    out["started_state"] = snap.session_state
    assert snap.session_state == "Recording", snap

    env = make_kuavo_hilserl_env(EnvConfig(), use_stub_robometer=True)
    runner = ActExecuteFirstRunner(_ZeroChunkPolicy(), ActRunnerConfig())
    writer = HILReplayWriter(
        args.root.parent, "unused", staging_dir=cfg.staging_dir(args.episode_id)
    )

    def on_step(step: dict) -> None:
        info = step["info"]
        stamps = now_stamps()
        executed = step.get("executed_action")
        policy_action = np.asarray(step["policy_action"], dtype=np.float32).tolist()
        rec = TransitionRecord(
            experiment_id="sim_verify",
            episode_id=args.episode_id,
            step_id=int(step["step_id"]),
            timestamp=stamps.wall_time_ns / 1e9,
            action=list(executed) if executed is not None else policy_action,
            reward=float(step["reward"]),
            reward_source=str(info.get("reward_source", "none")),
            terminated=bool(step["terminated"]),
            truncated=bool(step["truncated"]),
            fault_code=str(info.get("fault_code", "NONE")),
            is_intervention=bool(info.get("is_intervention", False)),
            action_clipped=bool(info.get("action_clipped", False)),
            extras={
                "policy_action": policy_action,
                "executed_action": executed,
                "intervention_mask": info.get("intervention_mask") or [0] * 16,
                "intervention_segment_id": info.get("intervention_segment_id", 0),
                "intervention_segment_step": info.get("intervention_segment_step", 0),
                "stamps": stamps.to_dict(),
            },
        )
        writer.log_transition(
            rec,
            observation=step["observation"],
            next_observation=step["next_observation"],
        )
        session.update_transition(
            {
                "step_id": int(step["step_id"]),
                "stamps": stamps,
                "policy_action": policy_action,
                "executed_action": executed,
                "intervention_mask": rec.extras["intervention_mask"],
                "intervention_segment_id": 0,
                "intervention_segment_step": 0,
                "reward": float(step["reward"]),
                "fault_code": rec.fault_code,
            }
        )
        if session.poll_stop_request() is not None:
            raise RuntimeError("unexpected_watchdog_stop")

    result = runner.run_episode(env, max_steps=args.steps, on_step=on_step)
    writer.close()
    session.record_event(
        ResultEvent(
            episode_id=args.episode_id,
            event_type="success",
            source="verify_hil_recording_sim",
            stamps=now_stamps(),
        )
    )
    session.request_stop(args.episode_id, reason="success")
    final = session.wait_finalized(args.episode_id, timeout_s=30.0)
    out["final_state"] = final.session_state
    out["result_type"] = final.result_type
    out["steps"] = result["n"]

    if final.session_state not in FINALIZED_OK:
        out["error"] = f"not finalized ok: {final.session_state} err={final.error_message}"
        session.close()
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 1

    export = session.publish_pending_review(args.episode_id)
    out["export"] = export.to_dict()
    pending = cfg.pending_review_dir / args.episode_id
    manifest = pending / "publish_manifest.json"
    transitions = pending / "transitions.jsonl"
    review_ready = pending / "REVIEW_READY"
    out["pending_exists"] = pending.exists()
    out["manifest_exists"] = manifest.exists()
    out["review_ready"] = review_ready.exists()
    out["accepted_blocked"] = not (cfg.accepted_replay_dir / args.episode_id).exists()
    out["transition_lines"] = (
        sum(1 for line in transitions.open() if line.strip()) if transitions.exists() else 0
    )
    if manifest.exists():
        out["manifest"] = json.loads(manifest.read_text(encoding="utf-8"))

    # staging should not retain train-visible transitions
    staging_t = cfg.session_dir(args.episode_id) / "staging" / "transitions.jsonl"
    out["staging_cleared"] = not staging_t.exists()

    ok = (
        export.status == "PendingReview"
        and out["pending_exists"]
        and out["manifest_exists"]
        and out["review_ready"]
        and out["accepted_blocked"]
        and out["transition_lines"] == args.steps
        and out["staging_cleared"]
        and out["manifest"].get("replay_schema_version") == "hil-replay-v002"
    )
    out["ok"] = ok
    session.close()
    env.close()

    report_path = args.root.parent / "sim_verify_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"wrote {report_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
