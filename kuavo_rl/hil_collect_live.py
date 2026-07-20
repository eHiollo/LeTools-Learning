"""Live VR collection — LeRobot-style long-running session.

Mirrors ``lerobot-record`` UX:
  - one process stays up across many episodes
  - RESET (teleop ok, no write) → RECORD → save → RESET …
  - clear phase lines (not a JSON dump every step)
  - Y+stick: → early end / start, ← rerecord, ↓ end session
"""

from __future__ import annotations

import logging
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kuavo_rl.act_runner import ActExecuteFirstRunner
from kuavo_rl.config import ActRunnerConfig
from kuavo_rl.hil_collection import HILCollectionOrchestrator, make_episode_id
from kuavo_rl.hil_recording.models import (
    FINALIZED_OK,
    PHASE_COLLECTION_ENDED,
    PHASE_RECORDING,
    PHASE_RESETTING,
    RecordRequest,
    ResultEvent,
    REVIEW_READY_MARKER,
)
from kuavo_rl.hil_recording.timebase import now_stamps
from kuavo_rl.quest_episode_control import (
    ModifierStickDetector,
    QuestEpisodeControlEventSource,
    StickEdgeDetector,
    load_stick_calibration,
)
from kuavo_rl.recording import HILReplayWriter, TransitionRecord


def _say(msg: str) -> None:
    """LeRobot ``log_say`` equivalent — one clear line to the operator."""
    print(f"[collect] {msg}", flush=True)


# Soft safety only — episodes must end with B (no normal step/time cutoff).
VR_SAFETY_MAX_STEPS = 100_000
VR_SAFETY_MAX_DURATION_S = 86_400.0


def _print_controls() -> None:
    _say("========== VR collect (LeRobot-style session) ==========")
    _say("Hold grip          → move arms (Quest IK)")
    _say("Hold Y + stick →   → start recording (RESET only)")
    _say("Hold Y + stick ←   → rerecord (discard current)")
    _say("Hold Y + stick ↓   → end session (only in RESET; finish ep with B first)")
    _say("B click            → SUCCESS + end → accepted_replay (train-ready)")
    _say("B double-click     → FAILURE + end → accepted_replay (train-ready)")
    _say("B hold ≥1.2s       → ABORT → quarantine (not train-ready)")
    _say("No step timeout    → every kept episode must end with B")
    _say("Solo mode          → no offline label/review; B is the label")
    _say("Ctrl-C             → abort current; twice → force quit")
    _say("=======================================================")


class HoldStatePolicy:
    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
        return state.reshape(1, -1)


@dataclass
class LiveCollectResult:
    status: str
    episode_id: str
    steps: int
    stop_reason: str
    session_state: str | None = None
    export: dict[str, Any] | None = None
    pending_review: str | None = None
    accepted_replay: str | None = None
    lerobot_dir: str | None = None
    review_ready: bool = False
    train_ready: bool = False

    def summary_line(self) -> str:
        path = (
            self.lerobot_dir
            or self.accepted_replay
            or self.pending_review
            or (self.export or {}).get("path")
            or "-"
        )
        return (
            f"{self.status}  eid={self.episode_id}  steps={self.steps}  "
            f"reason={self.stop_reason}  path={path}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "episode_id": self.episode_id,
            "steps": self.steps,
            "stop_reason": self.stop_reason,
            "session_state": self.session_state,
            "export": self.export,
            "pending_review": self.pending_review,
            "accepted_replay": self.accepted_replay,
            "lerobot_dir": self.lerobot_dir,
            "review_ready": self.review_ready,
            "train_ready": self.train_ready,
        }


def _make_kuavo_gym(deploy_config: Path, *, max_episode_steps: int):
    import gymnasium as gym
    import kuavo_deploy.kuavo_env  # noqa: F401
    from kuavo_deploy.config import load_kuavo_config

    deploy_cfg = load_kuavo_config(str(deploy_config))
    if deploy_cfg.env.env_name != "Kuavo-Sim":
        raise RuntimeError(
            f"deploy env_name={deploy_cfg.env.env_name!r}; expected Kuavo-Sim"
        )
    return gym.make(
        deploy_cfg.env.env_name,
        max_episode_steps=int(max_episode_steps),
        config=deploy_cfg,
    )


def _quiet_robot_logs() -> None:
    """Silence deploy/SDK chatter so only ``[collect]`` lines stay useful.

    Must run *after* gym/env init: LoggerManager.get_logger() resets level to INFO.
    """
    for name in ("robot", "env", "model", "kuavo_deploy", "kuavo-humanoid-sdk"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.ERROR)
        for h in lg.handlers:
            h.setLevel(logging.ERROR)

    # SDK LoggerClient prints to stdout when log-server (ws://localhost:8889) is down
    try:
        from kuavo_humanoid_sdk.kuavo import logger_client as _lc

        _lc.LoggerClient._print_log = lambda self, log_data: None  # noqa: ARG005
    except Exception:  # noqa: BLE001
        pass

    try:
        from kuavo_humanoid_sdk.common.logger import SDKLogger

        SDKLogger.setLevel(logging.ERROR)
        for h in SDKLogger.handlers:
            h.setLevel(logging.ERROR)
    except Exception:  # noqa: BLE001
        pass

    # gymnasium Box dtype warnings
    logging.getLogger("gymnasium").setLevel(logging.ERROR)


class VrCollectRuntime:
    """Owns gym/teleop/event source for the whole multi-episode session."""

    def __init__(
        self,
        orch: HILCollectionOrchestrator,
        *,
        deploy_config: Path,
        env_config: Path,
        max_steps: int,
        enable_quest_episode_control: bool = True,
    ):
        from kuavo_rl.adapter import make_kuavo_hilserl_env
        from kuavo_rl.config import build_env_config_from_dict, load_yaml
        from kuavo_rl.ros_teleop import RosTeleopAdapter, RosTeleopConfig

        self.orch = orch
        self.cfg = orch.config
        # max_steps arg is a soft safety ceiling only (default huge); B ends episodes.
        self.max_steps = int(max_steps) if max_steps is not None else VR_SAFETY_MAX_STEPS
        self.max_steps = max(self.max_steps, 1)
        self.runner = ActExecuteFirstRunner(
            HoldStatePolicy(), ActRunnerConfig(chunk_size=1, execute_steps=1)
        )

        raw = load_yaml(env_config) if env_config.exists() else {"env": {}}
        env_cfg = build_env_config_from_dict(raw)
        env_cfg.shadow_mode = True
        if getattr(env_cfg, "episode", None) is not None:
            env_cfg.episode.max_steps = self.max_steps
            env_cfg.episode.max_duration_s = VR_SAFETY_MAX_DURATION_S
        # VR teleop often soft-clips every step (delta/joint bounds). Config default
        # max_consecutive_clips=3 would end at step 3 — disable for B-only collect.
        if getattr(env_cfg, "safety", None) is not None:
            env_cfg.safety.max_consecutive_clips = 0

        _quiet_robot_logs()
        _say("Connecting Kuavo-Sim + Quest teleop (one-time setup)…")
        _say("Episode end = B only (no step/time / consecutive-clip cutoff)")
        kuavo_gym = _make_kuavo_gym(deploy_config, max_episode_steps=self.max_steps)
        teleop_raw = raw.get("teleop", {}) if isinstance(raw, dict) else {}
        allowed = {
            k: teleop_raw[k]
            for k in RosTeleopConfig.__dataclass_fields__
            if k in teleop_raw
        }
        self.teleop = RosTeleopAdapter(RosTeleopConfig(**allowed))
        self.teleop.start()
        self.env = make_kuavo_hilserl_env(
            env_cfg,
            kuavo_gym_env=kuavo_gym,
            use_stub_robometer=True,
            teleop=self.teleop,
        )

        self.event_src: QuestEpisodeControlEventSource | None = None
        if enable_quest_episode_control and self.cfg.episode_control.startswith("quest_"):
            cal = load_stick_calibration(self.cfg.stick_calibration_path)
            mod = ModifierStickDetector(
                stick=StickEdgeDetector(
                    trigger_threshold=self.cfg.right_stick_trigger_threshold,
                    rearm_neutral_threshold=self.cfg.right_stick_rearm_neutral_threshold,
                    debounce_s=self.cfg.right_stick_debounce_s,
                    calibration=cal,
                )
            )
            self.event_src = QuestEpisodeControlEventSource(
                mode=self.cfg.episode_control,
                mod_stick=mod,
                calibration=cal,
            )
            try:
                self.event_src.start()
            except Exception as exc:  # noqa: BLE001
                _say(f"WARN: Quest episode control unavailable ({exc})")
                self.event_src = None

        self._force_quit = False
        self._ctrl_c = 0
        # LoggerManager re-enables INFO during gym.make — quiet again for operator UX
        _quiet_robot_logs()

    def close(self) -> None:
        if self.event_src is not None:
            try:
                self.event_src.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self.teleop.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass

    def poll_control(self) -> str | None:
        if self.event_src is None:
            return None
        ev = self.event_src.poll()
        return None if ev is None else ev.event_type

    def idle_reset(
        self,
        *,
        timeout_s: float,
        wait_for_start: bool = True,
    ) -> str:
        """RESET phase: teleop OK, no HIL write. Returns control event or timeout/abort."""
        self.orch.phase.phase = PHASE_RESETTING
        deadline = time.monotonic() + float(timeout_s)
        obs, _ = self.env.reset()
        while time.monotonic() < deadline:
            if self._force_quit:
                return "ctrl_c_abort"
            # Keep VR path warm without recording.
            result = self.runner.select_action(obs)
            obs, _, term, trunc, info = self.env.step(result.action)
            ev = self.poll_control()
            if ev == "right_stick_right" and wait_for_start:
                return "right_stick_right"
            if ev == "right_stick_down":
                return "right_stick_down"
            if term or trunc:
                obs, _ = self.env.reset()
            time.sleep(0.02)
        return "timeout"

    def record_one(
        self,
        *,
        operator: str,
        scene_id: str | None,
        live_rosbag: bool,
        episode_index: int,
    ) -> LiveCollectResult:
        orch = self.orch
        cfg = self.cfg
        eid = make_episode_id(cfg.task_id)
        use_live_bag = bool(live_rosbag)

        orch.recording.dry_run_recorder = not use_live_bag
        orch.recording.skip_gate_ros = not use_live_bag

        metadata = {
            "operator": operator,
            "scene_id": scene_id or cfg.scene_id,
            "mode": "vr_only",
            "collection": "live_vr_sim",
            "episode_index": episode_index,
            "shadow_mode": True,
            "episode_control": cfg.episode_control,
        }
        req = RecordRequest(
            episode_id=eid,
            task_id=cfg.task_id,
            control_profile="act_vr",
            robot_type=cfg.robot_type,
            robot_version=cfg.robot_version,
            eef_type=cfg.eef_type,
            lower_commit=cfg.lower_commit,
            dry_run=not use_live_bag,
            skip_gate_ros=not use_live_bag,
            post_roll_s=float(cfg.post_roll_s),
            metadata=metadata,
        )
        orch.session.create(req)
        orch.session.register_producer("vr_collect", os.getpid(), "teleop")
        orch.session.start(eid)
        orch.phase.phase = PHASE_RECORDING
        orch.phase.episode_id = eid
        orch.index.append_event("episode_started", {"episode_id": eid, "mode": "vr_only"})

        staging = orch.recording.staging_dir(eid)
        writer = HILReplayWriter(cfg.root, "vr_only", staging_dir=staging)

        stop_reason = "timeout"
        result_type = "success"
        n_steps = 0

        def on_step(step: dict) -> None:
            nonlocal stop_reason, result_type, n_steps
            if self._force_quit:
                stop_reason, result_type = "ctrl_c_abort", "abort"
                raise RuntimeError("ctrl_c_abort")
            n_steps = int(step["step_id"]) + 1
            info = {**step["info"], "episode_id": eid}
            audit = info.get("action_audit", {})
            executed = step.get("executed_action") or audit.get("raw_action")
            policy_action = np.asarray(step["policy_action"], dtype=np.float32).tolist()
            stamps = now_stamps()
            # Keep schema fixed: missing teleop fields → zeros (not null / omit).
            act_dim = len(policy_action)
            mask = info.get("intervention_mask")
            if mask is None:
                mask = [False] * act_dim
            else:
                mask = np.asarray(mask, dtype=bool).reshape(-1).tolist()
                if len(mask) < act_dim:
                    mask = mask + [False] * (act_dim - len(mask))
                mask = mask[:act_dim]
            if executed is None:
                executed = [0.0] * act_dim
            else:
                executed = np.asarray(executed, dtype=np.float32).reshape(-1).tolist()
                if len(executed) < act_dim:
                    executed = executed + [0.0] * (act_dim - len(executed))
                executed = executed[:act_dim]
            rec = TransitionRecord(
                experiment_id="vr_only",
                episode_id=eid,
                step_id=int(step["step_id"]),
                timestamp=float(stamps.wall_time_ns) / 1e9,
                action=list(executed),
                reward=float(step["reward"]),
                reward_source=str(info.get("reward_source", "unknown")),
                terminated=bool(step["terminated"]),
                truncated=bool(step["truncated"]),
                fault_code=str(info.get("fault_code", "UNKNOWN")),
                is_intervention=bool(info.get("is_intervention", False)),
                action_clipped=bool(info.get("action_clipped", False)),
                extras={
                    "policy_action": policy_action,
                    "executed_action": executed,
                    "teleop_source": info.get("teleop_source", "none"),
                    "intervention_mask": mask,
                    "intervention_segment_id": info.get("intervention_segment_id") or 0,
                    "intervention_segment_step": int(
                        info.get("intervention_segment_step") or 0
                    ),
                    "stamps": stamps.to_dict(),
                    "mode": "vr_only",
                },
            )
            writer.log_transition(
                rec,
                observation=step["observation"],
                next_observation=step["next_observation"],
            )
            orch.session.update_transition(
                {
                    "step_id": int(step["step_id"]),
                    "stamps": stamps,
                    "policy_action": policy_action,
                    "executed_action": executed,
                    "intervention_mask": info.get("intervention_mask"),
                    "intervention_segment_id": info.get("intervention_segment_id"),
                    "intervention_segment_step": int(
                        info.get("intervention_segment_step") or 0
                    ),
                    "reward": float(step["reward"]),
                    "fault_code": str(info.get("fault_code", "UNKNOWN")),
                }
            )
            te = info.get("teleop_events") or {}
            if te.get("success"):
                orch.set_operator_hint(
                    eid, "success_candidate", stop_reason="success_button", actor=operator
                )
                stop_reason, result_type = "success_button", "success"
                _say("B → success — ending episode")
                raise RuntimeError("teleop_stop:success_button")
            if te.get("failure"):
                orch.set_operator_hint(
                    eid, "failure_candidate", stop_reason="failure_button", actor=operator
                )
                stop_reason, result_type = "failure_button", "failure"
                _say("B×2 → failure — ending episode")
                raise RuntimeError("teleop_stop:failure_button")
            if te.get("abort") or te.get("stop"):
                stop_reason = "abort" if te.get("abort") else "estop"
                result_type = stop_reason
                _say(f"teleop stop → {stop_reason} — ending episode")
                raise RuntimeError(f"teleop_stop:{stop_reason}")

            ev = self.poll_control()
            if ev == "right_stick_left":
                stop_reason, result_type = "rerecord", "abort"
                raise RuntimeError("episode_control:rerecord")
            if ev == "right_stick_right":
                # No unlabeled early-end: every kept episode must end with B.
                _say("Y+→ ignored while recording — end with B (success/fail/abort)")
            if ev == "right_stick_down":
                _say("Y+↓ ignored while recording — end episode with B first, then ↓ in RESET")
            stop_req = orch.session.poll_stop_request()
            if stop_req is not None:
                stop_reason, result_type = stop_req.reason, "fault"
                raise RuntimeError(f"watchdog_stop:{stop_req.reason}")

            # Env truncated/terminated without B (clips, fault, …) — not a labeled end.
            if step.get("truncated") or step.get("terminated"):
                fault = str(info.get("fault_code", "UNKNOWN"))
                src = str(info.get("reward_source", ""))
                stop_reason = f"env_end:{fault}"
                result_type = "abort"
                _say(
                    f"Episode stopped by env at step {n_steps} "
                    f"(fault={fault}, source={src}) — no B label → quarantine"
                )
                raise RuntimeError(f"env_end:{fault}")

            if n_steps % 50 == 0:
                _say(f"  … recording step {n_steps}  (end with B)")

        try:
            self.runner.run_episode(self.env, max_steps=self.max_steps, on_step=on_step)
            # Loop hit VR_SAFETY_MAX_STEPS without B / rerecord.
            if stop_reason == "timeout":
                stop_reason, result_type = "unlabeled_safety_limit", "abort"
                _say(
                    f"WARN: hit safety ceiling ({self.max_steps} steps) without B — quarantining"
                )
        except RuntimeError:
            pass
        finally:
            writer.close()

        label = orch.db.get_label(eid)
        # Keep B-set hint; only fill unknown when nothing was labeled.
        hint = (label.operator_label_hint if label else None) or "unknown"
        orch.set_operator_hint(eid, hint, stop_reason=stop_reason)
        orch.session.record_event(
            ResultEvent(
                episode_id=eid,
                event_type=result_type,
                source="collect_vr_session",
                stamps=now_stamps(),
                payload={"stop_reason": stop_reason, "steps": n_steps},
            )
        )
        orch.session.request_stop(eid, stop_reason)
        snap = orch.session.wait_finalized(eid, timeout_s=90.0)

        if stop_reason in {"rerecord", "unlabeled_safety_limit", "ctrl_c_abort"} or str(
            stop_reason
        ).startswith("env_end:"):
            from kuavo_rl.hil_recording.publish_replay import quarantine_episode

            q = quarantine_episode(orch.recording, orch.db, eid, reason=stop_reason)
            orch.index.append_event("export_transition", q.to_dict())
            orch.phase.phase = PHASE_RESETTING
            return LiveCollectResult(
                status=f"quarantined_{stop_reason}",
                episode_id=eid,
                steps=n_steps,
                stop_reason=stop_reason,
                session_state=snap.session_state,
                export=q.to_dict(),
            )

        if snap.session_state not in FINALIZED_OK:
            from kuavo_rl.hil_recording.publish_replay import quarantine_episode

            # Quality finalize already quarantined; refresh reason only (no data wipe).
            q = quarantine_episode(
                orch.recording, orch.db, eid, reason=f"quality:{snap.quality_status}"
            )
            return LiveCollectResult(
                status="quarantined_quality",
                episode_id=eid,
                steps=n_steps,
                stop_reason=stop_reason,
                session_state=snap.session_state,
                export=q.to_dict(),
            )

        if getattr(orch.config, "auto_accept", True):
            pub = orch.accept_after_collect(
                eid, operator=operator, stop_reason=stop_reason
            )
            accepted = orch.recording.accepted_replay_dir / eid
            train_ready = (accepted / "TRAIN_READY").exists()
            status = "accepted_replay" if train_ready else str(pub.get("status", "export"))
            if status == "Quarantined" or pub.get("status") == "quarantined":
                status = "quarantined"
            # Per-episode: only stage bag into batch_bags/. Convert once at session end.
            if train_ready and getattr(orch.config, "auto_export_lerobot", False):
                from kuavo_rl.brain_lerobot_export import stage_bag_for_batch

                dest, stage_why = stage_bag_for_batch(
                    root=Path(orch.config.root), episode_id=eid
                )
                pub = {
                    **(pub or {}),
                    "lerobot_staged": str(dest) if dest else None,
                    "lerobot_stage_notes": stage_why,
                }
                if dest is not None:
                    _say(f"Staged bag for batch LeRobot export → {dest.name}")
                else:
                    _say(
                        f"WARN: could not stage bag for LeRobot "
                        f"({'; '.join(stage_why[:2])})"
                    )
            return LiveCollectResult(
                status=status,
                episode_id=eid,
                steps=n_steps,
                stop_reason=stop_reason,
                session_state=snap.session_state,
                export=pub,
                accepted_replay=str(accepted) if train_ready else None,
                lerobot_dir=None,
                train_ready=train_ready,
            )

        pub = orch.publish_pending(eid)
        pending = orch.recording.pending_review_dir / eid
        return LiveCollectResult(
            status="pending_review",
            episode_id=eid,
            steps=n_steps,
            stop_reason=stop_reason,
            session_state=snap.session_state,
            export=pub,
            pending_review=str(pending),
            review_ready=(pending / REVIEW_READY_MARKER).exists(),
        )


def collect_vr_session(
    orch: HILCollectionOrchestrator,
    *,
    deploy_config: Path,
    env_config: Path,
    operator: str = "vr_demo",
    scene_id: str | None = None,
    max_steps: int | None = None,
    num_episodes: int = 50,
    reset_time_s: float | None = None,
    live_rosbag: bool | None = None,
    enable_quest_episode_control: bool = True,
    single_episode: bool = False,
) -> dict[str, Any]:
    """LeRobot-style multi-episode VR collect. Process stays up until ↓ / Ctrl-C / N eps."""
    _quiet_robot_logs()
    _print_controls()

    pf = orch.preflight(for_live_collect=False)
    if pf["status"] == "Block":
        _say("preflight Block — abort")
        return {"status": "Block", "preflight": pf}

    # Episodes end only via B; CLI --max-steps is ignored for cutoff (safety ceiling only).
    steps = VR_SAFETY_MAX_STEPS
    if max_steps is not None and int(max_steps) > VR_SAFETY_MAX_STEPS:
        steps = int(max_steps)
    reset_s = float(
        reset_time_s if reset_time_s is not None else orch.config.reset_time_s
    )
    use_live = bool(orch.config.live_rosbag if live_rosbag is None else live_rosbag)
    n_target = 1 if single_episode else max(1, int(num_episodes))

    cid = f"{time.strftime('%Y%m%d')}_{orch.config.task_id}_vr"
    orch.ensure_collection_manifest(
        collection_id=cid, operator=operator, scene_id=scene_id
    )

    rt = VrCollectRuntime(
        orch,
        deploy_config=deploy_config,
        env_config=env_config,
        max_steps=steps,
        enable_quest_episode_control=enable_quest_episode_control,
    )

    results: list[dict[str, Any]] = []
    completed = 0
    stop_session = False

    def _on_sigint(signum, frame):  # noqa: ARG001
        rt._ctrl_c += 1
        if rt._ctrl_c == 1:
            _say("Ctrl-C → finish current phase (press again to force quit)")
            rt._force_quit = True
        else:
            raise SystemExit(130)

    prev = signal.signal(signal.SIGINT, _on_sigint)
    try:
        # First RESET: wait for operator to tip Y+→ before episode 1
        while completed < n_target and not stop_session:
            _say(
                f"Reset the environment  "
                f"(episode {completed + 1}/{n_target})  "
                f"Y+→ start · Y+↓ end session · reset_timeout={reset_s:.0f}s"
            )
            ev = rt.idle_reset(timeout_s=reset_s, wait_for_start=True)
            if ev == "right_stick_down" or ev == "ctrl_c_abort":
                _say("End collection (no more episodes)")
                break
            if ev == "timeout":
                _say("Reset timeout — tip Y+→ to start, or Y+↓ to quit")
                # keep waiting in a soft loop until start or quit
                while True:
                    ev2 = rt.idle_reset(timeout_s=reset_s, wait_for_start=True)
                    if ev2 == "right_stick_right":
                        ev = ev2
                        break
                    if ev2 in {"right_stick_down", "ctrl_c_abort"}:
                        stop_session = True
                        break
                if stop_session:
                    break

            _say(f"Recording episode {completed + 1}/{n_target}")
            out = rt.record_one(
                operator=operator,
                scene_id=scene_id,
                live_rosbag=use_live,
                episode_index=completed + 1,
            )
            results.append(out.to_dict())
            _say(out.summary_line())

            if out.stop_reason == "collection_complete":
                _say("Collection complete (Y+↓)")
                break
            if out.stop_reason == "rerecord":
                _say("Re-record episode — back to reset (same slot, not counted)")
                continue
            if out.status == "Block":
                break

            completed += 1
            if single_episode or completed >= n_target:
                break
            # brief RESET after save (like lerobot: teleop, no write)
            if completed < n_target:
                _say("Reset the environment (place objects; Y+→ next)")
                # consume remaining reset window until start or end
                # loop continues at top which waits for start again
    finally:
        signal.signal(signal.SIGINT, prev)
        _say("Stop recording — releasing robot / teleop")
        rt.close()
        orch.phase.phase = PHASE_COLLECTION_ENDED
        orch.index.append_event(
            "collection_ended",
            {"completed": completed, "attempts": len(results)},
        )
        orch.index.rebuild_index(orch.db)

    lerobot_export: dict[str, Any] | None = None
    if getattr(orch.config, "auto_export_lerobot", False) and completed > 0:
        from kuavo_rl.brain_lerobot_export import export_batch_to_lerobot

        _say("Batch LeRobot export (all TRAIN_READY bags → one dataset)…")
        exp = export_batch_to_lerobot(
            root=Path(orch.config.root),
            resync_from_accepted=True,
            topic_profile=str(getattr(orch.config, "lerobot_topic_profile", "sim")),
            task_description=orch.config.task_text or orch.config.task_id,
        )
        lerobot_export = exp.to_dict()
        if exp.status == "ok":
            _say(
                f"Brain CvtRosbag2Lerobot batch OK  bags={exp.bag_count}  "
                f"→ {exp.lerobot_dir}"
            )
        else:
            _say(
                f"WARN: batch LeRobot export failed "
                f"({'; '.join(exp.reasons[:2])})"
            )

    _say(f"Done. completed={completed}  attempts={len(results)}  root={orch.config.root}")
    return {
        "status": "ok",
        "completed_episodes": completed,
        "attempts": len(results),
        "results": results,
        "root": str(orch.config.root),
        "lerobot_export": lerobot_export,
        "lerobot_dir": (lerobot_export or {}).get("lerobot_dir"),
    }


# Backward-compatible alias used by older CLI wiring
def collect_vr_only_sim(
    orch: HILCollectionOrchestrator,
    **kwargs: Any,
) -> LiveCollectResult:
    """Single-episode wrapper around the session loop."""
    out = collect_vr_session(orch, single_episode=True, num_episodes=1, **kwargs)
    if out.get("status") == "Block":
        return LiveCollectResult(
            status="Block", episode_id="", steps=0, stop_reason="preflight_block", export=out
        )
    rows = out.get("results") or []
    if not rows:
        return LiveCollectResult(
            status="ended", episode_id="", steps=0, stop_reason="no_episode"
        )
    r = rows[-1]
    return LiveCollectResult(
        status=r["status"],
        episode_id=r["episode_id"],
        steps=r["steps"],
        stop_reason=r["stop_reason"],
        session_state=r.get("session_state"),
        export=r.get("export"),
        pending_review=r.get("pending_review"),
        review_ready=bool(r.get("review_ready")),
    )
