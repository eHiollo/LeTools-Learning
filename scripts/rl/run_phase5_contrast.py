#!/usr/bin/env python3
"""Phase-5 contrast: zero / random / ACT execute-first / SAC explore proxy.

Default = MockBackend (no ROS). Deterministic reward only (Robometer off).
Does not claim SAC beats ACT until a trained SAC + success metric exists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from kuavo_rl.act_runner import ActExecuteFirstRunner, ConstantChunkPolicy
from kuavo_rl.adapter import make_kuavo_hilserl_env
from kuavo_rl.config import ActRunnerConfig, build_env_config_from_dict, load_yaml
from kuavo_rl.contracts import ACTION_DIM
from kuavo_rl.recording import build_manifest, write_manifest


ARMS = ("zero", "random", "act", "sac")


def _summarize_steps(steps: list[dict[str, Any]], *, arm: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    rewards = [float(s["reward"]) for s in steps]
    clips = sum(1 for s in steps if bool((s.get("info") or {}).get("action_clipped")))
    last = steps[-1] if steps else {}
    info = last.get("info") or {}
    actions = [np.asarray(s["action"], dtype=np.float32).reshape(-1) for s in steps if "action" in s]
    mean_abs = float(np.mean([np.mean(np.abs(a)) for a in actions])) if actions else 0.0
    action_dims = {int(a.size) for a in actions}
    out = {
        "arm": arm,
        "steps": len(steps),
        "return": float(sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "success": bool(info.get("success", False)),
        "terminated": bool(last.get("terminated", False)),
        "truncated": bool(last.get("truncated", False)),
        "last_fault": info.get("fault_code"),
        "n_clips": clips,
        "mean_abs_action": mean_abs,
        "action_dims": sorted(action_dims),
        "action_dim_ok": action_dims == {ACTION_DIM} if action_dims else False,
    }
    if extra:
        out.update(extra)
    return out


def _run_step_policy(
    env,
    *,
    max_steps: int,
    policy_fn: Callable[[dict, int, np.random.Generator], np.ndarray],
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    obs, _info = env.reset(seed=seed)
    history: list[dict[str, Any]] = []
    for t in range(max_steps):
        action = np.asarray(policy_fn(obs, t, rng), dtype=np.float32).reshape(-1)
        next_obs, reward, terminated, truncated, info = env.step(action)
        history.append(
            {
                "action": action,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        )
        obs = next_obs
        if terminated or truncated:
            break
    return history


def _arm_zero(obs: dict, _t: int, _rng: np.random.Generator) -> np.ndarray:
    state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    # Hold current joint pose (true zero-delta / freeze).
    return state.copy()


def _arm_random(obs: dict, _t: int, rng: np.random.Generator) -> np.ndarray:
    state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    delta = rng.normal(0.0, 0.05, size=ACTION_DIM).astype(np.float32)
    delta[7] = delta[15] = 0.0
    return state + delta


def _arm_sac_explore(obs: dict, _t: int, rng: np.random.Generator) -> np.ndarray:
    """Untrained SAC-like explore: small Gaussian around state (single-step 16-D)."""
    state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    delta = rng.normal(0.0, 0.08, size=ACTION_DIM).astype(np.float32)
    delta[7] = delta[15] = float(rng.uniform(0.0, 0.2))
    return state + delta


def _run_act(env, *, max_steps: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Constant near-hold chunk: verifies execute-first contract without loading ACT weights.
    chunk = np.zeros((10, ACTION_DIM), dtype=np.float32)
    runner = ActExecuteFirstRunner(ConstantChunkPolicy(chunk), ActRunnerConfig(chunk_size=10, execute_steps=1))
    env.reset(seed=seed)
    # ActExecuteFirstRunner.reset internally; call run_episode after fresh env.
    result = runner.run_episode(env, max_steps=max_steps)
    steps = result["steps"]
    discarded = [int(s.get("discarded", -1)) for s in steps]
    extra = {
        "policy": "act_execute_first_harness",
        "chunk_size": 10,
        "discarded_per_step": discarded[0] if discarded else None,
        "execute_first_ok": bool(discarded) and all(d == 9 for d in discarded),
        "note": "Harness uses ConstantChunkPolicy; real ACT ckpt is attached via --refs if present.",
    }
    return steps, extra


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _verdict(arms: dict[str, Any], refs: dict[str, Any]) -> dict[str, Any]:
    dim_ok = all(bool(arms[a].get("action_dim_ok")) for a in ARMS if a in arms)
    act_ok = bool(arms.get("act", {}).get("execute_first_ok"))
    ran_all = len(arms) == 4 and all(int(arms[a].get("steps", 0)) > 0 for a in ARMS)
    harness_ok = bool(dim_ok and act_ok and ran_all)
    sac_vs_act = {
        "interpretable_improvement": False,
        "reason": (
            "No trained SAC success metric yet; SAC arm is explore proxy / prior sim smoke only. "
            "Handbook: do not enter real Stage-B until SAC shows interpretable gain vs ACT."
        ),
    }
    if refs.get("act_sim") and refs.get("sac_sim"):
        sac_vs_act["refs_attached"] = True
        sac_vs_act["act_sim_fault"] = refs["act_sim"].get("last_fault")
        sac_vs_act["sac_sim_status"] = refs["sac_sim"].get("status")
        sac_vs_act["sac_episode_rewards"] = refs["sac_sim"].get("episode_rewards")
    return {
        "harness_ok": harness_ok,
        "action_dim_ok": dim_ok,
        "act_execute_first_ok": act_ok,
        "sac_vs_act": sac_vs_act,
        "enter_real_stage_b": False,
        "recommendation": (
            "Contrast harness OK with deterministic reward. Keep Robometer offline. "
            "Next: longer Kuavo-Sim ACT eval + trained SAC before claiming SAC>ACT."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rl/kuavo_phase5_contrast.yaml"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rl_runs/phase5_contrast_latest/manifest.json"),
    )
    parser.add_argument(
        "--act-ref",
        type=Path,
        default=Path("data/rl_runs/act_kuavo_sim_eval/manifest.json"),
    )
    parser.add_argument(
        "--sac-ref",
        type=Path,
        default=Path("data/rl_runs/kuavo_sac_sim_latest/manifest.json"),
    )
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config.exists() else {"env": {}}
    cfg = build_env_config_from_dict(raw)
    # Force Robometer off for this contrast.
    cfg.reward.use_robometer = False
    cfg.reward.robometer_mode = "disabled"

    env = make_kuavo_hilserl_env(cfg, use_stub_robometer=True)
    arms: dict[str, Any] = {}
    try:
        zero_steps = _run_step_policy(env, max_steps=args.steps, policy_fn=_arm_zero, seed=args.seed)
        arms["zero"] = _summarize_steps(zero_steps, arm="zero", extra={"policy": "hold_state"})

        rand_steps = _run_step_policy(env, max_steps=args.steps, policy_fn=_arm_random, seed=args.seed + 1)
        arms["random"] = _summarize_steps(rand_steps, arm="random", extra={"policy": "gaussian_delta"})

        act_steps, act_extra = _run_act(env, max_steps=args.steps, seed=args.seed + 2)
        arms["act"] = _summarize_steps(act_steps, arm="act", extra=act_extra)

        sac_steps = _run_step_policy(env, max_steps=args.steps, policy_fn=_arm_sac_explore, seed=args.seed + 3)
        arms["sac"] = _summarize_steps(
            sac_steps,
            arm="sac",
            extra={
                "policy": "sac_explore_proxy",
                "note": "Not a trained SAC actor; single-step 16-D explore only.",
            },
        )
    finally:
        env.close()

    refs = {
        "act_sim": _load_json(args.act_ref),
        "sac_sim": _load_json(args.sac_ref),
    }
    verdict = _verdict(arms, refs)
    report = {
        "mode": "mock",
        "reward": "deterministic",
        "robometer": "disabled",
        "steps_requested": args.steps,
        "arms": arms,
        "refs": {
            "act_sim_path": str(args.act_ref) if refs["act_sim"] else None,
            "sac_sim_path": str(args.sac_ref) if refs["sac_sim"] else None,
            "act_sim": refs["act_sim"],
            "sac_sim": refs["sac_sim"],
        },
        "verdict": verdict,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    man = build_manifest("phase5_contrast", extra=report)
    write_manifest(args.manifest, man)
    summary = args.manifest.with_name("summary.md")
    lines = [
        "# Phase-5 contrast",
        "",
        f"- harness_ok: **{verdict['harness_ok']}**",
        f"- enter_real_stage_b: **{verdict['enter_real_stage_b']}**",
        f"- reward: deterministic / Robometer disabled",
        "",
        "| arm | steps | return | last_fault | action_dim_ok | notes |",
        "|---|---:|---:|---|---|---|",
    ]
    for name in ARMS:
        a = arms[name]
        note = a.get("policy", "")
        if name == "act":
            note += f" discard={a.get('discarded_per_step')}"
        lines.append(
            f"| {name} | {a['steps']} | {a['return']:.3f} | {a.get('last_fault')} | "
            f"{a.get('action_dim_ok')} | {note} |"
        )
    lines.extend(["", "## Recommendation", "", verdict["recommendation"], ""])
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.manifest}")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
