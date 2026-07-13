#!/usr/bin/env python3
"""Load Stage-A ACT checkpoint and verify execute-first (chunk=10 -> take step 0).

Default path uses MockBackend env (no ROS). For live Kuavo-Sim:
  - start Docker infer: bash scripts/rl/run_act_infer_server.sh
  - host: bash scripts/rl/run_act_kuavo_sim_eval.sh
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _mock_obs_from_dataset(root: Path) -> dict:
    """Build one observation matching dataset feature keys (for dry-run)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="lerobot_merged", root=str(root), video_backend="pyav")
    sample = ds[0]
    obs = {}
    for k, v in sample.items():
        if k.startswith("observation."):
            if torch.is_tensor(v):
                obs[k] = v.detach().cpu().numpy()
            else:
                obs[k] = v
    return obs


def _make_policy(args):
    if args.policy == "remote":
        from kuavo_rl.act_policy import RemoteActChunkPolicy

        policy = RemoteActChunkPolicy(host=args.infer_host, port=args.infer_port)
        policy.connect()
        return policy, "remote"

    from kuavo_rl.act_policy import LerobotActChunkPolicy

    ckpt = args.checkpoint
    if not ckpt.exists():
        alt = Path("data/rl_runs/act_stage_a_latest")
        candidates = list(alt.glob("checkpoints/*/pretrained_model")) if alt.exists() else []
        if not candidates:
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        ckpt = candidates[-1]
    print(f"[eval] loading ACT from {ckpt}")
    return LerobotActChunkPolicy.from_checkpoint(str(ckpt), device=args.device), str(ckpt)


def _make_kuavo_env(deploy_config: Path):
    import gymnasium as gym
    import kuavo_deploy.kuavo_env  # noqa: F401
    from kuavo_deploy.config import load_kuavo_config

    deploy_cfg = load_kuavo_config(str(deploy_config))
    if deploy_cfg.env.env_name != "Kuavo-Sim":
        raise RuntimeError(
            f"deploy config env_name={deploy_cfg.env.env_name!r}; expected Kuavo-Sim "
            f"(set inference_env: sim in {deploy_config})"
        )
    return gym.make(
        deploy_cfg.env.env_name,
        max_episode_steps=int(deploy_cfg.inference.max_episode_steps),
        config=deploy_cfg,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/rl_runs/checkpoints/005000/pretrained_model"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot/lerobot_merged"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kuavo-env", action="store_true")
    parser.add_argument(
        "--policy",
        choices=("local", "remote"),
        default="local",
        help="local=load ACT in-process (needs Py>=3.11 lerobot); remote=TCP infer server",
    )
    parser.add_argument("--infer-host", default="127.0.0.1")
    parser.add_argument("--infer-port", type=int, default=8765)
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=Path("configs/deploy/total/deploy_sim_smoke_cams_total.yaml"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rl/kuavo_hilserl_sim_act.yaml"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rl_runs/act_execute_first_eval/manifest.json"),
    )
    parser.add_argument(
        "--shadow",
        action="store_true",
        help="Predict + safety only; never publish policy actions",
    )
    args = parser.parse_args()

    from kuavo_rl.act_runner import ActExecuteFirstRunner
    from kuavo_rl.adapter import make_kuavo_hilserl_env
    from kuavo_rl.config import ActRunnerConfig, build_env_config_from_dict, load_yaml
    from kuavo_rl.recording import build_manifest, write_manifest

    policy, policy_id = _make_policy(args)

    # Dry-run chunk shape when local + dataset available; remote uses zero obs.
    chunk_shape = None
    if args.policy == "local":
        obs_ds = _mock_obs_from_dataset(args.dataset_root)
        chunk = policy.predict_action_chunk(obs_ds)
        chunk_shape = list(chunk.shape)
        print(f"[eval] dataset-obs chunk shape={chunk.shape} (expect (10, 16))")
        if chunk.ndim != 2 or chunk.shape[1] != 16:
            raise RuntimeError(f"unexpected chunk shape {chunk.shape}")
    else:
        from kuavo_rl.contracts import IMAGE_KEYS, IMAGE_SHAPE_CHW

        warm = {"observation.state": np.zeros(16, dtype=np.float32)}
        for k in IMAGE_KEYS:
            warm[k] = np.zeros(IMAGE_SHAPE_CHW, dtype=np.float32)
        chunk = policy.predict_action_chunk(warm)
        chunk_shape = list(chunk.shape)
        print(f"[eval] remote warmup chunk shape={chunk.shape}")

    raw = load_yaml(args.config) if args.config.exists() else {"env": {}}
    cfg = build_env_config_from_dict(raw)
    if args.shadow:
        cfg.shadow_mode = True

    kuavo_gym_env = None
    mode = "mock"
    if args.kuavo_env:
        try:
            kuavo_gym_env = _make_kuavo_env(args.deploy_config)
            mode = "kuavo_sim"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Kuavo-Sim unavailable ({exc}); using mock")
            mode = "mock_fallback"

    env = make_kuavo_hilserl_env(cfg, kuavo_gym_env=kuavo_gym_env, use_stub_robometer=True)
    runner = ActExecuteFirstRunner(policy, ActRunnerConfig(chunk_size=10, execute_steps=1))

    summary = {
        "mode": mode,
        "policy": args.policy,
        "policy_id": policy_id,
        "checkpoint": str(args.checkpoint),
        "chunk_shape": chunk_shape,
        "discarded_tail_len": int(chunk.shape[0] - 1),
        "execute_first_ok": True,
        "shadow_mode": bool(cfg.shadow_mode),
    }

    if mode.startswith("kuavo"):
        result = runner.run_episode(env, max_steps=args.steps)
        summary["steps"] = result["n"]
        summary["discarded_per_step"] = result["steps"][0]["discarded"] if result["steps"] else None
        if result["steps"]:
            summary["last_fault"] = result["steps"][-1]["info"].get("fault_code")
            summary["last_reward"] = result["steps"][-1]["reward"]
    else:
        # Mock: validate select_action contract with synthetic/training-like obs
        from kuavo_rl.contracts import IMAGE_KEYS, IMAGE_SHAPE_CHW

        if args.policy == "local":
            step_obs = obs_ds  # noqa: F821 — set in local branch above
        else:
            step_obs = {"observation.state": np.zeros(16, dtype=np.float32)}
            for k in IMAGE_KEYS:
                step_obs[k] = np.zeros(IMAGE_SHAPE_CHW, dtype=np.float32)
        step = runner.select_action(step_obs)
        summary["executed_index"] = step.executed_index
        summary["discarded_tail_len"] = int(step.discarded_tail.shape[0])
        summary["action_dim"] = int(step.action.shape[0])
        assert step.executed_index == 0
        assert step.discarded_tail.shape[0] == chunk.shape[0] - 1
        assert len(runner.pending_queue) == 0

    env.close()
    if hasattr(policy, "close"):
        policy.close()
    man = build_manifest("act_execute_first_eval", extra=summary)
    write_manifest(args.manifest, man)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
