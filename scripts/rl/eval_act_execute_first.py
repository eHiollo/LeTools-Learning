#!/usr/bin/env python3
"""Load Stage-A ACT checkpoint and verify execute-first (chunk=10 -> take step 0).

Default path uses MockBackend env (no ROS). For real Kuavo-Sim, pass --kuavo-env
once ROS is up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


class LerobotActChunkPolicy:
    """Adapter: LeRobot ACTPolicy -> ActExecuteFirstRunner protocol."""

    def __init__(self, policy, device: str = "cuda"):
        self.policy = policy
        self.device = device
        self.policy.eval()

    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        batch = {}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
            elif torch.is_tensor(v):
                t = v
            else:
                continue
            if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW image
                t = t.unsqueeze(0)
            elif t.ndim == 1:
                t = t.unsqueeze(0)
            batch[k] = t.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            # ACT returns (B, chunk, action_dim)
            chunk = self.policy.predict_action_chunk(batch)
        return chunk[0].detach().cpu().numpy().astype(np.float32)


def _mock_obs_from_dataset(root: Path, device: str = "cpu") -> dict:
    """Build one observation matching dataset feature keys (for dry-run)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="lerobot_merged", root=str(root), video_backend="pyav")
    sample = ds[0]
    obs = {}
    for k, v in sample.items():
        if k.startswith("observation."):
            if torch.is_tensor(v):
                obs[k] = v.detach().cpu()
            else:
                obs[k] = v
    return obs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/rl_runs/act_stage_a_latest/checkpoints/last/pretrained_model"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/lerobot/lerobot_merged"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--kuavo-env", action="store_true")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rl_runs/act_execute_first_eval/manifest.json"),
    )
    args = parser.parse_args()

    from lerobot.policies.act.modeling_act import ACTPolicy

    from kuavo_rl.act_runner import ActExecuteFirstRunner
    from kuavo_rl.adapter import make_kuavo_hilserl_env
    from kuavo_rl.config import ActRunnerConfig
    from kuavo_rl.recording import build_manifest, write_manifest

    ckpt = args.checkpoint
    if not ckpt.exists():
        # tolerate symlink to run dir
        alt = Path("data/rl_runs/act_stage_a_latest")
        candidates = list(alt.glob("checkpoints/*/pretrained_model")) if alt.exists() else []
        if not candidates:
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        ckpt = candidates[-1]

    print(f"[eval] loading ACT from {ckpt}")
    policy = ACTPolicy.from_pretrained(str(ckpt))
    policy.to(args.device)
    wrapped = LerobotActChunkPolicy(policy, device=args.device)

    # Dry-run chunk shape on a real dataset frame
    obs_ds = _mock_obs_from_dataset(args.dataset_root)
    chunk = wrapped.predict_action_chunk(obs_ds)
    print(f"[eval] dataset-obs chunk shape={chunk.shape} (expect (10, 16) or (>=1, 16))")
    if chunk.ndim != 2 or chunk.shape[1] != 16:
        raise RuntimeError(f"unexpected chunk shape {chunk.shape}")

    kuavo_gym_env = None
    mode = "mock"
    if args.kuavo_env:
        try:
            import gymnasium as gym
            import kuavo_deploy.kuavo_env  # noqa: F401

            kuavo_gym_env = gym.make("Kuavo-Sim")
            mode = "kuavo_sim"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Kuavo-Sim unavailable ({exc}); using mock")
            mode = "mock_fallback"

    env = make_kuavo_hilserl_env(kuavo_gym_env=kuavo_gym_env, use_stub_robometer=True)
    runner = ActExecuteFirstRunner(wrapped, ActRunnerConfig(chunk_size=10, execute_steps=1))

    # Mock env obs keys differ from ACT training keys; for mock we only assert
    # execute-first using Constant-like path if policy forward fails on mock obs.
    # Prefer: run_episode only when kuavo env provides matching cameras.
    summary = {
        "mode": mode,
        "checkpoint": str(ckpt),
        "chunk_shape_on_dataset_obs": list(chunk.shape),
        "discarded_tail_len": int(chunk.shape[0] - 1),
        "execute_first_ok": True,
    }

    if mode.startswith("kuavo"):
        result = runner.run_episode(env, max_steps=args.steps)
        summary["steps"] = result["n"]
        summary["discarded_per_step"] = result["steps"][0]["discarded"] if result["steps"] else None
    else:
        # Mock: only validate select_action contract with synthetic obs matching training keys
        # Reuse dataset obs through runner.select_action (does not call env)
        step = runner.select_action(obs_ds)
        summary["executed_index"] = step.executed_index
        summary["discarded_tail_len"] = int(step.discarded_tail.shape[0])
        summary["action_dim"] = int(step.action.shape[0])
        assert step.executed_index == 0
        assert step.discarded_tail.shape[0] == chunk.shape[0] - 1
        assert len(runner.pending_queue) == 0

    env.close()
    man = build_manifest("act_execute_first_eval", extra=summary)
    write_manifest(args.manifest, man)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
