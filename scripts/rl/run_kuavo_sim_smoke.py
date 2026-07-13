#!/usr/bin/env python3
"""
Kuavo HIL-SERL smoke runner.

Default: MockBackend (no ROS).
With --kuavo-env: wrap an already importable Kuavo Gym env (requires ROS/SDK).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kuavo_rl.act_runner import ActExecuteFirstRunner, ConstantChunkPolicy
from kuavo_rl.adapter import make_kuavo_hilserl_env
from kuavo_rl.config import ActRunnerConfig, build_env_config_from_dict, load_yaml
from kuavo_rl.recording import build_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rl/kuavo_hilserl_sim.yaml"))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument(
        "--kuavo-env",
        action="store_true",
        help="Try to construct Kuavo-Sim via gymnasium.make (needs ROS)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/rl_runs/kuavo_sim_smoke/manifest.json"),
    )
    args = parser.parse_args()

    raw = load_yaml(args.config) if args.config.exists() else {"env": {}}
    cfg = build_env_config_from_dict(raw)
    if args.shadow:
        cfg.shadow_mode = True

    kuavo_gym_env = None
    mode = "mock"
    if args.kuavo_env:
        try:
            import gymnasium as gym
            import kuavo_deploy.kuavo_env  # noqa: F401  registers Kuavo-Sim

            kuavo_gym_env = gym.make("Kuavo-Sim")
            mode = "kuavo_sim"
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to make Kuavo-Sim ({exc}); falling back to MockBackend")
            kuavo_gym_env = None
            mode = "mock_fallback"

    env = make_kuavo_hilserl_env(cfg, kuavo_gym_env=kuavo_gym_env, use_stub_robometer=True)
    # Stage-A style execute-first smoke even for sim wiring check
    chunk = np.zeros((10, 16), dtype=np.float32)
    runner = ActExecuteFirstRunner(ConstantChunkPolicy(chunk), ActRunnerConfig())
    result = runner.run_episode(env, max_steps=args.steps)
    summary = {
        "mode": mode,
        "steps": result["n"],
        "shadow_mode": cfg.shadow_mode,
        "discarded_per_step": result["steps"][0]["discarded"] if result["steps"] else None,
        "last_fault": result["steps"][-1]["info"].get("fault_code") if result["steps"] else None,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    man = build_manifest("kuavo_sim_smoke", extra=summary)
    write_manifest(args.manifest, man)
    print(f"wrote manifest {args.manifest}")
    env.close()


if __name__ == "__main__":
    main()
