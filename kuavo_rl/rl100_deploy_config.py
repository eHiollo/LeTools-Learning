"""Deployment-only YAML parsing for RL-100 real-robot inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kuavo_rl.config import default_safety_config
from kuavo_rl.rl100_real_runner import RunnerLimits, make_safety_config


@dataclass(frozen=True)
class RL100DeployConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def checkpoint_path(self) -> str:
        return str(self.raw["checkpoint"]["path"])

    @property
    def device(self) -> str:
        return str(self.raw["checkpoint"].get("device", "cuda:0"))

    @property
    def model_source(self) -> str:
        return str(self.raw["checkpoint"].get("model_source", "auto"))

    @property
    def shadow_mode(self) -> bool:
        return bool(self.raw.get("mode", {}).get("shadow", True))

    @property
    def control_hz(self) -> int:
        value = int(self.raw.get("inference", {}).get("control_hz", 10))
        if value not in (5, 10):
            raise ValueError("RL-100 real deployment control_hz must be 5 or 10")
        return value

    def runner_limits(self) -> RunnerLimits:
        obs = self.raw.get("observation", {})
        safety = self.raw.get("safety", {})
        inference = self.raw.get("inference", {})
        return RunnerLimits(
            state_max_age_s=float(obs.get("state_max_age_s", 0.15)),
            hand_max_age_s=float(obs.get("hand_max_age_s", 0.15)),
            depth_max_age_s=float(obs.get("depth_max_age_s", 0.15)),
            max_state_hand_skew_s=float(obs.get("max_state_hand_skew_s", 0.10)),
            max_camera_skew_s=float(obs.get("max_camera_skew_s", 0.10)),
            max_state_cloud_skew_s=float(obs.get("max_state_cloud_skew_s", 0.10)),
            inference_timeout_s=float(inference.get("timeout_s", 1.0 / self.control_hz)),
            max_arm_state_jump_rad=float(safety.get("max_arm_state_jump_rad", 0.05)),
            max_gripper_state_jump=float(safety.get("max_gripper_state_jump", 0.10)),
            max_consecutive_source_failures=int(safety.get("max_consecutive_source_failures", 1)),
        )

    def safety_config(self, *, require_approved: bool) -> Any:
        safety = self.raw.get("safety", {})
        low, high = safety.get("arm_joint_low_rad"), safety.get("arm_joint_high_rad")
        if low is None or high is None:
            if require_approved:
                raise ValueError("live deploy requires field-approved 14-D arm_joint_low_rad/high_rad")
            return default_safety_config(control_dt_s=1.0 / self.control_hz)
        if len(low) != 14 or len(high) != 14:
            raise ValueError("arm_joint_low_rad/high_rad must each contain 14 values")
        return make_safety_config(
            np.asarray(low, dtype=np.float32),
            np.asarray(high, dtype=np.float32),
            max_arm_step_rad=float(safety.get("max_arm_step_rad", 0.02)),
            max_gripper_step=float(safety.get("max_gripper_step", 0.05)),
            control_dt_s=1.0 / self.control_hz,
            max_consecutive_clips=int(safety.get("max_consecutive_clips", 3)),
        )

    def expected_pose16(self, *, require_approved: bool) -> np.ndarray | None:
        pose = self.raw.get("startup", {}).get("expected_pose16")
        if pose is None:
            if require_approved:
                raise ValueError("live deploy requires startup.expected_pose16")
            return None
        array = np.asarray(pose, dtype=np.float32).reshape(-1)
        if array.shape != (16,):
            raise ValueError("startup.expected_pose16 must be 16-D")
        return array


def load_rl100_deploy_config(path: str | Path) -> RL100DeployConfig:
    resolved = Path(path).expanduser().resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("checkpoint"), dict):
        raise ValueError("deployment config requires a checkpoint mapping")
    if "path" not in raw["checkpoint"]:
        raise ValueError("deployment config checkpoint.path is required")
    return RL100DeployConfig(path=resolved, raw=raw)
