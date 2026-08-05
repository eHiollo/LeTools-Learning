"""Deployment-only YAML parsing for RL-100 real-robot inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from kuavo_rl.contracts import RL100_ACTION_DIM, RL100_STATE_DIM, RL100_TOPIC_NATIVE_CONTRACT
from kuavo_rl.rl100_real_runner import RL100TopicRunnerLimits, RL100TopicSafetyGate


@dataclass(frozen=True)
class RL100DeployConfig:
    path: Path
    raw: dict[str, Any]

    @property
    def checkpoint_path(self) -> str:
        return str(self.raw["checkpoint"]["path"])

    @property
    def contract(self) -> str:
        return str(self.raw.get("contract", ""))

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
        if value < 1 or value > 100:
            raise ValueError("RL-100 real deployment control_hz must be in [1,100]")
        return value

    @property
    def execute_steps(self) -> int:
        value = int(self.raw.get("inference", {}).get("execute_steps", 1))
        if value != 1:
            raise ValueError("topic-native live currently requires execute_steps=1")
        return value

    @property
    def pointcloud_config(self) -> str:
        return str(self.raw.get("observation", {}).get("pointcloud_config", ""))

    @property
    def require_publish_subscribers(self) -> bool:
        return bool(self.raw.get("publish", {}).get("require_subscribers", True))

    def runner_limits(self) -> RL100TopicRunnerLimits:
        obs = self.raw.get("observation", {})
        safety = self.raw.get("safety", {})
        inference = self.raw.get("inference", {})
        return RL100TopicRunnerLimits(
            joint_state_max_age_s=float(obs.get("joint_state_max_age_s", 0.15)),
            dexhand_state_max_age_s=float(obs.get("dexhand_state_max_age_s", 0.15)),
            state_max_skew_s=float(obs.get("state_max_skew_s", 0.10)),
            depth_max_age_s=float(obs.get("depth_max_age_s", 0.15)),
            max_camera_skew_s=float(obs.get("max_camera_skew_s", 0.10)),
            max_state_cloud_skew_s=float(obs.get("max_state_cloud_skew_s", 0.10)),
            inference_timeout_s=float(inference.get("timeout_s", 1.0 / self.control_hz)),
            max_consecutive_source_failures=int(safety.get("max_consecutive_source_failures", 1)),
            fault_hold_once_if_state_fresh=bool(safety.get("fault_hold_once_if_state_fresh", True)),
        )

    def safety_gate(self, *, require_approved: bool) -> RL100TopicSafetyGate:
        safety = self.raw.get("safety", {})
        low, high = safety.get("arm_joint_low_rad"), safety.get("arm_joint_high_rad")
        if low is None or high is None:
            if require_approved:
                raise ValueError("live deploy requires field-approved 14-D arm_joint_low_rad/high_rad")
            low = [-3.14] * 14
            high = [3.14] * 14
        if len(low) != 14 or len(high) != 14:
            raise ValueError("arm_joint_low_rad/high_rad must each contain 14 values")
        return RL100TopicSafetyGate(
            arm_low_rad=np.asarray(low, dtype=np.float32),
            arm_high_rad=np.asarray(high, dtype=np.float32),
            max_arm_step_rad=float(safety.get("max_arm_step_rad", 0.02)),
            max_hand_step=float(safety.get("max_hand_step", 5.0)),
            max_consecutive_clips=int(safety.get("max_consecutive_clips", 3)),
            max_arm_state_jump_rad=float(safety.get("max_arm_state_jump_rad", 0.05)),
            max_hand_state_jump=float(safety.get("max_hand_state_jump", 5.0)),
        )

    def startup_requirements(self, *, require_approved: bool) -> dict[str, Any]:
        startup = self.raw.get("startup", {})
        if require_approved and not bool(startup.get("require_operator_start", True)):
            raise ValueError("live deploy requires startup.require_operator_start=true")
        return {
            "hand_default_tolerance": float(startup.get("hand_default_tolerance", 2.0)),
            "require_physical_estop_ready": bool(startup.get("require_physical_estop_ready", True)),
        }


def load_rl100_deploy_config(path: str | Path) -> RL100DeployConfig:
    resolved = Path(path).expanduser().resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("checkpoint"), dict):
        raise ValueError("deployment config requires a checkpoint mapping")
    if "path" not in raw["checkpoint"]:
        raise ValueError("deployment config checkpoint.path is required")
    cfg = RL100DeployConfig(path=resolved, raw=raw)
    if cfg.contract != RL100_TOPIC_NATIVE_CONTRACT:
        raise ValueError(f"deployment contract must be {RL100_TOPIC_NATIVE_CONTRACT!r}")
    if int(raw.get("state_dim", RL100_STATE_DIM)) != RL100_STATE_DIM:
        raise ValueError("deployment state_dim must be 32")
    if int(raw.get("action_dim", RL100_ACTION_DIM)) != RL100_ACTION_DIM:
        raise ValueError("deployment action_dim must be 26")
    observation = raw.get("observation", {})
    if int(observation.get("expected_raw_joint_dim", 20)) != 20:
        raise ValueError("deployment expected_raw_joint_dim must be 20")
    if int(observation.get("expected_dexhand_dim", 12)) != 12:
        raise ValueError("deployment expected_dexhand_dim must be 12")
    publish = raw.get("publish", {})
    if str(publish.get("arm_unit", "degree")) != "degree":
        raise ValueError("topic-native arm publishing requires publish.arm_unit=degree")
    if list(publish.get("hand_range", [])) != [0, 100]:
        raise ValueError("publish.hand_range must be [0,100]")
    if str(publish.get("hand_rounding", "nearest")) != "nearest":
        raise ValueError("topic-native hand publishing requires nearest rounding")
    if list(publish.get("arm_joint_names", [])) != [f"arm_joint_{i}" for i in range(14)]:
        raise ValueError("publish.arm_joint_names must be arm_joint_0..arm_joint_13")
    cfg.execute_steps
    if not cfg.pointcloud_config:
        raise ValueError("deployment observation.pointcloud_config is required")
    return cfg
