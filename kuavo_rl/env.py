"""KuavoHILSerlEnv: Gymnasium env shared by ACT stage-A and HIL-SERL stage-B."""

from __future__ import annotations

import time
import uuid
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kuavo_rl.backend import MockBackend, RobotBackend
from kuavo_rl.config import EnvConfig, default_safety_config
from kuavo_rl.contracts import ACTION_DIM, IMAGE_KEYS, IMAGE_SHAPE_CHW, STATE_DIM, FaultCode
from kuavo_rl.reward import DeterministicRewardProvider, EpisodeFrame, RobometerRewardWorker
from kuavo_rl.ros_adapter import action_to_audit_dict, build_published_command, observation_contract_check
from kuavo_rl.safety import SafetyGate
from kuavo_rl.teleop import TeleopEvent, TeleopAdapter


class KuavoHILSerlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig | None = None,
        backend: RobotBackend | None = None,
        teleop: TeleopAdapter | None = None,
        reward_worker: RobometerRewardWorker | None = None,
    ):
        super().__init__()
        self.config = config or EnvConfig(safety=default_safety_config())
        assert self.config.safety is not None
        self.backend = backend or MockBackend()
        self.teleop = teleop or TeleopAdapter()
        self.gate = SafetyGate(self.config.safety)
        self.det_reward = DeterministicRewardProvider(
            self.config.reward,
            success_reward=self.config.episode.success_reward,
            failure_reward=self.config.episode.failure_reward,
            safety_penalty=self.config.episode.safety_penalty,
        )
        self.reward_worker = reward_worker or RobometerRewardWorker(self.config.reward)
        self.reward_worker.start()

        self.action_space = spaces.Box(
            low=self.config.safety.joint_position_low,
            high=self.config.safety.joint_position_high,
            dtype=np.float32,
        )
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=IMAGE_SHAPE_CHW,
            dtype=np.uint8,
        )
        self.observation_space = spaces.Dict(
            {
                "observation.state": spaces.Box(
                    low=-np.pi, high=np.pi, shape=(STATE_DIM,), dtype=np.float32
                ),
                **{k: image_space for k in IMAGE_KEYS},
            }
        )

        self._episode_id = ""
        self._step_count = 0
        self._episode_start_s = 0.0
        self._frames: list[EpisodeFrame] = []
        self._closed = False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._episode_start_s = time.time()
        self._frames = []
        self.teleop.reset()
        be_obs = self.backend.reset(seed=seed)
        self.gate.reset(initial_action=be_obs.state)
        obs = be_obs.as_gym_obs()
        errors = observation_contract_check(obs)
        if errors:
            raise ValueError(f"observation contract failed on reset: {errors}")
        info = {
            "episode_id": self._episode_id,
            "fault_code": FaultCode.NONE.value,
            "is_intervention": False,
            "success": False,
            "action_clipped": False,
            "timestamp": be_obs.timestamp_s,
            "shadow_mode": self.config.shadow_mode,
        }
        return obs, info

    def step(self, action):
        if self._closed:
            raise RuntimeError("env already closed")

        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        event = self.teleop.poll()

        # Manual events have priority over policy
        manual = self.det_reward.from_manual(
            success=event.success, failure=event.failure, abort=event.abort
        )
        if manual is not None:
            be_obs = self.backend.get_observation()
            obs = be_obs.as_gym_obs()
            info = self._info(
                be_obs.timestamp_s,
                fault=manual.fault_code,
                intervention=event.is_intervention,
                success=manual.success,
                clipped=False,
                reward_source=manual.source,
            )
            return obs, float(manual.reward), manual.terminated, manual.truncated, info

        # Pause: do not publish new action
        if self.backend.is_pause() or event.pause:
            be_obs = self.backend.get_observation()
            paused_for = time.time() - self._episode_start_s
            if paused_for > self.config.episode.pause_timeout_s:
                decision = self.det_reward.from_fault(FaultCode.PAUSE_TIMEOUT)
                info = self._info(
                    be_obs.timestamp_s,
                    fault=decision.fault_code,
                    intervention=False,
                    success=False,
                    clipped=False,
                    reward_source=decision.source,
                )
                return be_obs.as_gym_obs(), float(decision.reward), False, True, info
            info = self._info(
                be_obs.timestamp_s,
                fault=FaultCode.NONE,
                intervention=False,
                success=False,
                clipped=False,
                reward_source="pause_hold",
            )
            return be_obs.as_gym_obs(), 0.0, False, False, info

        # Teleop override
        step_action = raw
        is_intervention = False
        if event.is_intervention and event.action is not None:
            if self.config.safety.require_deadman_for_teleop and not event.deadman:
                # ignore teleop without deadman
                pass
            else:
                step_action = np.asarray(event.action, dtype=np.float32).reshape(-1)
                is_intervention = True

        be_obs_pre = self.backend.get_observation()
        gate = self.gate.check(
            step_action,
            stop=self.backend.is_stop() or event.stop,
            ros_shutdown=self.backend.is_shutdown(),
            observation_age_s=be_obs_pre.observation_age_s,
            cross_topic_skew_s=be_obs_pre.cross_topic_skew_s,
        )
        if not gate.ok:
            decision = self.det_reward.from_fault(gate.fault_code)
            info = self._info(
                be_obs_pre.timestamp_s,
                fault=gate.fault_code,
                intervention=is_intervention,
                success=False,
                clipped=gate.clipped,
                reward_source=decision.source,
                audit={"held_action": gate.action.tolist()},
            )
            return be_obs_pre.as_gym_obs(), float(decision.reward), decision.terminated, decision.truncated, info

        cmd = build_published_command(
            raw_action=step_action,
            clipped_action=gate.action,
            claw_scale=self.config.claw_command_scale,
        )

        if not self.config.shadow_mode:
            try:
                self.backend.publish(cmd)
            except Exception as exc:  # noqa: BLE001
                decision = self.det_reward.from_fault(FaultCode.SDK_EXCEPTION)
                info = self._info(
                    be_obs_pre.timestamp_s,
                    fault=FaultCode.SDK_EXCEPTION,
                    intervention=is_intervention,
                    success=False,
                    clipped=gate.clipped,
                    reward_source=decision.source,
                    audit={"error": str(exc)},
                )
                return be_obs_pre.as_gym_obs(), float(decision.reward), True, False, info

        if self.gate.clips_exceeded():
            be_obs = self.backend.get_observation()
            decision = self.det_reward.from_fault(FaultCode.ACTION_LIMIT)
            info = self._info(
                be_obs.timestamp_s,
                fault=FaultCode.ACTION_LIMIT,
                intervention=is_intervention,
                success=False,
                clipped=True,
                reward_source="max_consecutive_clips",
                audit=action_to_audit_dict(cmd),
            )
            return be_obs.as_gym_obs(), float(decision.reward), False, True, info

        be_obs = self.backend.get_observation()
        obs = be_obs.as_gym_obs()
        self._step_count += 1
        # Buffer head camera for async reward
        head = obs.get("observation.images.head_cam_h")
        if head is not None:
            self._frames.append(EpisodeFrame(image=np.asarray(head), timestamp_s=be_obs.timestamp_s))

        terminated = False
        truncated = False
        reward = 0.0
        fault = FaultCode.NONE
        reward_source = "step_zero"
        success = False

        if self._step_count >= self.config.episode.max_steps or (
            time.time() - self._episode_start_s
        ) >= self.config.episode.max_duration_s:
            truncated = True
            fault = FaultCode.EPISODE_TIMEOUT
            decision = self.det_reward.from_fault(fault)
            reward = float(decision.reward)
            reward_source = decision.source
            self._submit_robometer()

        info = self._info(
            be_obs.timestamp_s,
            fault=fault,
            intervention=is_intervention,
            success=success,
            clipped=gate.clipped,
            reward_source=reward_source,
            audit=action_to_audit_dict(cmd),
        )
        info["step"] = self._step_count
        return obs, reward, terminated, truncated, info

    def _submit_robometer(self) -> None:
        if self.config.reward.robometer_mode == "disabled":
            return
        self.reward_worker.submit(self._episode_id, list(self._frames), self.config.reward.task_text)

    def _info(
        self,
        timestamp: float,
        *,
        fault: FaultCode,
        intervention: bool,
        success: bool,
        clipped: bool,
        reward_source: str,
        audit: dict | None = None,
    ) -> dict[str, Any]:
        return {
            "episode_id": self._episode_id,
            "fault_code": fault.value if isinstance(fault, FaultCode) else str(fault),
            "is_intervention": bool(intervention),
            "success": bool(success),
            "action_clipped": bool(clipped),
            "timestamp": float(timestamp),
            "reward_source": reward_source,
            "shadow_mode": self.config.shadow_mode,
            "action_audit": audit or {},
        }

    def close(self):
        self._closed = True
        self.reward_worker.stop()
        self.backend.close()
