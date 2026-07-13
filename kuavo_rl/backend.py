"""Robot backends: Mock for tests, optional ROS wrapper for sim/real."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kuavo_rl.contracts import IMAGE_KEYS, IMAGE_SHAPE_CHW
from kuavo_rl.ros_adapter import PublishedCommand


@dataclass
class BackendObservation:
    state: np.ndarray
    images: dict[str, np.ndarray]
    timestamp_s: float
    observation_age_s: float = 0.0
    cross_topic_skew_s: float = 0.0
    raw_joint_dim: int = 28
    extras: dict[str, Any] = field(default_factory=dict)

    def as_gym_obs(self) -> dict[str, Any]:
        obs: dict[str, Any] = {"observation.state": self.state.astype(np.float32)}
        for k, v in self.images.items():
            obs[k] = v
        return obs


class RobotBackend(ABC):
    @abstractmethod
    def reset(self, *, seed: int | None = None) -> BackendObservation:
        raise NotImplementedError

    @abstractmethod
    def get_observation(self) -> BackendObservation:
        raise NotImplementedError

    @abstractmethod
    def publish(self, command: PublishedCommand) -> None:
        raise NotImplementedError

    @abstractmethod
    def is_stop(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_pause(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_shutdown(self) -> bool:
        raise NotImplementedError

    def close(self) -> None:
        return None


class MockBackend(RobotBackend):
    """Deterministic in-memory backend for unit tests (no ROS)."""

    def __init__(self, *, raw_joint_dim: int = 28, image_chw: bool = True):
        from kuavo_rl.contracts import STATE_DIM

        self.raw_joint_dim = raw_joint_dim
        self.image_chw = image_chw
        self._state = np.zeros(STATE_DIM, dtype=np.float32)
        self._stop = False
        self._pause = False
        self._shutdown = False
        self._last_command: PublishedCommand | None = None
        self._t0 = time.time()
        self.publish_count = 0

    def set_signals(
        self,
        *,
        stop: bool | None = None,
        pause: bool | None = None,
        shutdown: bool | None = None,
    ) -> None:
        if stop is not None:
            self._stop = stop
        if pause is not None:
            self._pause = pause
        if shutdown is not None:
            self._shutdown = shutdown

    def reset(self, *, seed: int | None = None) -> BackendObservation:
        from kuavo_rl.contracts import STATE_DIM

        rng = np.random.default_rng(seed)
        self._state = rng.normal(0, 0.05, size=STATE_DIM).astype(np.float32)
        self._state[7] = self._state[15] = 0.0
        self._last_command = None
        self.publish_count = 0
        self._stop = False
        self._pause = False
        self._shutdown = False
        return self.get_observation()

    def get_observation(self) -> BackendObservation:
        c, h, w = IMAGE_SHAPE_CHW
        images = {}
        for key in IMAGE_KEYS:
            if self.image_chw:
                images[key] = np.zeros((c, h, w), dtype=np.uint8)
            else:
                images[key] = np.zeros((h, w, c), dtype=np.uint8)
        return BackendObservation(
            state=self._state.copy(),
            images=images,
            timestamp_s=time.time() - self._t0,
            observation_age_s=0.01,
            cross_topic_skew_s=0.005,
            raw_joint_dim=self.raw_joint_dim,
        )

    def publish(self, command: PublishedCommand) -> None:
        if self._stop or self._shutdown:
            raise RuntimeError("refusing to publish under stop/shutdown")
        self._state = command.clipped_action.astype(np.float32).copy()
        self._last_command = command
        self.publish_count += 1

    def is_stop(self) -> bool:
        return self._stop

    def is_pause(self) -> bool:
        return self._pause

    def is_shutdown(self) -> bool:
        return self._shutdown


class ROSBackend(RobotBackend):
    """
    Optional ROS/SDK backend wrapping an existing Kuavo Gym env.

    Uses KuavoGymBridge to normalize torch-batched obs into float32/uint8 contracts.
    Assumes SDK `control_arm_joint_positions` takes **radians** (current deploy path).
    """

    def __init__(self, kuavo_env: Any, *, publish_unit: str = "rad_to_sdk"):
        from kuavo_rl.kuavo_bridge import KuavoGymBridge

        self.env = kuavo_env
        self.publish_unit = publish_unit
        self.bridge = KuavoGymBridge(kuavo_env)

    def reset(self, *, seed: int | None = None) -> BackendObservation:
        obs = self.bridge.reset(seed=seed)
        return self._from_normalized(obs)

    def get_observation(self) -> BackendObservation:
        return self._from_normalized(self.bridge.get_obs())

    def publish(self, command: PublishedCommand) -> None:
        if self.publish_unit == "rad_to_sdk":
            self.bridge.publish_command(command)
        elif self.publish_unit == "rad_to_deg_topic":
            raise NotImplementedError(
                "direct /kuavo_arm_traj deg publish path must be wired to a ROS publisher; "
                "use rad_to_sdk unless verified"
            )
        else:
            raise ValueError(f"unknown publish_unit={self.publish_unit}")

    def is_stop(self) -> bool:
        return self.bridge.is_stop()

    def is_pause(self) -> bool:
        return self.bridge.is_pause()

    def is_shutdown(self) -> bool:
        try:
            import rospy

            return bool(rospy.is_shutdown())
        except Exception:
            return False

    def _from_normalized(self, obs: dict) -> BackendObservation:
        images = {k: np.asarray(obs[k]) for k in IMAGE_KEYS if k in obs}
        for key in IMAGE_KEYS:
            if key not in images:
                c, h, w = IMAGE_SHAPE_CHW
                images[key] = np.zeros((c, h, w), dtype=np.uint8)
        return BackendObservation(
            state=np.asarray(obs["observation.state"], dtype=np.float32),
            images=images,
            timestamp_s=time.time(),
            observation_age_s=float(obs.get("observation_age_s", 0.0)),
            cross_topic_skew_s=float(obs.get("cross_topic_skew_s", 0.0)),
            raw_joint_dim=int(obs.get("raw_joint_dim", 28)),
        )
