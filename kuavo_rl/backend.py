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

    def __init__(
        self,
        *,
        raw_joint_dim: int = 28,
        image_chw: bool = True,
        image_shape_chw: tuple[int, int, int] | None = None,
    ):
        from kuavo_rl.contracts import STATE_DIM

        self.raw_joint_dim = raw_joint_dim
        self.image_chw = image_chw
        self.image_shape_chw = image_shape_chw or IMAGE_SHAPE_CHW
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
        c, h, w = self.image_shape_chw
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

    def __init__(
        self,
        kuavo_env: Any,
        *,
        publish_unit: str = "rad_to_sdk",
        image_shape_chw: tuple[int, int, int] | None = None,
    ):
        from kuavo_rl.kuavo_bridge import KuavoGymBridge

        self.env = kuavo_env
        self.publish_unit = publish_unit
        self.image_shape_chw = image_shape_chw or IMAGE_SHAPE_CHW
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

    def close(self) -> None:
        try:
            closer = getattr(self.env, "close", None)
            if callable(closer):
                closer()
        except Exception:  # noqa: BLE001
            pass

    def _from_normalized(self, obs: dict) -> BackendObservation:
        images = {k: np.asarray(obs[k]) for k in IMAGE_KEYS if k in obs}
        c, h, w = self.image_shape_chw
        for key in IMAGE_KEYS:
            if key not in images:
                images[key] = np.zeros((c, h, w), dtype=np.uint8)
                continue
            img = images[key]
            if img.shape != (c, h, w):
                import cv2

                hwc = np.transpose(img, (1, 2, 0)) if img.ndim == 3 and img.shape[0] in (1, 3) else img
                if hwc.dtype != np.uint8:
                    if float(np.max(hwc)) <= 1.0:
                        hwc = (hwc * 255.0).clip(0, 255)
                    hwc = hwc.astype(np.uint8)
                hwc = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_AREA)
                images[key] = np.transpose(hwc, (2, 0, 1))
        return BackendObservation(
            state=np.asarray(obs["observation.state"], dtype=np.float32),
            images=images,
            timestamp_s=time.time(),
            observation_age_s=float(obs.get("observation_age_s", 0.0)),
            cross_topic_skew_s=float(obs.get("cross_topic_skew_s", 0.0)),
            raw_joint_dim=int(obs.get("raw_joint_dim", 28)),
        )


class ProxyBackend(RobotBackend):
    """Docker-side backend: talk to host ROS bridge over TCP (--network host)."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8877,
        timeout_s: float = 30.0,
        image_shape_chw: tuple[int, int, int] | None = None,
    ):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.image_shape_chw = image_shape_chw or IMAGE_SHAPE_CHW
        self._sock: Any = None
        self._t0 = time.time()

    def _connect(self) -> None:
        if self._sock is not None:
            return
        import socket

        from kuavo_rl.ipc import recv_msg, send_msg

        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        # Must consume the hello ack; otherwise the next RPC reads this response.
        send_msg(self._sock, {"cmd": "hello", "image_shape_chw": list(self.image_shape_chw)})
        hello = recv_msg(self._sock)
        if not isinstance(hello, dict) or hello.get("error"):
            self.close()
            raise RuntimeError(f"ros bridge hello failed: {hello!r}")
        if not hello.get("ok", False):
            self.close()
            raise RuntimeError(f"ros bridge hello rejected: {hello!r}")

    def _rpc(self, req: dict) -> dict:
        from kuavo_rl.ipc import recv_msg, send_msg

        self._connect()
        send_msg(self._sock, req)
        resp = recv_msg(self._sock)
        if not isinstance(resp, dict):
            raise RuntimeError(f"bad proxy response type {type(resp)}")
        if "error" in resp:
            raise RuntimeError(f"ros bridge error: {resp['error']}")
        return resp

    def reset(self, *, seed: int | None = None) -> BackendObservation:
        resp = self._rpc({"cmd": "reset", "seed": seed})
        return self._obs_from_resp(resp)

    def get_observation(self) -> BackendObservation:
        resp = self._rpc({"cmd": "get_obs"})
        return self._obs_from_resp(resp)

    def publish(self, command: PublishedCommand) -> None:
        from kuavo_rl.ipc import pack_arrays

        self._rpc(
            {
                "cmd": "publish",
                "action": pack_arrays({"action": command.clipped_action.astype(np.float32)})["action"],
            }
        )

    def is_stop(self) -> bool:
        return bool(self._rpc({"cmd": "signals"}).get("stop", False))

    def is_pause(self) -> bool:
        return bool(self._rpc({"cmd": "signals"}).get("pause", False))

    def is_shutdown(self) -> bool:
        return bool(self._rpc({"cmd": "signals"}).get("shutdown", False))

    def close(self) -> None:
        try:
            if self._sock is not None:
                from kuavo_rl.ipc import send_msg

                send_msg(self._sock, {"cmd": "close"})
                self._sock.close()
        except Exception:  # noqa: BLE001
            pass
        self._sock = None

    def _obs_from_resp(self, resp: dict) -> BackendObservation:
        from kuavo_rl.ipc import unpack_arrays

        arrays = unpack_arrays(resp["arrays"])
        images = {k: arrays[k] for k in IMAGE_KEYS if k in arrays}
        c, h, w = self.image_shape_chw
        for key in IMAGE_KEYS:
            if key not in images:
                images[key] = np.zeros((c, h, w), dtype=np.uint8)
        return BackendObservation(
            state=arrays["observation.state"].astype(np.float32),
            images=images,
            timestamp_s=float(resp.get("timestamp_s", time.time() - self._t0)),
            observation_age_s=float(resp.get("observation_age_s", 0.0)),
            cross_topic_skew_s=float(resp.get("cross_topic_skew_s", 0.0)),
            raw_joint_dim=int(resp.get("raw_joint_dim", 28)),
        )
