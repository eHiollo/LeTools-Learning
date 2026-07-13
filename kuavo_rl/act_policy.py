"""ACT policy adapters for Stage-A execute-first (local LeRobot or remote TCP)."""

from __future__ import annotations

import socket
from typing import Any

import numpy as np

from kuavo_rl.contracts import ACTION_DIM, IMAGE_KEYS, IMAGE_SHAPE_CHW
from kuavo_rl.ipc import pack_arrays, recv_msg, send_msg, unpack_arrays

# Back-compat aliases used by act_infer_server.py
_pack_arrays = pack_arrays
_unpack_arrays = unpack_arrays
send_pickle = send_msg
recv_pickle = recv_msg


def obs_to_act_numpy(obs: dict) -> dict[str, np.ndarray]:
    """Convert HIL-SERL obs (uint8 CHW or float) to ACT float32 CHW in [0, 1]."""
    out: dict[str, np.ndarray] = {}
    state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    if state.shape[0] != 16:
        raise ValueError(f"observation.state dim {state.shape[0]} != 16")
    out["observation.state"] = state

    _, h, w = IMAGE_SHAPE_CHW
    for key in IMAGE_KEYS:
        img = obs.get(key)
        if img is None:
            out[key] = np.zeros(IMAGE_SHAPE_CHW, dtype=np.float32)
            continue
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = np.transpose(arr, (2, 0, 1))
        if arr.shape[-2:] != (h, w):
            import cv2

            hwc = np.transpose(arr, (1, 2, 0))
            if hwc.dtype != np.uint8:
                if hwc.max() <= 1.0:
                    hwc = (hwc * 255.0).clip(0, 255)
                hwc = hwc.astype(np.uint8)
            hwc = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_AREA)
            arr = np.transpose(hwc, (2, 0, 1))
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
        out[key] = arr
    return out


class LerobotActChunkPolicy:
    """LeRobot ACTPolicy + preprocessor/postprocessor -> (chunk, 16) numpy."""

    def __init__(self, policy, preprocessor, postprocessor, device: str = "cuda"):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.policy.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint: str, device: str = "cuda") -> "LerobotActChunkPolicy":
        import torch
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies.factory import make_pre_post_processors

        policy = ACTPolicy.from_pretrained(checkpoint)
        policy.to(device)
        pre, post = make_pre_post_processors(
            policy_cfg=policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        return cls(policy, pre, post, device=device)

    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        import torch

        prepared = obs_to_act_numpy(obs)
        batch: dict[str, Any] = {}
        for k, v in prepared.items():
            t = torch.from_numpy(np.asarray(v))
            if t.ndim == 3:
                t = t.unsqueeze(0)
            elif t.ndim == 1:
                t = t.unsqueeze(0)
            batch[k] = t.to(self.device, dtype=torch.float32)

        with torch.inference_mode():
            batch = self.preprocessor(batch)
            batch = {
                k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            chunk = self.policy.predict_action_chunk(batch)
            chunk = self.postprocessor(chunk)

        arr = chunk.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[1] != ACTION_DIM:
            raise RuntimeError(f"unexpected ACT chunk shape {arr.shape}")
        return arr


class RemoteActChunkPolicy:
    """Host-side client: send obs to Docker ACT infer server, receive chunk."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, timeout_s: float = 30.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock

    def close(self, *, shutdown_server: bool = False) -> None:
        """Close client socket. Does not stop the Docker infer server by default."""
        if self._sock is not None:
            if shutdown_server:
                try:
                    send_msg(self._sock, {"cmd": "shutdown"})
                except Exception:  # noqa: BLE001
                    pass
            try:
                self._sock.close()
            except Exception:  # noqa: BLE001
                pass
            self._sock = None

    def predict_action_chunk(self, obs: dict) -> np.ndarray:
        self.connect()
        assert self._sock is not None
        prepared = obs_to_act_numpy(obs)
        send_msg(self._sock, {"cmd": "infer", "obs": _pack_arrays(prepared)})
        resp = recv_msg(self._sock)
        if not isinstance(resp, dict):
            raise RuntimeError(f"bad infer response type {type(resp)}")
        if "error" in resp:
            raise RuntimeError(f"infer server error: {resp['error']}")
        if "chunk" in resp and isinstance(resp["chunk"], tuple):
            chunk = _unpack_arrays({"chunk": resp["chunk"]})["chunk"]
        else:
            chunk = np.asarray(resp["chunk"], dtype=np.float32)
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
            raise RuntimeError(f"unexpected remote chunk shape {chunk.shape}")
        return chunk