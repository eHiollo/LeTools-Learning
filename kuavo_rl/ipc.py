"""Length-prefixed pickle IPC with NumPy-safe array packing (cross-version)."""

from __future__ import annotations

import pickle
import socket
import struct
from typing import Any

import numpy as np


def pack_arrays(arrays: dict[str, np.ndarray]) -> dict[str, tuple[str, list[int], bytes]]:
    packed: dict[str, tuple[str, list[int], bytes]] = {}
    for key, value in arrays.items():
        arr = np.ascontiguousarray(np.asarray(value))
        packed[key] = (str(arr.dtype), list(arr.shape), arr.tobytes())
    return packed


def unpack_arrays(packed: dict[str, tuple[str, list[int], bytes]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, (dtype, shape, data) in packed.items():
        out[key] = np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape).copy()
    return out


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed while receiving")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(conn: socket.socket, obj: Any) -> None:
    payload = pickle.dumps(obj, protocol=4)
    conn.sendall(struct.pack("!I", len(payload)) + payload)


def recv_msg(conn: socket.socket) -> Any:
    (n,) = struct.unpack("!I", _recv_exact(conn, 4))
    if n > 256 * 1024 * 1024:
        raise ValueError(f"payload too large: {n}")
    return pickle.loads(_recv_exact(conn, n))
