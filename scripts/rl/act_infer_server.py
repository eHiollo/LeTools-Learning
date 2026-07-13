#!/usr/bin/env python3
"""ACT infer TCP server (runs inside letools-train:hilserl; host has no Py3.11+ lerobot).

Protocol: length-prefixed pickle.
  request:  {"cmd": "infer", "obs": {observation.*: np.ndarray}}
            {"cmd": "ping"} / {"cmd": "shutdown"}
  response: {"chunk": float32 (T, 16)} or {"ok": True} or {"error": str}
"""

from __future__ import annotations

import argparse
import socket
import traceback
from pathlib import Path

from kuavo_rl.act_policy import (
    LerobotActChunkPolicy,
    _pack_arrays,
    _unpack_arrays,
    recv_msg,
    send_msg,
)


def serve(host: str, port: int, checkpoint: Path, device: str) -> None:
    print(f"[act-infer] loading {checkpoint} on {device}", flush=True)
    policy = LerobotActChunkPolicy.from_checkpoint(str(checkpoint), device=device)
    # Warmup once so first live step is not cold-start
    import numpy as np

    from kuavo_rl.contracts import IMAGE_KEYS, IMAGE_SHAPE_CHW

    warm = {"observation.state": np.zeros(16, dtype=np.float32)}
    for k in IMAGE_KEYS:
        warm[k] = np.zeros(IMAGE_SHAPE_CHW, dtype=np.float32)
    chunk = policy.predict_action_chunk(warm)
    print(f"[act-infer] warmup chunk={chunk.shape}", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    print(f"[act-infer] listening on {host}:{port}", flush=True)

    while True:
        conn, addr = sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[act-infer] client {addr}", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                cmd = req.get("cmd", "infer") if isinstance(req, dict) else "infer"
                if cmd == "ping":
                    send_msg(conn, {"ok": True})
                    continue
                if cmd == "shutdown":
                    send_msg(conn, {"ok": True})
                    print("[act-infer] shutdown requested", flush=True)
                    conn.close()
                    sock.close()
                    return
                try:
                    obs = req["obs"]
                    if isinstance(obs, dict) and obs and isinstance(next(iter(obs.values())), tuple):
                        obs = _unpack_arrays(obs)
                    chunk = policy.predict_action_chunk(obs)
                    packed = _pack_arrays({"chunk": chunk})["chunk"]
                    send_msg(conn, {"chunk": packed})
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    send_msg(conn, {"error": str(exc)})
        except (ConnectionError, EOFError, OSError) as exc:
            print(f"[act-infer] client disconnected: {exc}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/rl_runs/checkpoints/005000/pretrained_model"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    serve(args.host, args.port, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
