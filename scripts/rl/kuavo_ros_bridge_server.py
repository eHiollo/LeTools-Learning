#!/usr/bin/env python3
"""Host ROS bridge for Stage-B Docker actor (Kuavo-Sim / real).

Docker runs lerobot actor/learner (no ROS). This process owns Kuavo Gym + ROS and
serves reset/get_obs/publish over TCP. Use with --network host so Docker reaches
127.0.0.1:8877.
"""

from __future__ import annotations

import argparse
import socket
import traceback
from pathlib import Path

import numpy as np

from kuavo_rl.contracts import IMAGE_KEYS
from kuavo_rl.ipc import pack_arrays, recv_msg, send_msg, unpack_arrays
from kuavo_rl.kuavo_bridge import KuavoGymBridge
from kuavo_rl.ros_adapter import build_published_command


def _resize_obs(obs: dict, image_shape_chw: tuple[int, int, int]) -> dict:
    import cv2

    c, h, w = image_shape_chw
    out = {"observation.state": np.asarray(obs["observation.state"], dtype=np.float32)}
    for key in IMAGE_KEYS:
        img = np.asarray(obs.get(key, np.zeros((c, h, w), dtype=np.uint8)))
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        if img.shape != (c, h, w):
            hwc = np.transpose(img, (1, 2, 0))
            if hwc.dtype != np.uint8:
                if float(np.max(hwc)) <= 1.0:
                    hwc = (hwc * 255.0).clip(0, 255)
                hwc = hwc.astype(np.uint8)
            hwc = cv2.resize(hwc, (w, h), interpolation=cv2.INTER_AREA)
            img = np.transpose(hwc, (2, 0, 1))
        out[key] = img.astype(np.uint8)
    for meta in ("observation_age_s", "cross_topic_skew_s", "raw_joint_dim"):
        if meta in obs:
            out[meta] = obs[meta]
    return out


def _obs_response(obs: dict, image_shape_chw: tuple[int, int, int]) -> dict:
    obs = _resize_obs(obs, image_shape_chw)
    arrays = {"observation.state": obs["observation.state"]}
    for key in IMAGE_KEYS:
        arrays[key] = obs[key]
    return {
        "arrays": pack_arrays(arrays),
        "observation_age_s": float(obs.get("observation_age_s", 0.0)),
        "cross_topic_skew_s": float(obs.get("cross_topic_skew_s", 0.0)),
        "raw_joint_dim": int(obs.get("raw_joint_dim", 28)),
    }


def make_bridge(deploy_config: Path) -> KuavoGymBridge:
    import gymnasium as gym
    import kuavo_deploy.kuavo_env  # noqa: F401
    from kuavo_deploy.config import load_kuavo_config

    deploy_cfg = load_kuavo_config(str(deploy_config))
    if deploy_cfg.env.env_name != "Kuavo-Sim" and deploy_cfg.env.env_name != "Kuavo-Real":
        raise RuntimeError(f"unexpected env_name={deploy_cfg.env.env_name}")
    env = gym.make(
        deploy_cfg.env.env_name,
        max_episode_steps=int(deploy_cfg.inference.max_episode_steps),
        config=deploy_cfg,
    )
    return KuavoGymBridge(env)


def serve(host: str, port: int, deploy_config: Path, default_image_shape: tuple[int, int, int]) -> None:
    print(f"[ros-bridge] making Kuavo env from {deploy_config}", flush=True)
    bridge = make_bridge(deploy_config)
    image_shape = default_image_shape
    print(f"[ros-bridge] listening on {host}:{port}", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # rospy may set socket.setdefaulttimeout(60); accept must not inherit that.
    sock.settimeout(None)
    sock.bind((host, port))
    sock.listen(1)

    while True:
        conn, addr = sock.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"[ros-bridge] client {addr}", flush=True)
        try:
            while True:
                req = recv_msg(conn)
                cmd = req.get("cmd") if isinstance(req, dict) else None
                try:
                    if cmd == "hello":
                        shape = req.get("image_shape_chw")
                        if shape and len(shape) == 3:
                            image_shape = tuple(int(x) for x in shape)
                        send_msg(conn, {"ok": True, "image_shape_chw": list(image_shape)})
                    elif cmd == "reset":
                        obs = bridge.reset(seed=req.get("seed"))
                        send_msg(conn, _obs_response(obs, image_shape))
                    elif cmd == "get_obs":
                        obs = bridge.get_obs()
                        send_msg(conn, _obs_response(obs, image_shape))
                    elif cmd == "publish":
                        action = unpack_arrays({"action": req["action"]})["action"].astype(np.float32)
                        cmd_pub = build_published_command(action, action)
                        bridge.publish_command(cmd_pub)
                        send_msg(conn, {"ok": True})
                    elif cmd == "signals":
                        send_msg(
                            conn,
                            {
                                "stop": bridge.is_stop(),
                                "pause": bridge.is_pause(),
                                "shutdown": False,
                            },
                        )
                    elif cmd == "close":
                        send_msg(conn, {"ok": True})
                        break
                    elif cmd == "shutdown":
                        send_msg(conn, {"ok": True})
                        conn.close()
                        sock.close()
                        return
                    else:
                        send_msg(conn, {"error": f"unknown cmd {cmd}"})
                except Exception as exc:  # noqa: BLE001
                    traceback.print_exc()
                    send_msg(conn, {"error": str(exc)})
        except (ConnectionError, EOFError, OSError) as exc:
            print(f"[ros-bridge] client disconnected: {exc}", flush=True)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument(
        "--deploy-config",
        type=Path,
        default=Path("configs/deploy/total/deploy_sim_smoke_cams_total.yaml"),
    )
    parser.add_argument("--image-h", type=int, default=128)
    parser.add_argument("--image-w", type=int, default=128)
    args = parser.parse_args()
    serve(args.host, args.port, args.deploy_config, (3, args.image_h, args.image_w))


if __name__ == "__main__":
    main()
