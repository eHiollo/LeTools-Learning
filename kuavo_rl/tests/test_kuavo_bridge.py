"""Unit tests for Kuavo obs/action bridge (no ROS)."""

import numpy as np
import torch

from kuavo_rl.kuavo_bridge import KuavoGymBridge, normalize_kuavo_obs
from kuavo_rl.ros_adapter import build_published_command


class FakeKuavoEnv:
    def __init__(self):
        self.published = []
        self.control_signal_manager = type(
            "C",
            (),
            {"stop_flag": type("E", (), {"is_set": lambda self: False})(), "pause_flag": type("E", (), {"is_set": lambda self: False})()},
        )()

    def reset(self, seed=None):
        return self.get_obs(), {}

    def get_obs(self):
        return {
            "observation.state": torch.zeros(1, 16),
            "observation.images.head_cam_h": torch.zeros(1, 3, 480, 848),
            "wrist_cam_l": np.zeros((480, 848, 3), dtype=np.uint8),
            "observation.images.wrist_cam_r": (np.ones((3, 480, 848), dtype=np.float32) * 0.5),
        }

    def exec_action(self, action):
        self.published.append(np.asarray(action).copy())


def test_normalize_batched_torch_and_hwc():
    env = FakeKuavoEnv()
    obs = normalize_kuavo_obs(env.get_obs())
    assert obs["observation.state"].shape == (16,)
    assert obs["observation.state"].dtype == np.float32
    assert obs["observation.images.head_cam_h"].shape == (3, 480, 848)
    assert obs["observation.images.wrist_cam_l"].shape == (3, 480, 848)
    assert obs["observation.images.wrist_cam_r"].dtype == np.uint8
    assert obs["observation.images.wrist_cam_r"].max() == 127 or obs["observation.images.wrist_cam_r"].max() == 128


def test_bridge_exec_action():
    env = FakeKuavoEnv()
    bridge = KuavoGymBridge(env)
    cmd = build_published_command(np.zeros(16, np.float32), np.zeros(16, np.float32))
    bridge.publish_command(cmd)
    assert len(env.published) == 1
    assert env.published[0].shape == (16,)
