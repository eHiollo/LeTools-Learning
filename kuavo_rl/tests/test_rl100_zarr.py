"""Unit tests for RL-100 zarr helpers (no ROS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kuavo_rl.contracts import ACTION_DIM, STATE_DIM
from kuavo_rl.rl100_zarr.episode import append_labeled_episode
from kuavo_rl.rl100_zarr.live_collect import episode_action_quality_errors
from kuavo_rl.rl100_zarr.pointcloud import (
    build_rl100_point_cloud,
    depth_to_point_cloud,
    downsample_fps,
)
from kuavo_rl.rl100_zarr.reward import assign_episode_rewards, should_keep_episode
from kuavo_rl.rl100_zarr.schema import NUM_POINTS, ZarrEpisodeBuffers
from kuavo_rl.rl100_zarr.staging import build_zarr_from_episode_dir, save_episode_npz
from kuavo_rl.rl100_zarr.writer import write_rl100_zarr


def test_assign_episode_rewards_success_vs_failure():
    actions = np.zeros((10, ACTION_DIM), dtype=np.float32)
    actions[5:] = 1.0
    r_ok = assign_episode_rewards(actions, is_success=True, lambda_penalty=0.05, smooth_penalty=0.0)
    r_fail = assign_episode_rewards(actions, is_success=False, lambda_penalty=0.05, smooth_penalty=0.0)
    assert r_ok[-1] > 0.9
    assert r_fail[-1] == 0.0
    assert np.allclose(r_ok[:-1], 0.0)


def test_should_keep_episode_filter():
    assert should_keep_episode("success", only_success=False)
    assert should_keep_episode("failure", only_success=False)
    assert not should_keep_episode("failure", only_success=True)
    assert not should_keep_episode("abort", only_success=False)


def test_label_from_teleop_info_reads_env_step_events():
    """live_collect must label from env info — not a second teleop.poll()."""
    from kuavo_rl.rl100_zarr.live_collect import _label_from_teleop_info

    assert _label_from_teleop_info({"teleop_events": {"success": True}}) == (
        "success",
        "success_button",
    )
    assert _label_from_teleop_info({"teleop_events": {"failure": True}}) == (
        "failure",
        "failure_button",
    )
    assert _label_from_teleop_info({"reward_source": "manual_success"}) == (
        "success",
        "success_button",
    )
    assert _label_from_teleop_info({"teleop_events": {}, "reward_source": "step_zero"}) is None


def test_depth_to_pc_and_downsample():
    depth = np.full((48, 64), 1.0, dtype=np.float32)
    fx = fy = 100.0
    cx, cy = 32.0, 24.0
    pc = depth_to_point_cloud(depth, (fx, fy, cx, cy), np.eye(4, dtype=np.float32))
    assert pc.ndim == 2 and pc.shape[1] == 3
    assert pc.shape[0] > 100
    out = downsample_fps(pc, NUM_POINTS)
    assert out.shape == (NUM_POINTS, 3)


def test_fuse_three_cams_to_rl100_shape():
    rng = np.random.default_rng(1)
    clouds = [rng.normal(size=(2000, 3)).astype(np.float32) for _ in range(3)]
    out = build_rl100_point_cloud(clouds, num_points=NUM_POINTS)
    assert out.shape == (NUM_POINTS, 3)


def test_empty_pointcloud_hard_fail():
    with pytest.raises(RuntimeError, match="empty after workspace crop"):
        build_rl100_point_cloud(
            [np.zeros((0, 3), dtype=np.float32)],
            num_points=NUM_POINTS,
            raise_on_empty=True,
            min_points=32,
        )


def test_default_config_aligns_real_collect():
    from kuavo_rl.rl100_zarr.config import RL100CollectConfig, load_rl100_collect_config

    cfg = RL100CollectConfig()
    assert cfg.deploy_config.endswith("deploy_total.yaml")
    assert cfg.env_config.endswith("kuavo_hilserl_real_mvp.yaml")
    assert cfg.require_all_cameras is True
    assert cfg.fail_on_empty_pointcloud is True
    assert cfg.require_command_action is True
    assert all(c.depth_msg_type == "compressed_depth" for c in cfg.cameras)
    assert all("compressedDepth" in c.depth_topic for c in cfg.cameras)

    yaml_cfg = load_rl100_collect_config("configs/rl/rl100_zarr_collect.yaml")
    assert yaml_cfg.task == "box_to_chest_v1"
    assert "deploy_total.yaml" in yaml_cfg.deploy_config
    assert yaml_cfg.live_max_steps >= 100_000

    upper_cfg = load_rl100_collect_config(
        "configs/rl/rl100_zarr_collect_upper_cams.yaml"
    )
    assert upper_cfg.task == "grasp_8_4_v2"
    assert upper_cfg.only_success is True
    assert upper_cfg.hand_command_topic == "/control_robot_hand_position"
    assert upper_cfg.require_gripper_motion is True


def test_episode_action_quality_rejects_hold_state_and_constant_gripper():
    states = [np.zeros(STATE_DIM, dtype=np.float32) for _ in range(4)]
    actions = [s.copy() for s in states]
    errors = episode_action_quality_errors(
        states,
        actions,
        min_arm_action_state_delta_rad=1e-4,
        require_gripper_motion=True,
        min_gripper_action_range=0.05,
    )
    assert any("indistinguishable" in error for error in errors)
    assert any("no effective gripper" in error for error in errors)


def test_episode_action_quality_accepts_real_arm_and_gripper_commands():
    states = [np.zeros(STATE_DIM, dtype=np.float32) for _ in range(4)]
    actions = [s.copy() for s in states]
    for i, action in enumerate(actions):
        action[0] = 0.01 * (i + 1)
        action[7] = i / 3
    errors = episode_action_quality_errors(
        states,
        actions,
        min_arm_action_state_delta_rad=1e-4,
        require_gripper_motion=True,
        min_gripper_action_range=0.05,
    )
    assert errors == []


def test_write_zarr_roundtrip(tmp_path: Path):
    pytest.importorskip("zarr")
    buffers = ZarrEpisodeBuffers()
    rng = np.random.default_rng(2)
    states = [rng.normal(size=(STATE_DIM,)).astype(np.float32) for _ in range(5)]
    actions = [rng.normal(size=(ACTION_DIM,)).astype(np.float32) for _ in range(5)]
    pcs = [downsample_fps(rng.normal(size=(3000, 3)).astype(np.float32)) for _ in range(5)]
    assert append_labeled_episode(
        buffers,
        states=states,
        actions=actions,
        point_clouds=pcs,
        result_type="success",
        smooth_penalty=0.0,
    )
    # failure episode also kept
    assert append_labeled_episode(
        buffers,
        states=states,
        actions=actions,
        point_clouds=pcs,
        result_type="failure",
        smooth_penalty=0.0,
    )
    zpath = tmp_path / "demo.zarr"
    write_rl100_zarr(buffers, zpath, overwrite=True, attrs={"task": "unit"})
    import zarr

    root = zarr.open(str(zpath), mode="r")
    assert root["data"]["state"].shape == (10, STATE_DIM)
    assert root["data"]["action"].shape == (10, ACTION_DIM)
    assert root["data"]["point_cloud"].shape == (10, NUM_POINTS, 3)
    assert list(root["meta"]["episode_ends"][:]) == [5, 10]


def test_staging_build_only_success(tmp_path: Path):
    pytest.importorskip("zarr")
    ep_dir = tmp_path / "episodes"
    rng = np.random.default_rng(3)
    for i, result in enumerate(["success", "failure"]):
        t = 4
        save_episode_npz(
            ep_dir / f"{i}_{result}.npz",
            states=[rng.normal(size=(STATE_DIM,)).astype(np.float32) for _ in range(t)],
            actions=[rng.normal(size=(ACTION_DIM,)).astype(np.float32) for _ in range(t)],
            point_clouds=[
                downsample_fps(rng.normal(size=(1000, 3)).astype(np.float32)) for _ in range(t)
            ],
            result_type=result,
        )
    report = build_zarr_from_episode_dir(
        ep_dir,
        tmp_path / "only_ok.zarr",
        only_success=True,
        overwrite=True,
        smooth_penalty=0.0,
    )
    assert report["episodes_kept"] == 1
    assert report["episodes_skipped"] == 1
    assert report["n_episodes"] == 1
