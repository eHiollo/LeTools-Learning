#!/usr/bin/env python3
"""Driver: call KuavoBrain CvtRosbag2Lerobot in-process (alignment untouched).

Run only via kuavo_rl.brain_lerobot_export (sets PYTHONPATH to challenge + its lerobot).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _apply_sim_camera_aliases(kuavo_mod) -> None:
    """Point Brain reader camera keys at HIL/sim Image topics; keep Brain decode/align."""
    from kuavo_rl.brain_lerobot_export import SIM_CAMERA_TOPIC_ALIASES

    reader_cls = kuavo_mod.KuavoRosbagReader
    orig_init = reader_cls.__init__

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for cam_key, topic in SIM_CAMERA_TOPIC_ALIASES.items():
            if cam_key in self._topic_process_map:
                self._topic_process_map[cam_key]["topic"] = topic
                # Keep Brain's process_color_image (supports raw Image + CompressedImage).

    reader_cls.__init__ = patched_init  # type: ignore[method-assign]


def _apply_brain_h265_camera_aliases(kuavo_mod) -> None:
    """v3 production bags: /cam_*/color/h265_stream + PyAV HEVC decode."""
    import cv2
    import numpy as np
    from kuavo_rl.brain_lerobot_export import BRAIN_CAMERA_TOPICS
    from kuavo_rl.h265_decode import H265StreamDecoder, is_h265_compressed

    reader_cls = kuavo_mod.KuavoRosbagReader
    orig_init = reader_cls.__init__
    resize_w = int(getattr(kuavo_mod, "RESIZE_W", 848))
    resize_h = int(getattr(kuavo_mod, "RESIZE_H", 480))

    def patched_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for cam_key, topic in BRAIN_CAMERA_TOPICS.items():
            if cam_key not in self._topic_process_map:
                continue
            decoder = H265StreamDecoder()

            def _make_fn(dec: H265StreamDecoder):
                def process_fn(msg):
                    fmt = getattr(msg, "format", None)
                    if is_h265_compressed(fmt):
                        rgb = dec.decode(bytes(msg.data))
                        if rgb is None:
                            rgb = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
                        else:
                            rgb = cv2.resize(rgb, (resize_w, resize_h))
                        return {
                            "data": rgb,
                            "timestamp": msg.header.stamp.to_sec(),
                        }
                    if hasattr(msg, "encoding"):
                        return kuavo_mod.KuavoMsgProcesser.process_color_image(msg)
                    # CompressedImage without format: try JPEG then H.265
                    try:
                        return kuavo_mod.KuavoMsgProcesser.process_color_image(msg)
                    except Exception:  # noqa: BLE001
                        rgb = dec.decode(bytes(msg.data))
                        if rgb is None:
                            rgb = np.zeros((resize_h, resize_w, 3), dtype=np.uint8)
                        else:
                            rgb = cv2.resize(rgb, (resize_w, resize_h))
                        return {
                            "data": rgb,
                            "timestamp": msg.header.stamp.to_sec(),
                        }

                return process_fn

            self._topic_process_map[cam_key]["topic"] = topic
            self._topic_process_map[cam_key]["msg_process_fn"] = _make_fn(decoder)

    reader_cls.__init__ = patched_init  # type: ignore[method-assign]


def _apply_missing_stream_zero_fill(kuavo_mod) -> None:
    """Keep LeRobot schema: missing arm/claw streams → zeros (do not drop frames).

    Brain's CvtRosbag2Lerobot.on_frame early-returns when arm_traj / eef are absent.
    Alignment only zero-fills when a topic has *some* messages; entirely missing
    topics stay None. Wrap the chunked callback to inject fixed-dim zeros.
    """
    import numpy as np
    from kuavo_data.common.config_platform import get_arm_joint_slice

    reader_cls = kuavo_mod.KuavoRosbagReader
    orig = reader_cls.process_rosbag_chunked

    def patched(self, bag_file, frame_callback, *args, **kwargs):
        arm_start, arm_end = get_arm_joint_slice(kuavo_mod.PLATFORM_TYPE)
        arm_dim = max(0, int(arm_end) - int(arm_start))
        claw_dim = len(kuavo_mod.DEFAULT_LEJUCLAW_JOINT_NAMES)
        dex_dim = len(kuavo_mod.DEFAULT_DEXHAND_JOINT_NAMES)

        def _ensure(frame: dict, key: str, dim: int) -> None:
            if dim <= 0:
                return
            item = frame.get(key)
            data = None if item is None else item.get("data")
            if data is None or (hasattr(data, "__len__") and len(data) == 0):
                ts = 0.0 if item is None else item.get("timestamp", 0.0)
                frame[key] = {
                    "data": np.zeros(dim, dtype=np.float32),
                    "timestamp": ts,
                }

        def wrapped_cb(aligned_frame: dict, frame_idx: int):
            _ensure(aligned_frame, "action.kuavo_arm_traj", arm_dim)
            _ensure(aligned_frame, "action.kuavo_arm_traj_alt", arm_dim)
            if kuavo_mod.USE_LEJU_CLAW:
                _ensure(aligned_frame, "observation.claw", claw_dim)
                _ensure(aligned_frame, "action.claw", claw_dim)
            if kuavo_mod.USE_QIANGNAO:
                _ensure(aligned_frame, "observation.qiangnao", dex_dim)
                _ensure(aligned_frame, "action.qiangnao", dex_dim)
            frame_callback(aligned_frame, frame_idx)

        return orig(self, bag_file, wrapped_cb, *args, **kwargs)

    reader_cls.process_rosbag_chunked = patched  # type: ignore[method-assign]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag-dir", required=True, type=Path)
    ap.add_argument("--lerobot-dir-name", required=True)
    ap.add_argument("--config-dir", required=True, type=Path)
    ap.add_argument("--config-name", default="KuavoRosbag2Lerobot_hil")
    ap.add_argument("--topic-profile", choices=["sim", "brain"], default="sim")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    # Challenge root must be on sys.path (export sets PYTHONPATH).
    from omegaconf import OmegaConf

    cfg_path = args.config_dir / f"{args.config_name}.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: missing config {cfg_path}", file=sys.stderr)
        return 2
    cfg = OmegaConf.load(cfg_path)
    if args.override:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(args.override)))

    # Absolute path required: this process may run with cwd=challenge root.
    bag_dir_abs = args.bag_dir if args.bag_dir.is_absolute() else args.bag_dir.resolve()
    cfg.rosbag.rosbag_dir = str(bag_dir_abs)
    cfg.rosbag.lerobot_dir = str(args.lerobot_dir_name)

    from kuavo_data.common import kuavo_dataset as kuavo
    from kuavo_data.CvtRosbag2Lerobot import (
        setup_logging,
        port_kuavo_rosbag_chunked,
    )
    from kuavo_data.common.config_platform import get_arm_head_start

    setup_logging()
    kuavo.init_parameters(cfg)

    if args.topic_profile == "sim":
        _apply_sim_camera_aliases(kuavo)
    elif args.topic_profile == "brain":
        _apply_brain_h265_camera_aliases(kuavo)
    _apply_missing_stream_zero_fill(kuavo)

    # Mirror CvtRosbag2Lerobot.main joint-name construction (required globals).
    import kuavo_data.CvtRosbag2Lerobot as cvt

    half_arm = len(kuavo.DEFAULT_ARM_JOINT_NAMES) // 2
    half_claw = len(kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES) // 2
    half_dexhand = len(kuavo.DEFAULT_DEXHAND_JOINT_NAMES) // 2
    up_start = get_arm_head_start(kuavo.PLATFORM_TYPE)
    if kuavo.USE_LEJU_CLAW:
        default_arm = (
            kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm]
            + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[:half_claw]
            + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:]
            + kuavo.DEFAULT_LEJUCLAW_JOINT_NAMES[half_claw:]
        )
        arm_slice = [
            (
                kuavo.SLICE_ROBOT[0][0] - up_start,
                kuavo.SLICE_ROBOT[0][-1] - up_start,
            ),
            (
                kuavo.SLICE_CLAW[0][0] + half_arm,
                kuavo.SLICE_CLAW[0][-1] + half_arm,
            ),
            (
                kuavo.SLICE_ROBOT[1][0] - up_start + half_claw,
                kuavo.SLICE_ROBOT[1][-1] - up_start + half_claw,
            ),
            (
                kuavo.SLICE_CLAW[1][0] + half_arm * 2,
                kuavo.SLICE_CLAW[1][-1] + half_arm * 2,
            ),
        ]
    elif kuavo.USE_QIANGNAO:
        default_arm = (
            kuavo.DEFAULT_ARM_JOINT_NAMES[:half_arm]
            + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[:half_dexhand]
            + kuavo.DEFAULT_ARM_JOINT_NAMES[half_arm:]
            + kuavo.DEFAULT_DEXHAND_JOINT_NAMES[half_dexhand:]
        )
        arm_slice = [
            (
                kuavo.SLICE_ROBOT[0][0] - up_start,
                kuavo.SLICE_ROBOT[0][-1] - up_start,
            ),
            (
                kuavo.SLICE_DEX[0][0] + half_arm,
                kuavo.SLICE_DEX[0][-1] + half_arm,
            ),
            (
                kuavo.SLICE_ROBOT[1][0] - up_start + half_dexhand,
                kuavo.SLICE_ROBOT[1][-1] - up_start + half_dexhand,
            ),
            (
                kuavo.SLICE_DEX[1][0] + half_arm * 2,
                kuavo.SLICE_DEX[1][-1] + half_arm * 2,
            ),
        ]
    else:
        raise RuntimeError("HIL export expects leju_claw or qiangnao eef_type")

    cvt.DEFAULT_JOINT_NAMES_LIST = [
        default_arm[k] for l, r in arm_slice for k in range(l, r)
    ]

    # Brain Cvt logs len(raw_dir); pass str (not Path) to match Hydra main().
    raw_dir = str(Path(cfg.rosbag.rosbag_dir).resolve())
    version = cfg.rosbag.lerobot_dir
    task_name = os.path.basename(raw_dir.rstrip(os.sep))
    repo_id = f"lerobot/{task_name}"
    # Batch layout: <batch_bags>/*.bag → <batch_bags>/../<version>/lerobot
    lerobot_dir = os.path.abspath(os.path.join(raw_dir, "..", str(version), "lerobot"))
    if os.path.exists(lerobot_dir):
        shutil.rmtree(lerobot_dir)

    chunk_size = int(OmegaConf.select(cfg, "rosbag.chunk_size", default=100))
    port_kuavo_rosbag_chunked(
        raw_dir=raw_dir,
        repo_id=repo_id,
        task=kuavo.TASK_DESCRIPTION,
        mode="video",
        root=lerobot_dir,
        n=cfg.rosbag.num_used,
        chunk_size=chunk_size,
        platform_type=kuavo.PLATFORM_TYPE,
    )
    print(f"[brain_cvt] ok root={lerobot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
