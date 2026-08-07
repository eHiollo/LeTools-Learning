from __future__ import annotations

import numpy as np
import pytest

from kuavo_rl.rl100_zarr.ros_depth import (
    CameraSyncError,
    DepthPointCloudHub,
    TimedDepthFrame,
    depth_subscriber_kwargs,
    select_synchronized_frames,
)
from kuavo_rl.rl100_zarr.config import CameraPCConfig, RL100CollectConfig


def test_depth_subscriber_can_buffer_multiple_full_resolution_frames():
    kwargs = depth_subscriber_kwargs()
    wrist_depth_bytes = 848 * 480 * 2
    assert kwargs["queue_size"] == 1
    assert kwargs["buff_size"] >= 2 * wrist_depth_bytes


def _frame(
    camera: str,
    stamp: float,
    received: float,
    *,
    sequence: int = 1,
) -> TimedDepthFrame:
    return TimedDepthFrame(
        camera_name=camera,
        depth_m=np.ones((2, 2), dtype=np.float32),
        intrinsics=(10.0, 10.0, 1.0, 1.0),
        frame_id=f"{camera}_frame",
        header_stamp_s=stamp,
        received_wall_s=1000.0 + received,
        received_monotonic_s=received,
        sequence=sequence,
    )


def _histories(*frames: TimedDepthFrame) -> dict[str, list[TimedDepthFrame]]:
    result: dict[str, list[TimedDepthFrame]] = {}
    for frame in frames:
        result.setdefault(frame.camera_name, []).append(frame)
    return result


def test_selector_chooses_nearest_frame_before_cutoff():
    bundle = select_synchronized_frames(
        _histories(
            _frame("head", 10.000, 1.0),
            _frame("head", 10.033, 1.1),
            _frame("left", 10.030, 1.1),
            _frame("right", 10.040, 1.1),
        ),
        cutoff_monotonic_s=1.2,
        reference_camera="head",
        max_received_age_s=1.0,
        max_header_skew_s=0.05,
        max_receive_skew_s=0.2,
        now_monotonic_s=1.2,
    )
    assert bundle.reference_stamp_s == pytest.approx(10.033)
    assert bundle.camera_header_stamps == {
        "head": pytest.approx(10.033),
        "left": pytest.approx(10.030),
        "right": pytest.approx(10.040),
    }


def test_selector_never_uses_frame_received_after_cutoff():
    with pytest.raises(CameraSyncError, match="no fresh frame"):
        select_synchronized_frames(
            _histories(
                _frame("head", 10.000, 1.0),
                _frame("left", 10.000, 1.3),
            ),
            cutoff_monotonic_s=1.2,
            reference_camera="head",
            max_received_age_s=1.0,
            max_header_skew_s=0.05,
            now_monotonic_s=1.2,
        )


def test_selector_tie_breaks_to_earlier_header():
    bundle = select_synchronized_frames(
        _histories(
            _frame("head", 10.000, 1.0),
            _frame("left", 9.990, 1.1),
            _frame("left", 10.010, 1.1),
        ),
        cutoff_monotonic_s=1.2,
        reference_camera="head",
        max_received_age_s=1.0,
        max_header_skew_s=0.05,
        now_monotonic_s=1.2,
    )
    assert bundle.camera_header_stamps["left"] == pytest.approx(9.990)


def test_selector_rejects_header_skew():
    with pytest.raises(CameraSyncError) as error:
        select_synchronized_frames(
            _histories(_frame("head", 10.0, 1.0), _frame("left", 10.2, 1.0)),
            cutoff_monotonic_s=1.1,
            reference_camera="head",
            max_received_age_s=1.0,
            max_header_skew_s=0.1,
            now_monotonic_s=1.1,
        )
    assert error.value.code == "header_skew"


def test_selector_rejects_receive_skew_separately():
    with pytest.raises(CameraSyncError) as error:
        select_synchronized_frames(
            _histories(_frame("head", 10.0, 1.0), _frame("left", 10.01, 1.5)),
            cutoff_monotonic_s=1.6,
            reference_camera="head",
            max_received_age_s=1.0,
            max_header_skew_s=0.1,
            max_receive_skew_s=0.2,
            now_monotonic_s=1.6,
        )
    assert error.value.code == "receive_skew"


def test_selector_rejects_stale_reference():
    with pytest.raises(CameraSyncError) as error:
        select_synchronized_frames(
            _histories(_frame("head", 10.0, 1.0), _frame("left", 10.01, 1.0)),
            cutoff_monotonic_s=2.0,
            reference_camera="head",
            max_received_age_s=0.2,
            max_header_skew_s=0.1,
            now_monotonic_s=2.0,
        )
    assert error.value.code == "reference_stale"


def test_selector_rejects_missing_intrinsics():
    invalid = _frame("left", 10.0, 1.0)
    invalid = TimedDepthFrame(
        camera_name=invalid.camera_name,
        depth_m=invalid.depth_m,
        intrinsics=None,
        frame_id=invalid.frame_id,
        header_stamp_s=invalid.header_stamp_s,
        received_wall_s=invalid.received_wall_s,
        received_monotonic_s=invalid.received_monotonic_s,
    )
    with pytest.raises(CameraSyncError):
        select_synchronized_frames(
            _histories(_frame("head", 10.0, 1.0), invalid),
            cutoff_monotonic_s=1.1,
            reference_camera="head",
            max_received_age_s=1.0,
            max_header_skew_s=0.1,
            now_monotonic_s=1.1,
        )


def test_hub_rejects_backwards_header_and_keeps_previous_frame():
    cfg = RL100CollectConfig(
        cameras=[CameraPCConfig("head_cam_h", "/d", "/i", frame_id="head")],
        camera_reference_camera="head_cam_h",
    )
    hub = DepthPointCloudHub(cfg)

    class Header:
        def __init__(self, stamp: float):
            self.stamp = type("Stamp", (), {"to_sec": lambda self: stamp})()
            self.frame_id = "head"

    class Message:
        def __init__(self, stamp: float):
            self.header = Header(stamp)

    hub._on_info("head_cam_h", type("Info", (), {"K": [10, 0, 1, 0, 10, 1, 0, 0, 1], "header": Header(1.0)})())
    hub._store_depth("head_cam_h", np.ones((2, 2), dtype=np.float32), Message(10.0))
    hub._store_depth("head_cam_h", np.ones((2, 2), dtype=np.float32), Message(9.0))
    assert hub.invalid_counts["head_cam_h_header_backwards"] == 1
    assert len(hub._histories["head_cam_h"]) == 1


def test_hub_tf_query_receives_image_stamp(monkeypatch):
    cfg = RL100CollectConfig(
        cameras=[CameraPCConfig("head_cam_h", "/d", "/i", frame_id="head")],
        camera_reference_camera="head_cam_h",
        camera_tf_at_image_stamp=True,
    )
    hub = DepthPointCloudHub(cfg)
    calls = []
    monkeypatch.setattr(
        hub,
        "_lookup_T_base_cam",
        lambda frame_id, stamp_s=None: calls.append((frame_id, stamp_s)) or np.eye(4, dtype=np.float32),
    )
    hub._on_info("head_cam_h", type("Info", (), {"K": [10, 0, 1, 0, 10, 1, 0, 0, 1], "header": type("H", (), {"frame_id": "head"})()})())
    class Stamp:
        def to_sec(self):
            return 10.0
    message = type("Message", (), {"header": type("Header", (), {"stamp": Stamp(), "frame_id": "head"})()})()
    hub._store_depth("head_cam_h", np.ones((20, 20), dtype=np.float32), message)
    hub.get_point_cloud_sample(max_received_age_s=1.0, max_header_skew_s=0.1)
    assert calls == [("head", pytest.approx(10.0))]
