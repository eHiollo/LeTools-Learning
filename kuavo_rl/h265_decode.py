"""Decode KuavoBrain v3 ``sensor_msgs/CompressedImage`` H.265 (Annex-B) streams.

v3 production publishes ``format=h265`` (or ``h265; …``) on
``/cam_*/color/h265_stream``. OpenCV ``imdecode`` cannot read these; use PyAV.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def is_h265_compressed(format_str: str | None) -> bool:
    fmt = (format_str or "").lower()
    return "h265" in fmt or "hevc" in fmt


class H265StreamDecoder:
    """Stateful HEVC decoder for one ROS topic stream."""

    def __init__(self) -> None:
        import av

        self._av = av
        self._codec = av.CodecContext.create("hevc", "r")
        self._last_rgb: Optional[np.ndarray] = None

    def decode(self, data: bytes) -> Optional[np.ndarray]:
        """Feed one CompressedImage payload; return latest RGB frame or last good."""
        if not data:
            return self._last_rgb
        try:
            packets = self._codec.parse(data)
            frames = []
            for packet in packets:
                frames.extend(self._codec.decode(packet))
            if frames:
                rgb = frames[-1].to_ndarray(format="rgb24")
                self._last_rgb = rgb
                return rgb
        except Exception:  # noqa: BLE001 — wait for SPS/IDR; keep last frame
            return self._last_rgb
        return self._last_rgb

    def reset(self) -> None:
        import av

        self._codec = av.CodecContext.create("hevc", "r")
        self._last_rgb = None
