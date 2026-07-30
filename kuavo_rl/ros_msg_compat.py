"""ROS message import helpers for mixed SDK / workspace environments."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _sdk_msg_root() -> Path:
    import kuavo_humanoid_sdk

    return Path(kuavo_humanoid_sdk.__file__).resolve().parent / "msg"


def _inject_all_missing(kind: str, live_pkg) -> None:
    """Attach every SDK-generated msg/srv class missing from the live package."""
    sdk_dir = _sdk_msg_root() / "kuavo_msgs" / kind
    if not sdk_dir.is_dir():
        return

    for mod_path in sorted(sdk_dir.glob("_*.py")):
        stem = mod_path.name[1:-3]  # _Foo.py → Foo
        # Fast path: primary symbol already present and (for srv) Request too.
        if hasattr(live_pkg, stem):
            if kind != "srv" or hasattr(live_pkg, f"{stem}Request"):
                continue

        spec_name = f"kuavo_msgs.{kind}._{stem}_sdk"
        if spec_name in sys.modules:
            mod = sys.modules[spec_name]
        else:
            spec = importlib.util.spec_from_file_location(spec_name, mod_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(spec_name, None)
                continue

        for attr, val in vars(mod).items():
            if attr.startswith("_"):
                continue
            if not hasattr(live_pkg, attr):
                setattr(live_pkg, attr, val)


def ensure_foot_pose_6d_msgs() -> None:
    """Make ``kuavo_humanoid_sdk`` importable without swapping live sensorsData.

    Older ``kuavo_ros_application`` lacks several SDK msg/srv types. Prepending
    the whole SDK ``kuavo_msgs`` package breaks ``/sensors_data_raw`` (MD5
    mismatch → obs buffer forever ``0/N``). Instead, inject only missing
    generated classes into the live workspace packages.
    """
    try:
        import kuavo_msgs.msg as live_msg
        import kuavo_msgs.srv as live_srv
    except ImportError:
        msg_root = str(_sdk_msg_root())
        if msg_root not in sys.path:
            sys.path.insert(0, msg_root)
        return

    # msgs first — some srv modules import msg types.
    _inject_all_missing("msg", live_msg)
    _inject_all_missing("srv", live_srv)


def prefer_sdk_kuavo_msgs() -> Path:
    """Deprecated alias — keeps live sensorsData, injects missing SDK types."""
    ensure_foot_pose_6d_msgs()
    return _sdk_msg_root()
