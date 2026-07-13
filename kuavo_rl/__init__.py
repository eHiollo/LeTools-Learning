"""Kuavo HIL-SERL adapter package (main repo, not third_party/lerobot)."""

from kuavo_rl.contracts import ACTION_DIM, ACTION_NAMES, FaultCode

__all__ = [
    "ACTION_DIM",
    "ACTION_NAMES",
    "FaultCode",
    "KuavoHILSerlEnv",
]


def __getattr__(name: str):
    if name == "KuavoHILSerlEnv":
        from kuavo_rl.env import KuavoHILSerlEnv as _Env

        return _Env
    raise AttributeError(name)
