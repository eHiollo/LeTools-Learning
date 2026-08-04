"""Inference-only loader for RL-100 workspace checkpoints.

This module intentionally does not import ROS.  A workspace checkpoint contains
the resolved Hydra policy configuration and the fitted normalizer, so deployment
must construct the policy from the checkpoint rather than from a second YAML.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from kuavo_rl.contracts import ACTION_DIM, STATE_DIM

ModelSource = Literal["auto", "ema_model", "model"]


@dataclass(frozen=True)
class RL100CheckpointInfo:
    checkpoint_path: Path
    checkpoint_sha256: str
    state_dict_key: str
    n_obs_steps: int
    n_action_steps: int
    horizon: int
    point_count: int
    point_dim: int
    state_dim: int
    action_dim: int
    scheduler_type: str
    use_cm: bool


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _shape(meta: Any, path: tuple[str, ...]) -> tuple[int, ...]:
    value = meta
    for key in path:
        value = _get(value, key)
        if value is None:
            raise ValueError(f"checkpoint shape_meta missing {'.'.join(path)}")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"checkpoint shape {'.'.join(path)} must be a list, got {value!r}")
    return tuple(int(v) for v in value)


def validate_shape_meta(shape_meta: Any) -> tuple[int, int, int, int]:
    """Validate the frozen RL-100 real-robot observation/action contract."""
    point_shape = _shape(shape_meta, ("obs", "point_cloud", "shape"))
    state_shape = _shape(shape_meta, ("obs", "agent_pos", "shape"))
    action_shape = _shape(shape_meta, ("action", "shape"))
    if point_shape != (1024, 3):
        raise ValueError(f"point_cloud shape {point_shape} != (1024, 3)")
    if state_shape != (STATE_DIM,):
        raise ValueError(f"agent_pos shape {state_shape} != ({STATE_DIM},)")
    if action_shape != (ACTION_DIM,):
        raise ValueError(f"action shape {action_shape} != ({ACTION_DIM},)")
    return point_shape[0], point_shape[1], state_shape[0], action_shape[0]


def select_state_dict_key(
    state_dicts: dict[str, Any], cfg: Any, model_source: ModelSource = "auto"
) -> str:
    """Choose EMA only when it was trained and exists in this checkpoint."""
    if model_source not in {"auto", "ema_model", "model"}:
        raise ValueError(f"unknown model_source={model_source!r}")
    if not isinstance(state_dicts, dict):
        raise ValueError("checkpoint state_dicts must be a mapping")
    use_ema = bool(_get(_get(cfg, "training", {}), "use_ema", False))
    if model_source == "ema_model":
        if "ema_model" not in state_dicts:
            raise ValueError("EMA model requested but checkpoint has no state_dicts.ema_model")
        return "ema_model"
    if model_source == "auto" and use_ema and "ema_model" in state_dicts:
        return "ema_model"
    if "model" not in state_dicts:
        raise ValueError("checkpoint has no state_dicts.model")
    return "model"


def normalize_module_prefix(state_dict: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Remove DDP's ``module.`` prefix only when every key has it."""
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("selected state dict must be a non-empty mapping")
    keys = list(state_dict)
    if all(str(key).startswith("module.") for key in keys):
        return {str(key)[7:]: value for key, value in state_dict.items()}, True
    return state_dict, False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RL100Policy:
    """A loaded RL-100 policy returning unnormalized absolute 16-D actions."""

    def __init__(
        self,
        policy: Any,
        info: RL100CheckpointInfo,
        *,
        device: str,
        deterministic: bool = False,
        distill2mean: bool = False,
        use_cm: bool | None = None,
    ) -> None:
        self.policy = policy
        self.info = info
        self.device = device
        self.deterministic = bool(deterministic)
        self.distill2mean = bool(distill2mean)
        self.use_cm = info.use_cm if use_cm is None else bool(use_cm)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cuda:0",
        model_source: ModelSource = "auto",
        *,
        deterministic: bool = False,
        distill2mean: bool = False,
        use_cm: bool | None = None,
    ) -> "RL100Policy":
        path = Path(checkpoint_path).expanduser().resolve()
        if path.suffix != ".ckpt":
            raise ValueError(f"RL-100 deploy accepts workspace .ckpt only, got {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            import dill
            import hydra
            import torch
        except ImportError as exc:  # pragma: no cover - depends on RL-100 environment
            raise RuntimeError("RL-100 deployment requires torch, dill and hydra-core") from exc

        # Workspace checkpoints contain Hydra config objects, therefore PyTorch's
        # newer weights-only default is not sufficient here.
        try:
            payload = torch.load(
                path.open("rb"), pickle_module=dill, map_location="cpu", weights_only=False
            )
        except TypeError:  # PyTorch < 2.0
            payload = torch.load(path.open("rb"), pickle_module=dill, map_location="cpu")
        if not isinstance(payload, dict) or "cfg" not in payload or "state_dicts" not in payload:
            raise ValueError("not an RL-100 workspace checkpoint: expected cfg and state_dicts")
        cfg = payload["cfg"]
        policy_cfg = _get(cfg, "policy")
        if policy_cfg is None:
            raise ValueError("checkpoint cfg has no policy")
        shape_meta = _get(cfg, "shape_meta", _get(policy_cfg, "shape_meta"))
        point_count, point_dim, state_dim, action_dim = validate_shape_meta(shape_meta)
        n_obs_steps = int(_get(cfg, "n_obs_steps", _get(policy_cfg, "n_obs_steps", 0)))
        n_action_steps = int(_get(cfg, "n_action_steps", _get(policy_cfg, "n_action_steps", 0)))
        horizon = int(_get(cfg, "horizon", _get(policy_cfg, "horizon", 0)))
        if n_obs_steps < 1 or n_action_steps < 1 or horizon < 1:
            raise ValueError("checkpoint has invalid horizon/n_obs_steps/n_action_steps")
        if n_action_steps > horizon:
            raise ValueError("checkpoint n_action_steps exceeds horizon")

        state_key = select_state_dict_key(payload["state_dicts"], cfg, model_source)
        state_dict, stripped_prefix = normalize_module_prefix(payload["state_dicts"][state_key])
        policy = hydra.utils.instantiate(policy_cfg)
        incompatible = policy.load_state_dict(state_dict, strict=True)
        if getattr(incompatible, "missing_keys", ()) or getattr(incompatible, "unexpected_keys", ()):
            raise RuntimeError("strict RL-100 checkpoint load reported incompatible keys")
        policy.to(torch.device(device))
        policy.eval()
        info = RL100CheckpointInfo(
            checkpoint_path=path,
            checkpoint_sha256=sha256_file(path),
            state_dict_key=state_key + (" (module-prefix-stripped)" if stripped_prefix else ""),
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            horizon=horizon,
            point_count=point_count,
            point_dim=point_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            scheduler_type=str(_get(policy_cfg, "scheduler_type", "unknown")),
            use_cm=bool(_get(policy_cfg, "use_cm", _get(policy, "use_cm", False))),
        )
        return cls(
            policy,
            info,
            device=str(device),
            deterministic=deterministic,
            distill2mean=distill2mean,
            use_cm=use_cm,
        )

    def predict(self, point_cloud_history: np.ndarray, state_history: np.ndarray) -> np.ndarray:
        """Predict an action chunk from unbatched observation histories."""
        points = np.asarray(point_cloud_history, dtype=np.float32)
        states = np.asarray(state_history, dtype=np.float32)
        expected_points = (self.info.n_obs_steps, self.info.point_count, self.info.point_dim)
        expected_states = (self.info.n_obs_steps, self.info.state_dim)
        if points.shape != expected_points:
            raise ValueError(f"point_cloud history shape {points.shape} != {expected_points}")
        if states.shape != expected_states:
            raise ValueError(f"agent_pos history shape {states.shape} != {expected_states}")
        if not np.isfinite(points).all() or not np.isfinite(states).all():
            raise ValueError("non-finite RL-100 observation")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torch is required for RL-100 inference") from exc
        obs = {
            "point_cloud": torch.from_numpy(points[None]).to(self.device),
            "agent_pos": torch.from_numpy(states[None]).to(self.device),
        }
        with torch.inference_mode():
            result = self.policy.predict_action(
                obs,
                deterministic=self.deterministic,
                distill2mean=self.distill2mean,
                use_cm=self.use_cm,
            )
        if "action" not in result:
            raise RuntimeError("RL-100 policy returned no action")
        action = result["action"].detach().float().cpu().numpy()
        if action.shape != (1, self.info.n_action_steps, ACTION_DIM):
            raise RuntimeError(
                f"RL-100 action shape {action.shape} != (1, {self.info.n_action_steps}, {ACTION_DIM})"
            )
        action = action[0].astype(np.float32, copy=False)
        if not np.isfinite(action).all():
            raise RuntimeError("RL-100 policy produced NaN/Inf")
        return action

    def warmup(self, runs: int = 1) -> dict[str, Any]:
        if runs < 1:
            raise ValueError("warmup runs must be >= 1")
        points = np.zeros(
            (self.info.n_obs_steps, self.info.point_count, self.info.point_dim), dtype=np.float32
        )
        states = np.zeros((self.info.n_obs_steps, self.info.state_dim), dtype=np.float32)
        import time

        durations: list[float] = []
        for _ in range(runs):
            started = time.monotonic()
            action = self.predict(points, states)
            durations.append(time.monotonic() - started)
        return {
            "runs": runs,
            "action_shape": list(action.shape),
            "all_finite": bool(np.isfinite(action).all()),
            "latency_s": durations,
        }
