"""Lazy Robometer-4B scorer for offline calibration and async reward worker.

Never call from the 10 Hz control path synchronously. Load failures / OOM return
``ok=False`` scores instead of crashing the control process.

Weights can come from Hugging Face IDs or local dirs (ModelScope prefetch)::

  data/models/Robometer-4B
  data/models/Qwen3-VL-4B-Instruct

Env overrides: ``KUAVO_ROBOMETER_MODEL``, ``KUAVO_ROBOMETER_BASE_MODEL``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from kuavo_rl.contracts import DEFAULT_TASK_TEXT
from kuavo_rl.reward import EpisodeFrame, RobometerScore

logger = logging.getLogger(__name__)

DEFAULT_ROBOMETER_MODEL_ID = "lerobot/Robometer-4B"
DEFAULT_IMAGE_KEY = "observation.images.head_cam_h"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOCAL_ROBOMETER = _REPO_ROOT / "data" / "models" / "Robometer-4B"
_DEFAULT_LOCAL_BASE = _REPO_ROOT / "data" / "models" / "Qwen3-VL-4B-Instruct"


def resolve_model_path(explicit: str | None = None, *, kind: str = "robometer") -> str:
    """Prefer local dirs / env over hub IDs (HF often unreachable).

    Order: env path → explicit local path → default ModelScope prefetch dir →
    explicit hub id → default hub id.
    """
    if kind == "base":
        env = os.environ.get("KUAVO_ROBOMETER_BASE_MODEL")
        local = _DEFAULT_LOCAL_BASE
        fallback = "Qwen/Qwen3-VL-4B-Instruct"
    else:
        env = os.environ.get("KUAVO_ROBOMETER_MODEL")
        local = _DEFAULT_LOCAL_ROBOMETER
        fallback = DEFAULT_ROBOMETER_MODEL_ID

    def _as_local_dir(candidate: str | None) -> str | None:
        if not candidate:
            return None
        p = Path(candidate)
        if p.is_dir():
            try:
                if any(p.iterdir()):
                    return str(p.resolve())
            except OSError:
                return None
        if p.is_file():
            return str(p.resolve())
        return None

    for candidate in (env, explicit, str(local)):
        resolved = _as_local_dir(candidate)
        if resolved:
            return resolved

    # Hub ids only if no local snapshot exists.
    for candidate in (explicit, fallback):
        if not candidate:
            continue
        p = Path(candidate)
        if not p.exists() and "/" in candidate and not candidate.startswith(("/", ".")):
            return candidate
    return fallback


def _stack_frames_chw(
    frames_chw: np.ndarray | Sequence[np.ndarray],
    *,
    max_frames: int,
) -> np.ndarray:
    """Return (T,C,H,W) uint8/float array, uniformly subsampled to max_frames."""
    if isinstance(frames_chw, np.ndarray) and frames_chw.ndim == 4:
        arr = frames_chw
    else:
        seq = [np.asarray(f) for f in frames_chw]
        if not seq:
            raise ValueError("empty frame sequence")
        fixed: list[np.ndarray] = []
        for f in seq:
            if f.ndim == 3 and f.shape[-1] == 3:
                f = np.transpose(f, (2, 0, 1))
            if f.ndim != 3:
                raise ValueError(f"expected CHW/HWC frame, got shape {f.shape}")
            fixed.append(f)
        arr = np.stack(fixed, axis=0)
    t = arr.shape[0]
    if t > max_frames:
        idx = np.linspace(0, t - 1, max_frames).round().astype(np.int64)
        arr = arr[idx]
    return np.ascontiguousarray(arr)


class RobometerScorer:
    """Load Robometer once and score frame sequences → progress/success curves."""

    def __init__(
        self,
        *,
        model_id: str | None = None,
        base_model_id: str | None = None,
        device: str = "cuda",
        max_frames: int = 8,
        image_key: str = DEFAULT_IMAGE_KEY,
        default_task: str = DEFAULT_TASK_TEXT,
    ):
        self.model_id = resolve_model_path(model_id, kind="robometer")
        self.base_model_id = resolve_model_path(base_model_id, kind="base")
        self.device = device
        self.max_frames = max_frames
        self.image_key = image_key
        self.default_task = default_task
        self._model: Any = None
        self._encoder: Any = None
        self._config: Any = None
        self._load_error: str | None = None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_error is not None:
            raise RuntimeError(self._load_error)
        try:
            import torch
            from lerobot.processor import TransitionKey
            from lerobot.rewards.robometer.configuration_robometer import RobometerConfig
            from lerobot.rewards.robometer.modeling_robometer import (
                ROBOMETER_FEATURE_PREFIX,
                ROBOMETER_INPUT_KEYS,
                RobometerRewardModel,
                decode_progress_outputs,
            )
            from lerobot.rewards.robometer.processor_robometer import RobometerEncoderProcessorStep

            self._torch = torch
            self._TransitionKey = TransitionKey
            self._ROBOMETER_FEATURE_PREFIX = ROBOMETER_FEATURE_PREFIX
            self._ROBOMETER_INPUT_KEYS = ROBOMETER_INPUT_KEYS
            self._decode_progress_outputs = decode_progress_outputs

            local_only = Path(self.model_id).is_dir() and Path(self.base_model_id).is_dir()
            local_only = local_only or bool(os.environ.get("ROBOMETER_LOCAL_FILES_ONLY"))

            cfg = RobometerConfig(
                pretrained_path=self.model_id,
                device=self.device,
                max_frames=self.max_frames,
                image_key=self.image_key,
                default_task=self.default_task,
                base_model_id=self.base_model_id,
            )
            model = RobometerRewardModel.from_pretrained(
                self.model_id,
                config=cfg,
                local_files_only=local_only,
            )
            model.to(self.device).eval()
            encoder = RobometerEncoderProcessorStep(
                base_model_id=self.base_model_id,
                image_key=cfg.image_key,
                task_key=cfg.task_key,
                default_task=cfg.default_task or self.default_task,
                max_frames=self.max_frames,
                use_multi_image=cfg.use_multi_image,
                use_per_frame_progress_token=cfg.use_per_frame_progress_token,
            )
            self._model = model
            self._encoder = encoder
            self._config = cfg
            logger.info(
                "Robometer loaded model=%s base=%s device=%s",
                self.model_id,
                self.base_model_id,
                self.device,
            )
        except Exception as exc:  # noqa: BLE001
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Robometer load failed")
            raise RuntimeError(self._load_error) from exc

    def unload(self) -> None:
        self._model = None
        self._encoder = None
        self._config = None
        if hasattr(self, "_torch"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def score_chw_frames(
        self,
        frames_chw: np.ndarray | Sequence[np.ndarray],
        task_text: str | None = None,
    ) -> tuple[list[float], list[float], float]:
        """Score (T,C,H,W) or list of (C,H,W) frames.

        Returns ``(progress_curve, success_curve, latency_s)``.
        """
        self.ensure_loaded()
        t0 = time.time()
        torch = self._torch
        TransitionKey = self._TransitionKey

        arr = _stack_frames_chw(frames_chw, max_frames=self.max_frames)
        video = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)  # 1,T,C,H,W
        if float(video.max()) > 1.5:
            video = video / 255.0

        task = task_text or self.default_task
        transition = {
            TransitionKey.OBSERVATION: {self.image_key: video},
            TransitionKey.COMPLEMENTARY_DATA: {"task": task},
        }
        encoded = self._encoder(transition)
        obs = encoded[TransitionKey.OBSERVATION]
        batch = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in obs.items()
        }

        prefix = self._ROBOMETER_FEATURE_PREFIX
        inputs = {
            key: batch[f"{prefix}{key}"]
            for key in self._ROBOMETER_INPUT_KEYS
            if f"{prefix}{key}" in batch
        }
        inputs = dict(inputs)

        with torch.no_grad():
            progress_logits, success_logits = self._model._compute_rbm_logits(inputs)
        decoded = self._decode_progress_outputs(
            progress_logits,
            success_logits,
            is_discrete_mode=self._config.use_discrete_progress,
        )
        progress = [float(np.clip(x, 0.0, 1.0)) for x in decoded["progress_pred"][0]]
        success = [float(x) for x in decoded["success_probs"][0]]
        return progress, success, time.time() - t0

    def score_episode_frames(
        self,
        frames: list[EpisodeFrame],
        task_text: str | None = None,
    ) -> RobometerScore:
        """Worker-facing scorer: last-frame progress/success + ok flag."""
        try:
            images = [f.image for f in frames]
            if not images:
                return RobometerScore(0.0, 0.0, False, error="empty frames")
            progress, success, latency = self.score_chw_frames(images, task_text)
            return RobometerScore(
                progress=float(progress[-1]) if progress else 0.0,
                success=float(success[-1]) if success else 0.0,
                ok=True,
                latency_s=latency,
            )
        except Exception as exc:  # noqa: BLE001
            return RobometerScore(0.0, 0.0, False, error=str(exc), latency_s=0.0)


_GLOBAL_SCORER: RobometerScorer | None = None


def get_shared_scorer(**kwargs: Any) -> RobometerScorer:
    """Process-wide lazy singleton (optional kwargs applied only on first create)."""
    global _GLOBAL_SCORER
    if _GLOBAL_SCORER is None:
        _GLOBAL_SCORER = RobometerScorer(**kwargs)
    return _GLOBAL_SCORER
