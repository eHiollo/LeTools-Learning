"""LeRobot HIL-SERL processors for KuavoHILSerlEnv (Stage-B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor

from lerobot.configs import PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    InterventionActionProcessorStep,
    Numpy2TorchActionProcessorStep,
    ProcessorStep,
    TimeLimitProcessorStep,
    Torch2NumpyActionProcessorStep,
    TransitionKey,
    identity_transition,
)
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.utils.constants import OBS_STATE


@dataclass
class KuavoObservationProcessorStep(ProcessorStep):
    """Convert Kuavo CHW uint8 / float obs into batched float32 tensors in [0, 1]."""

    def __call__(self, transition: dict) -> dict:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}
        new_obs: dict[str, Any] = {}
        for key, value in obs.items():
            if key.startswith("observation.images."):
                new_obs[key] = self._image_to_tensor(value)
            elif key == OBS_STATE or key == "observation.state":
                t = torch.as_tensor(np.asarray(value), dtype=torch.float32)
                if t.ndim == 1:
                    t = t.unsqueeze(0)
                new_obs[OBS_STATE] = t
            else:
                new_obs[key] = value

        info = transition.get(TransitionKey.INFO, {}) or {}
        if "is_intervention" in info:
            info[TeleopEvents.IS_INTERVENTION] = bool(info["is_intervention"])
            transition[TransitionKey.INFO] = info

        transition[TransitionKey.OBSERVATION] = new_obs
        return transition

    @staticmethod
    def _image_to_tensor(value: Any) -> Tensor:
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            # HWC -> CHW
            arr = np.transpose(arr, (2, 0, 1))
        t = torch.from_numpy(np.ascontiguousarray(arr))
        if t.ndim == 3:
            t = t.unsqueeze(0)
        if t.dtype == torch.uint8:
            t = t.to(dtype=torch.float32) / 255.0
        else:
            t = t.to(dtype=torch.float32)
            if float(t.max()) > 1.0:
                t = t / 255.0
        return t

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


def make_kuavo_processors(cfg, device: str = "cpu") -> tuple[DataProcessorPipeline, DataProcessorPipeline]:
    """Env/action processors for Kuavo Stage-B (single-step 16-D, no ACT chunk)."""
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )
    action_pipeline_steps = [
        InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
        Torch2NumpyActionProcessorStep(),
    ]
    env_pipeline_steps: list[ProcessorStep] = [
        KuavoObservationProcessorStep(),
        Numpy2TorchActionProcessorStep(),
    ]
    if cfg.processor.reset is not None:
        env_pipeline_steps.append(
            TimeLimitProcessorStep(max_episode_steps=int(cfg.processor.reset.control_time_s * cfg.fps))
        )
    env_pipeline_steps.extend(
        [
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device),
        ]
    )
    return (
        DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ),
        DataProcessorPipeline(
            steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ),
    )
