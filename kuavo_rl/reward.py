"""Deterministic reward fusion + optional async Robometer worker."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from kuavo_rl.config import RewardConfig
from kuavo_rl.contracts import DEFAULT_TASK_TEXT, FaultCode

logger = logging.getLogger(__name__)


@dataclass
class RewardDecision:
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    fault_code: FaultCode
    source: str
    extras: dict[str, Any] = field(default_factory=dict)


class DeterministicRewardProvider:
    """Safety / timeout / manual events — never waits on a VLM."""

    def __init__(self, config: RewardConfig, *, success_reward=1.0, failure_reward=0.0, safety_penalty=-1.0):
        self.config = config
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.safety_penalty = safety_penalty

    def from_fault(self, fault: FaultCode) -> RewardDecision:
        if fault in (
            FaultCode.STOP_SIGNAL,
            FaultCode.ROS_SHUTDOWN,
            FaultCode.ACTION_NAN,
            FaultCode.ACTION_SHAPE,
            FaultCode.VELOCITY_LIMIT,
            FaultCode.SDK_EXCEPTION,
            FaultCode.STALE_OBSERVATION,
        ):
            return RewardDecision(
                self.safety_penalty, True, False, False, fault, "safety"
            )
        if fault in (FaultCode.EPISODE_TIMEOUT, FaultCode.PAUSE_TIMEOUT, FaultCode.HUMAN_ABORT):
            return RewardDecision(
                self.failure_reward, False, True, False, fault, "timeout_or_abort"
            )
        if fault == FaultCode.REWARD_MODEL_ERROR:
            return RewardDecision(
                self.failure_reward, False, True, False, fault, "reward_model_error"
            )
        return RewardDecision(0.0, False, False, False, FaultCode.NONE, "none")

    def from_manual(
        self, *, success: bool = False, failure: bool = False, abort: bool = False
    ) -> RewardDecision | None:
        if success:
            return RewardDecision(
                self.success_reward, True, False, True, FaultCode.NONE, "manual_success"
            )
        if failure:
            return RewardDecision(
                self.failure_reward, False, True, False, FaultCode.NONE, "manual_failure"
            )
        if abort:
            return RewardDecision(
                self.failure_reward, False, True, False, FaultCode.HUMAN_ABORT, "manual_abort"
            )
        return None


@dataclass
class EpisodeFrame:
    image: np.ndarray
    timestamp_s: float


@dataclass
class RobometerScore:
    progress: float
    success: float
    ok: bool
    error: str = ""
    latency_s: float = 0.0


class RobometerRewardWorker:
    """
    Asynchronous / optional Robometer scorer.

    Never call into the 10 Hz control loop synchronously.
    If model unavailable, returns stub scores and marks ok=False.
    """

    def __init__(
        self,
        config: RewardConfig,
        *,
        scorer: Callable[[list[EpisodeFrame], str], RobometerScore] | None = None,
    ):
        self.config = config
        self._scorer = scorer
        self._queue: queue.Queue = queue.Queue()
        self._results: dict[str, RobometerScore] = {}
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.enabled = config.use_robometer and config.robometer_mode != "disabled"

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="robometer-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join(timeout=2.0)
            self._thread = None

    def submit(self, episode_id: str, frames: list[EpisodeFrame], task_text: str | None = None) -> None:
        if not self.enabled:
            return
        self._queue.put(
            {
                "episode_id": episode_id,
                "frames": frames,
                "task_text": task_text or self.config.task_text or DEFAULT_TASK_TEXT,
            }
        )

    def get(self, episode_id: str, *, wait_s: float = 0.0) -> RobometerScore | None:
        deadline = time.time() + wait_s
        while True:
            with self._lock:
                if episode_id in self._results:
                    return self._results[episode_id]
            if time.time() >= deadline:
                return None
            time.sleep(0.05)

    def _loop(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            eid = item["episode_id"]
            try:
                score = self._score(item["frames"], item["task_text"])
            except Exception as exc:  # noqa: BLE001
                logger.exception("Robometer worker failed")
                score = RobometerScore(0.0, 0.0, False, error=str(exc))
            with self._lock:
                self._results[eid] = score

    def _score(self, frames: list[EpisodeFrame], task_text: str) -> RobometerScore:
        t0 = time.time()
        if self._scorer is not None:
            out = self._scorer(frames, task_text)
            out.latency_s = time.time() - t0
            return out
        # Lazy real model path — optional
        try:
            return self._score_with_lerobot(frames, task_text, t0)
        except Exception as exc:  # noqa: BLE001
            return RobometerScore(0.0, 0.0, False, error=str(exc), latency_s=time.time() - t0)

    def _score_with_lerobot(
        self, frames: list[EpisodeFrame], task_text: str, t0: float
    ) -> RobometerScore:
        from kuavo_rl.robometer_scorer import RobometerScorer, get_shared_scorer

        model_id = getattr(self.config, "robometer_model_id", None) or "lerobot/Robometer-4B"
        scorer: RobometerScorer = get_shared_scorer(model_id=model_id)
        score = scorer.score_episode_frames(frames, task_text)
        if score.latency_s <= 0:
            score.latency_s = time.time() - t0
        return score


def stub_scorer(frames: list[EpisodeFrame], task_text: str) -> RobometerScore:
    """Test stub: progress grows with frame count."""
    n = max(len(frames), 1)
    progress = min(1.0, n / 50.0)
    return RobometerScore(progress=progress, success=float(progress > 0.9), ok=True)
