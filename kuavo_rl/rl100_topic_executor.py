"""Small asynchronous action-chunk executor for RL-100 topic deployment.

The model is evaluated in a worker thread while the ROS control loop consumes
one action per dataset timestep.  This keeps the action timing at the
training rate without requiring the model to complete an inference every
control tick.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
import math
import threading
import time
from typing import Protocol

import numpy as np

from kuavo_rl.contracts import RL100_ACTION_DIM


class ActionChunkPolicy(Protocol):
    def predict(self, point_cloud_history: np.ndarray, state_history: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class TopicActionStep:
    """One action selected from a predicted chunk."""

    action: np.ndarray
    inference_s: float
    chunk_id: int


@dataclass(frozen=True)
class TopicActionSchedulerState:
    """Scheduler metadata needed by the audit log and runner."""

    pending: int
    inference_running: bool
    inference_ready: bool
    inference_s: float | None
    predicted_chunk: np.ndarray | None
    stale_steps_dropped: int
    last_error: str | None


@dataclass(frozen=True)
class TopicActionSchedulerResult:
    step: TopicActionStep | None
    state: TopicActionSchedulerState
    waiting: bool


@dataclass(frozen=True)
class _InferenceResult:
    generation: int
    chunk: np.ndarray | None
    inference_s: float
    error: str | None


class RL100TopicActionScheduler:
    """Consume one action per tick and refill a small action buffer in the background.

    The first inference is allowed to complete synchronously for a bounded
    startup wait.  Later inferences are submitted before the buffer is empty;
    a slow result is still accepted, but actions that are already in the past
    are removed according to the measured inference latency.
    """

    def __init__(
        self,
        *,
        policy: ActionChunkPolicy,
        action_hz: float,
        execute_steps: int,
        buffer_size: int,
        low_watermark: int,
        inference_timeout_s: float,
    ) -> None:
        if action_hz <= 0:
            raise ValueError("action_hz must be positive")
        if execute_steps <= 0:
            raise ValueError("execute_steps must be positive")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        if low_watermark < 0 or low_watermark >= buffer_size:
            raise ValueError("low_watermark must be in [0, buffer_size)")
        if inference_timeout_s <= 0:
            raise ValueError("inference_timeout_s must be positive")
        self.policy = policy
        self.action_hz = float(action_hz)
        self.execute_steps = int(execute_steps)
        self.buffer_size = int(buffer_size)
        self.low_watermark = int(low_watermark)
        self.inference_timeout_s = float(inference_timeout_s)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rl100-inference")
        self._future: Future[_InferenceResult] | None = None
        self._pending: list[TopicActionStep] = []
        self._generation = 0
        self._chunk_id = 0
        self._started = False
        self._last_error: str | None = None
        self._last_chunk: np.ndarray | None = None
        self._last_inference_s: float | None = None
        self._last_stale_steps_dropped = 0
        self._lock = threading.Lock()
        self._closed = False

    def reset(self) -> None:
        """Drop queued actions and invalidate an in-flight result."""
        with self._lock:
            self._generation += 1
            self._pending.clear()
            self._started = False
            self._last_error = None
            self._last_chunk = None
            self._last_inference_s = None
            self._last_stale_steps_dropped = 0
            future = self._future
            self._future = None
        if future is not None:
            future.cancel()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            future = self._future
            self._future = None
        if future is not None:
            future.cancel()
        # ``cancel_futures`` was added after the Python version used by some
        # Kuavo images; the currently running future was cancelled above and
        # queued work is invalidated by the generation counter.
        self._executor.shutdown(wait=False)

    @property
    def has_inference_in_flight(self) -> bool:
        with self._lock:
            return self._future is not None

    @property
    def max_empty_ticks(self) -> int:
        # Permit a slow model to refill the queue, while still bounding a
        # missing/failed inference instead of replaying a stale target forever.
        return max(1, int(math.ceil(self.inference_timeout_s * self.action_hz)) + 1)

    def _infer(
        self,
        points: np.ndarray,
        states: np.ndarray,
        generation: int,
        submitted_at: float,
    ) -> _InferenceResult:
        started = submitted_at
        try:
            chunk = np.asarray(self.policy.predict(points, states), dtype=np.float32)
            inference_s = max(0.0, float(time.monotonic() - started))
            if chunk.ndim != 2 or chunk.shape[1] != RL100_ACTION_DIM or chunk.shape[0] < 1:
                return _InferenceResult(
                    generation,
                    None,
                    inference_s,
                    f"policy chunk shape {chunk.shape}",
                )
            if not np.isfinite(chunk).all():
                return _InferenceResult(generation, None, inference_s, "policy chunk has NaN/Inf")
            return _InferenceResult(generation, chunk, inference_s, None)
        except Exception as exc:  # noqa: BLE001
            inference_s = max(0.0, float(time.monotonic() - started))
            return _InferenceResult(generation, None, inference_s, str(exc))

    def _submit(self, points: np.ndarray, states: np.ndarray) -> None:
        with self._lock:
            if self._closed or self._future is not None:
                return
            generation = self._generation
            submitted_at = time.monotonic()
            self._last_error = None
            self._future = self._executor.submit(
                self._infer,
                np.asarray(points, dtype=np.float32).copy(),
                np.asarray(states, dtype=np.float32).copy(),
                generation,
                submitted_at,
            )

    def _accept_result(self, result: _InferenceResult) -> None:
        with self._lock:
            if result.generation != self._generation:
                return
            self._last_inference_s = result.inference_s
            self._last_stale_steps_dropped = 0
            self._last_chunk = None if result.chunk is None else result.chunk.copy()
            self._last_error = result.error
            if result.chunk is None:
                return
            stale_steps = min(
                result.chunk.shape[0],
                max(0, int(math.floor(result.inference_s * self.action_hz))),
            )
            self._last_stale_steps_dropped = stale_steps
            chunk = result.chunk[stale_steps : self.execute_steps].copy()
            if chunk.shape[0] == 0:
                self._last_error = (
                    f"inference {result.inference_s:.3f}s consumed the entire "
                    f"{result.chunk.shape[0]}-step action chunk"
                )
                return
            available = max(0, self.buffer_size - len(self._pending))
            for action in chunk[:available]:
                self._chunk_id += 1
                self._pending.append(
                    TopicActionStep(action.copy(), result.inference_s, self._chunk_id)
                )

    def _poll(self) -> bool:
        with self._lock:
            future = self._future
            if future is None or not future.done():
                return False
            self._future = None
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - executor failures are defensive
            result = _InferenceResult(self._generation, None, 0.0, str(exc))
        self._accept_result(result)
        return True

    def step(self, points: np.ndarray, states: np.ndarray) -> TopicActionSchedulerResult:
        """Poll/refill the queue and return the next action, if available."""
        if self._closed:
            raise RuntimeError("RL100 topic action scheduler is closed")
        self._poll()
        with self._lock:
            pending = len(self._pending)
            first = not self._started
            should_submit = self._future is None and pending <= self.low_watermark
            self._started = True
        if should_submit:
            self._submit(points, states)
            # Startup should not add a full control-period delay when the
            # checkpoint is fast.  A slow first inference is handled by the
            # bounded empty-buffer hold in the runner.
            if first:
                with self._lock:
                    future = self._future
                if future is not None:
                    try:
                        future.result(timeout=self.inference_timeout_s)
                    except TimeoutError:
                        pass
                    self._poll()

        with self._lock:
            step = self._pending.pop(0) if self._pending else None
            state = TopicActionSchedulerState(
                pending=len(self._pending),
                inference_running=self._future is not None,
                inference_ready=self._last_inference_s is not None,
                inference_s=self._last_inference_s,
                predicted_chunk=None if self._last_chunk is None else self._last_chunk.copy(),
                stale_steps_dropped=self._last_stale_steps_dropped,
                last_error=self._last_error,
            )
        return TopicActionSchedulerResult(step=step, state=state, waiting=step is None)
