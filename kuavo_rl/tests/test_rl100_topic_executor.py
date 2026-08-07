from __future__ import annotations

import time

import numpy as np

from kuavo_rl.rl100_topic_executor import RL100TopicActionScheduler


class _Policy:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def predict(self, points, states):
        assert points.shape == (2, 1024, 3)
        assert states.shape == (2, 32)
        self.calls += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        base = self.calls * 10.0
        return np.stack(
            [np.full(26, base + i, dtype=np.float32) for i in range(4)],
            axis=0,
        )


def _inputs():
    return np.zeros((2, 1024, 3), dtype=np.float32), np.zeros((2, 32), dtype=np.float32)


def test_scheduler_consumes_chunk_before_refilling():
    policy = _Policy()
    scheduler = RL100TopicActionScheduler(
        policy=policy,
        action_hz=10.0,
        execute_steps=4,
        buffer_size=8,
        low_watermark=1,
        inference_timeout_s=0.5,
    )
    try:
        points, states = _inputs()
        first = scheduler.step(points, states)
        second = scheduler.step(points, states)
        third = scheduler.step(points, states)
        assert first.step is not None and first.step.action[0] == 10.0
        assert second.step is not None and second.step.action[0] == 11.0
        assert third.step is not None and third.step.action[0] == 12.0
        # The fourth tick finishes the first chunk.  The refill is already
        # running and is consumed on the next tick.
        fourth = scheduler.step(points, states)
        assert fourth.step is not None and fourth.step.action[0] == 13.0
        fifth = None
        for _ in range(20):
            fifth = scheduler.step(points, states)
            if fifth.step is not None:
                break
            time.sleep(0.005)
        # Inference #2 was submitted when 1 action (13.0) was still pending.
        # That action was consumed while inference ran, so 1 step of the new
        # chunk is stale and dropped: we get 21.0, not 20.0.
        assert fifth is not None and fifth.step is not None and fifth.step.action[0] == 21.0
        assert fifth.state.stale_steps_dropped == 1
        assert policy.calls == 2
    finally:
        scheduler.close()


def test_first_inference_with_empty_buffer_drops_no_steps():
    """The bug fix: a slow first inference must not drop any chunk steps
    because the buffer was empty and the robot was holding, not executing."""
    policy = _Policy(delay_s=0.21)
    scheduler = RL100TopicActionScheduler(
        policy=policy,
        action_hz=10.0,
        execute_steps=4,
        buffer_size=8,
        low_watermark=1,
        inference_timeout_s=0.5,
    )
    try:
        points, states = _inputs()
        result = scheduler.step(points, states)
        assert result.step is not None
        assert result.step.action[0] == 10.0
        assert result.state.stale_steps_dropped == 0
    finally:
        scheduler.close()


def test_steady_state_drops_steps_consumed_during_inference():
    """When the buffer drains while a slow inference runs, only the actually
    consumed number of steps are dropped, not the wall-clock estimate."""
    policy = _Policy(delay_s=0.15)
    scheduler = RL100TopicActionScheduler(
        policy=policy,
        action_hz=10.0,
        execute_steps=4,
        buffer_size=8,
        low_watermark=1,
        inference_timeout_s=0.5,
    )
    try:
        points, states = _inputs()
        # First inference: instant fill, 4 actions [10,11,12,13].
        scheduler.step(points, states)  # pop 10, pending=[11,12,13]
        scheduler.step(points, states)  # pop 11, pending=[12,13]
        scheduler.step(points, states)  # pop 12, pending=[13], submit inf #2
        scheduler.step(points, states)  # pop 13, pending=[], inf #2 running
        # Wait for inference #2 (0.15s delay). 1 step was consumed during it.
        result = None
        for _ in range(40):
            result = scheduler.step(points, states)
            if result.step is not None:
                break
            time.sleep(0.005)
        assert result is not None and result.step is not None
        assert result.state.stale_steps_dropped == 1
        assert result.step.action[0] == 21.0
    finally:
        scheduler.close()
