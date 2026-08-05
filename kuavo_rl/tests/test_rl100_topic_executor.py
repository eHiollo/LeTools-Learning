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
        assert fifth is not None and fifth.step is not None and fifth.step.action[0] == 20.0
        assert policy.calls == 2
    finally:
        scheduler.close()


def test_scheduler_drops_actions_that_elapsed_during_inference():
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
        assert result.step.action[0] >= 12.0
        assert result.state.stale_steps_dropped >= 2
    finally:
        scheduler.close()
