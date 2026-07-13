import time

from kuavo_rl.config import RewardConfig
from kuavo_rl.reward import EpisodeFrame, RobometerRewardWorker, stub_scorer
import numpy as np


def test_async_worker_with_stub():
    cfg = RewardConfig(use_robometer=True, robometer_mode="episode_end")
    worker = RobometerRewardWorker(cfg, scorer=stub_scorer)
    worker.start()
    frames = [EpisodeFrame(np.zeros((3, 8, 8), dtype=np.uint8), 0.0) for _ in range(5)]
    worker.submit("ep1", frames, "test task")
    score = worker.get("ep1", wait_s=2.0)
    assert score is not None
    assert score.ok
    assert score.progress > 0
    worker.stop()


def test_disabled_worker_noop():
    cfg = RewardConfig(use_robometer=False, robometer_mode="disabled")
    worker = RobometerRewardWorker(cfg, scorer=stub_scorer)
    worker.start()
    worker.submit("ep2", [], "x")
    time.sleep(0.1)
    assert worker.get("ep2", wait_s=0.0) is None
    worker.stop()
