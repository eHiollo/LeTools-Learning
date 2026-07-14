import time

import numpy as np

from kuavo_rl.calibration_metrics import (
    binary_auc,
    evaluate_calibration_gates,
    mean_spearman_progress,
    spearman_rank_corr,
)
from kuavo_rl.config import RewardConfig
from kuavo_rl.reward import EpisodeFrame, RobometerRewardWorker, stub_scorer


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


def test_spearman_monotonic():
    x = list(range(10))
    y = [0.1 * i for i in range(10)]
    assert spearman_rank_corr(x, y) > 0.99


def test_binary_auc_perfect():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [0, 0, 1, 1]
    assert binary_auc(scores, labels) == 1.0


def test_calibration_gates_pass_on_synthetic():
    success_curves = [[0.1, 0.3, 0.6, 0.9], [0.0, 0.4, 0.7, 1.0]]
    success_finals = [0.9, 1.0]
    failure_finals = [0.1, 0.2]
    gates = evaluate_calibration_gates(
        success_progress_curves=success_curves,
        success_final_scores=success_finals,
        failure_final_scores=failure_finals,
    )
    assert gates["passed"] is True
    assert gates["spearman_progress"] >= 0.7
    assert gates["success_vs_fail_auc"] >= 0.85


def test_calibration_gates_fail_when_indiscriminate():
    success_curves = [[0.5, 0.5, 0.5]]
    gates = evaluate_calibration_gates(
        success_progress_curves=success_curves,
        success_final_scores=[0.5, 0.5],
        failure_final_scores=[0.5, 0.5],
    )
    assert gates["passed"] is False


def test_worker_uses_injected_scorer_not_global_model():
    calls = {"n": 0}

    def fake_scorer(frames, task_text):
        calls["n"] += 1
        from kuavo_rl.reward import RobometerScore

        return RobometerScore(progress=0.42, success=0.1, ok=True)

    cfg = RewardConfig(use_robometer=True, robometer_mode="episode_end")
    worker = RobometerRewardWorker(cfg, scorer=fake_scorer)
    worker.start()
    frames = [EpisodeFrame(np.zeros((3, 4, 4), dtype=np.uint8), 0.0)]
    worker.submit("ep3", frames, "t")
    score = worker.get("ep3", wait_s=2.0)
    worker.stop()
    assert score is not None and score.progress == 0.42
    assert calls["n"] == 1


def test_mean_spearman_ignores_short():
    assert np.isnan(mean_spearman_progress([[0.1]])) or mean_spearman_progress([[0.1, 0.2]]) > 0
