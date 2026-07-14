"""Phase 3.4 Robometer calibration metrics (pure numpy, no model deps)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

SPEARMAN_PROGRESS_MIN = 0.7
SUCCESS_AUC_MIN = 0.85


def spearman_rank_corr(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman ρ; returns nan if undefined (constant series / too short)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size < 2 or b.size != a.size:
        return float("nan")
    ra = a.argsort().argsort().astype(np.float64)
    rb = b.argsort().argsort().astype(np.float64)
    if float(np.std(ra)) < 1e-12 or float(np.std(rb)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def binary_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC for binary labels (1=positive). Ties handled via rank average."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.int64).ravel()
    if s.size == 0 or s.size != y.size:
        return float("nan")
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(s, dtype=np.float64)
    # average ranks for ties
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[order[j + 1]] == s[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0  # 1-based
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos_ranks = float(np.sum(ranks[y == 1]))
    return (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mean_spearman_progress(curves: Sequence[Sequence[float]]) -> float:
    """Mean Spearman(progress, time_index) over episodes with valid curves."""
    vals: list[float] = []
    for curve in curves:
        c = np.asarray(curve, dtype=np.float64).ravel()
        if c.size < 2:
            continue
        rho = spearman_rank_corr(np.arange(c.size, dtype=np.float64), c)
        if np.isfinite(rho):
            vals.append(rho)
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def evaluate_calibration_gates(
    *,
    success_progress_curves: Sequence[Sequence[float]],
    success_final_scores: Sequence[float],
    failure_final_scores: Sequence[float],
    spearman_min: float = SPEARMAN_PROGRESS_MIN,
    auc_min: float = SUCCESS_AUC_MIN,
) -> dict:
    """Compute handbook 3.4 gates: monotonicity + success/fail discrimination."""
    spearman = mean_spearman_progress(success_progress_curves)
    scores = list(success_final_scores) + list(failure_final_scores)
    labels = [1] * len(success_final_scores) + [0] * len(failure_final_scores)
    auc = binary_auc(scores, labels)
    passed = (
        bool(np.isfinite(spearman) and spearman >= spearman_min)
        and bool(np.isfinite(auc) and auc >= auc_min)
        and len(success_final_scores) > 0
        and len(failure_final_scores) > 0
    )
    return {
        "spearman_progress": spearman,
        "success_vs_fail_auc": auc,
        "spearman_progress_min": spearman_min,
        "success_auc_min": auc_min,
        "n_success": len(success_final_scores),
        "n_failure": len(failure_final_scores),
        "passed": passed,
    }
