"""Metrics for binary failure prediction."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np


def _validate_binary_scores(y_true: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(np.uint8).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if len(y) != len(s):
        raise ValueError(f"y_true has length {len(y)}, scores has length {len(s)}")
    if len(np.unique(y)) != 2:
        raise ValueError("Binary metric requires both positive and negative examples")
    return y, s


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with average ranks for ties."""

    y, s = _validate_binary_scores(y_true, scores)
    order = np.argsort(s)
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    rank_sum_pos = float(np.sum(ranks[y == 1]))
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def precision_recall_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision style AUPRC."""

    y, s = _validate_binary_scores(y_true, scores)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(float(np.sum(y == 1)), 1.0)

    distinct = np.r_[np.where(np.diff(s[order]))[0], len(s) - 1]
    precision = precision[distinct]
    recall = recall[distinct]
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Smallest FPR among thresholds reaching at least target TPR."""

    y, s = _validate_binary_scores(y_true, scores)
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    n_pos = max(float(np.sum(y == 1)), 1.0)
    n_neg = max(float(np.sum(y == 0)), 1.0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    ok = tpr >= target_tpr
    if not np.any(ok):
        return 1.0
    return float(np.min(fpr[ok]))


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    """Compute the common Phase 1 binary metrics."""

    y, s = _validate_binary_scores(y_true, scores)
    return {
        "auroc": float(roc_auc(y, s)),
        "auprc": float(precision_recall_auc(y, s)),
        "fpr_at_95_tpr": float(fpr_at_tpr(y, s, target_tpr=0.95)),
        "positive_rate": float(np.mean(y)),
    }


def decile_failure_rates(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, list]:
    """Failure rate by score decile, low score to high score."""

    y = np.asarray(y_true).astype(np.uint8).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    order = np.argsort(s)
    bins = np.array_split(order, n_bins)
    rows = []
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": int(i),
                "n": int(len(idx)),
                "score_min": float(np.min(s[idx])),
                "score_max": float(np.max(s[idx])),
                "score_median": float(np.median(s[idx])),
                "failure_rate": float(np.mean(y[idx])),
            }
        )
    return {"bins_low_to_high": rows}


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    midranks = np.zeros(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        midranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return midranks


def _fast_delong(predictions_sorted: np.ndarray, n_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fast DeLong covariance for one or more correlated ROC curves."""

    n_classifiers = predictions_sorted.shape[0]
    n_neg = predictions_sorted.shape[1] - n_pos
    positives = predictions_sorted[:, :n_pos]
    negatives = predictions_sorted[:, n_pos:]
    tx = np.empty((n_classifiers, n_pos), dtype=np.float64)
    ty = np.empty((n_classifiers, n_neg), dtype=np.float64)
    tz = np.empty_like(predictions_sorted, dtype=np.float64)
    for r in range(n_classifiers):
        tx[r] = _compute_midrank(positives[r])
        ty[r] = _compute_midrank(negatives[r])
        tz[r] = _compute_midrank(predictions_sorted[r])
    aucs = tz[:, :n_pos].sum(axis=1) / (n_pos * n_neg) - (n_pos + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :n_pos] - tx) / n_neg
    v10 = 1.0 - (tz[:, n_pos:] - ty) / n_pos
    sx = np.cov(v01)
    sy = np.cov(v10)
    if n_classifiers == 1:
        sx = np.asarray([[float(sx)]])
        sy = np.asarray([[float(sy)]])
    covariance = sx / n_pos + sy / n_neg
    return aucs, covariance


def delong_roc_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> Dict[str, float]:
    """Two-sided DeLong test for correlated AUROC difference."""

    y = np.asarray(y_true).astype(np.uint8).ravel()
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    if len(y) != len(a) or len(y) != len(b):
        raise ValueError("Inputs must have equal length")
    if len(np.unique(y)) != 2:
        raise ValueError("DeLong test requires both positive and negative examples")

    order = np.argsort(-y.astype(np.int8))
    n_pos = int(np.sum(y == 1))
    preds = np.vstack([a, b])[:, order]
    aucs, covariance = _fast_delong(preds, n_pos)
    diff = float(aucs[0] - aucs[1])
    var = float(covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1])
    if var <= 0:
        z = math.inf if diff != 0 else 0.0
        p_value = 0.0 if diff != 0 else 1.0
    else:
        z = diff / math.sqrt(var)
        # two-sided normal tail using erfc, avoiding scipy dependency
        p_value = math.erfc(abs(z) / math.sqrt(2.0))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": diff,
        "z": float(z),
        "p_value_two_sided": float(p_value),
    }

