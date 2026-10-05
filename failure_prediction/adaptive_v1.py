"""Adaptive Decoder V1: OSD-0, then d_H-gated truncated one-free-variable CS.

This is the main adaptive-decoding pipeline, not a mechanism study.

    BP@20 → OSD-0 → d_H(e_BP, e_OSD0) → gate
        d_H <  τ  → return OSD-0
        d_H ≥  τ  → truncated one-free-variable CS of budget K

d_H is the Hamming distance between the BP hard decision and the OSD-0
solution. Both are already produced by the fast decoder; no extra BP/OSD work
is required to compute it.

The expensive branch perturbs one OSD *free variable* at a time (the first K
in ascending-LPR order) and re-solves He = s on the pivots. That is not a
single-qubit search. The order-2 double perturbation is not used: it rescued
0/915 WC instances.

K=500 is the primary budget (~80% of WC rescue in the mechanism study).
K=1000 is a sensitivity point. Full one-free-variable CS (all k free columns)
is the accuracy reference for the strong branch; package OSD-CS is timed on a
subsample as the legacy wall-clock reference.

Scoring is the package surrogate S(x) = sum_{i: x_i=1} log(1/p_i).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from failure_prediction.osd_internals import (
    extract_columns,
    free_columns,
    osd0_information_set,
)

K_PRIMARY = 500
K_SENSITIVITY = 1000


def _logical_fail(logicals: np.ndarray, e: np.ndarray, actual: np.ndarray) -> int:
    pred = (logicals @ e.astype(np.int64)) % 2
    return int(np.any(pred != actual))


def score_one_free_prefix(
    info: Dict[str, np.ndarray],
    log_inv_p: np.ndarray,
    n_cols: int,
    k_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ΔS of the first k_max one-free-variable candidates, plus reconstruction data.

    Returns (delta, free_prefix, rf) with rf shape (rank, k_used).
    """

    e0 = info["solution"]
    pivot_cols = info["pivot_cols"]
    free = free_columns(info, n_cols)
    k_used = int(min(k_max, free.size))
    free_p = free[:k_used]
    rf = extract_columns(info["reduced"], free_p)
    v_piv = np.where(
        e0[pivot_cols] == 1, log_inv_p[pivot_cols], -log_inv_p[pivot_cols]
    )
    # Improvement Δ = S(e0) - S(e_j), matching osd_cs_landscape. OSD-0 holds
    # every free bit at 0, so flipping free j adds log(1/p_j) to S and subtracts
    # it from Δ.
    delta = v_piv @ rf.astype(np.float64) - log_inv_p[free_p]
    return np.asarray(delta, dtype=np.float64), free_p, rf


def pick_one_free(
    info: Dict[str, np.ndarray],
    delta: np.ndarray,
    free_p: np.ndarray,
    rf: np.ndarray,
    k: int,
) -> np.ndarray:
    """OSD-0, or the best one-free-variable candidate among the first k."""

    e0 = info["solution"].copy()
    if k <= 0 or delta.size == 0:
        return e0
    head = delta[: min(k, delta.size)]
    t = int(np.argmax(head))
    if head[t] <= 0.0:
        return e0
    e0[info["pivot_cols"]] ^= rf[:, t]
    e0[int(free_p[t])] = 1
    return e0


def evaluate_shot(
    decoder,
    syndrome: np.ndarray,
    actual: np.ndarray,
    packed,
    log_inv_p: np.ndarray,
    logicals: np.ndarray,
    n_cols: int,
    rank: int,
    k_values: Sequence[int] = (K_PRIMARY, K_SENSITIVITY),
) -> Dict[str, float]:
    """Run the fast path and the truncated strong path; record fails and times.

    Always computes truncated solutions so a τ sweep can be done offline.
    When BP converges, OSD never runs and every branch returns the BP output.
    """

    syndrome = np.asarray(syndrome, dtype=np.uint8)
    actual = np.asarray(actual, dtype=np.uint8).ravel()

    t0 = time.perf_counter()
    decoder.decode(syndrome)
    t_osd0 = time.perf_counter() - t0

    conv = bool(decoder.converge)
    e_bp = np.asarray(decoder.bp_decoding, dtype=np.uint8) % 2
    e0 = np.asarray(decoder.osd0_decoding, dtype=np.uint8) % 2
    d_h = int(np.count_nonzero(e_bp ^ e0))
    fail0 = _logical_fail(logicals, e0, actual)

    out: Dict[str, float] = {
        "d_h": float(d_h),
        "bp_converged": float(conv),
        "fail_osd0": float(fail0),
        "t_osd0_ms": 1e3 * t_osd0,
        "t_ge_ms": 0.0,
        "t_k500_ms": 0.0,
        "t_k1000_ms": 0.0,
        "t_full_ms": 0.0,
        "fail_k500": float(fail0),
        "fail_k1000": float(fail0),
        "fail_full": float(fail0),
    }
    if conv:
        return out

    llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
    t1 = time.perf_counter()
    info = osd0_information_set(packed, syndrome, llr, n_cols=n_cols, rank=rank)
    t_ge = time.perf_counter() - t1
    out["t_ge_ms"] = 1e3 * t_ge
    if info is None:
        return out

    # Score every free column once. Adaptive cost later charges only the first
    # K of that work; the remainder is the accuracy-reference full 1FV sweep.
    t2 = time.perf_counter()
    delta, free_p, rf = score_one_free_prefix(info, log_inv_p, n_cols, n_cols)
    t_score = time.perf_counter() - t2
    n_scored = max(int(delta.size), 1)
    t_per = t_score / n_scored
    n500 = min(K_PRIMARY, n_scored)
    n1000 = min(K_SENSITIVITY, n_scored)
    out["t_k500_ms"] = 1e3 * (t_ge + t_per * n500)
    out["t_k1000_ms"] = 1e3 * (t_ge + t_per * n1000)
    out["t_full_ms"] = 1e3 * (t_ge + t_score)

    e500 = pick_one_free(info, delta, free_p, rf, K_PRIMARY)
    e1000 = pick_one_free(info, delta, free_p, rf, K_SENSITIVITY)
    e_full = pick_one_free(info, delta, free_p, rf, int(delta.size))
    out["fail_k500"] = float(_logical_fail(logicals, e500, actual))
    out["fail_k1000"] = float(_logical_fail(logicals, e1000, actual))
    out["fail_full"] = float(_logical_fail(logicals, e_full, actual))
    return out


def evaluate_batch(
    decoder,
    syndromes: np.ndarray,
    actual_obs: np.ndarray,
    packed,
    log_inv_p: np.ndarray,
    logicals: np.ndarray,
    n_cols: int,
    rank: int,
    progress_every: int = 100,
) -> Dict[str, np.ndarray]:
    rows: List[Dict[str, float]] = []
    t0 = time.time()
    for i, syn in enumerate(syndromes):
        rows.append(
            evaluate_shot(
                decoder,
                syn,
                actual_obs[i],
                packed,
                log_inv_p,
                logicals,
                n_cols,
                rank,
            )
        )
        if progress_every and (i + 1) % progress_every == 0:
            print(
                f"  {i + 1}/{len(syndromes)}  "
                f"{(time.time() - t0) / (i + 1):.3f}s/shot",
                flush=True,
            )
    keys = list(rows[0].keys())
    return {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys}
