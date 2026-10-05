"""Full OSD-CS candidate landscape for one shot, without running the sweep.

What `ldpc` 2.4.1 actually does with `osd_method="osd_cs"`, `osd_order=2`
(`ldpc::osd::OsdDecoder::decode` in `osd.hpp`), stated precisely because the
textbook description does not match it:

* it builds `k = n - rank(H)` single-flip candidate patterns, one per non-pivot
  ("free") column, in ascending BP log-probability-ratio order, plus exactly
  *one* double-flip pattern on free indices 0 and 1 -- not all C(k,2) pairs.
  With `osd_order = lambda`, the doubles are the pairs drawn from the first
  `lambda` free indices only, so order 2 contributes a single extra candidate
  and order 8 contributes C(8,2) = 28. Total scored here: k + 1 = 10447;
* it scores a candidate by

      S(x) = sum_{i : x_i = 1} log(1 / p_i)

  over the *channel* priors from the detector error model. This is a
  prior-weighted count of the mechanisms the candidate turns on. It is neither a
  probability nor a plain Hamming weight, and it ignores the 0-bits entirely, so
  it is not the log-likelihood of the error vector either;
* it keeps a candidate only on a strict improvement (`<`), so the winner is the
  *first* candidate attaining the minimum score, and OSD-0 wins if nothing beats
  it.

The sweep costs about 1.3 s/shot because it runs `k+1` separate `lu_solve`
calls. None of that is necessary offline. Writing the candidate for free column
j as `e_j = e_0 XOR n_j`, where `n_j = u_j XOR unit(j)` and `u_j` is column j
expressed in the pivot basis, gives

      H n_j = H[:,j] XOR H[:,j] = 0,

so every candidate differs from OSD-0 by an element of ker(H), and all the `u_j`
are read straight off the single reduced matrix already computed for OSD-0.
The scores then follow from one matrix-vector product:

      Delta S_j = S(e_0) - S(e_j) = sum_{i in supp(n_j)} v_i,
      v_i = +log(1/p_i) if e_0[i] = 1 else -log(1/p_i),

because flipping a bit that OSD-0 had set removes its weight and flipping a bit
it had clear adds it. The same reduction gives the logical action of every
candidate, `L e_j = L e_0 XOR L n_j`, which is what decides whether a candidate
sits in a different logical coset.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from failure_prediction.osd_internals import unpack_columns

WINNER_OSD0, WINNER_SINGLE, WINNER_DOUBLE = 0, 1, 2
WINNER_NAMES = ["osd0", "single", "double"]


def _blocked_real_matvec(v: np.ndarray, u: np.ndarray, block: int = 2048) -> np.ndarray:
    """v @ u in float64 without materialising a float copy of the whole matrix."""

    out = np.empty(u.shape[1], dtype=np.float64)
    for start in range(0, u.shape[1], block):
        stop = min(start + block, u.shape[1])
        out[start:stop] = v @ u[:, start:stop].astype(np.float64)
    return out


def _blocked_gf2_matmul(a: np.ndarray, u: np.ndarray, block: int = 2048) -> np.ndarray:
    """(a @ u) mod 2 for small a, blocked over the columns of u."""

    out = np.empty((a.shape[0], u.shape[1]), dtype=np.uint8)
    a32 = a.astype(np.float32)
    for start in range(0, u.shape[1], block):
        stop = min(start + block, u.shape[1])
        prod = a32 @ u[:, start:stop].astype(np.float32)
        out[:, start:stop] = (prod.astype(np.int64) & 1).astype(np.uint8)
    return out


def candidate_landscape(
    info: Dict[str, np.ndarray],
    log_inv_p: np.ndarray,
    logicals: np.ndarray,
    n_cols: int,
    rank: int,
    osd_order: int = 2,
) -> Optional[Dict[str, np.ndarray]]:
    """Score and logically classify every OSD-CS candidate for one shot.

    `logicals` is the (num_observables, n) DEM observable matrix, so
    `logicals @ e` is the logical action of an error vector.
    """

    e0 = info["solution"]
    pivot_cols = info["pivot_cols"]
    order = info["order"]

    u = unpack_columns(info["reduced"], n_cols)
    is_pivot = np.zeros(n_cols, dtype=bool)
    is_pivot[pivot_cols] = True
    free_cols = order[~is_pivot[order]]
    if free_cols.size < 2:
        return None

    v = np.where(e0 == 1, log_inv_p, -log_inv_p)
    delta_all = _blocked_real_matvec(v[pivot_cols], u)
    delta_single = delta_all[free_cols] + v[free_cols]

    # Logical action of each nullspace generator n_j = u_j XOR unit(j).
    l_null = _blocked_gf2_matmul(logicals[:, pivot_cols], u) ^ logicals
    l_null_single = l_null[:, free_cols]

    # The one double-flip pattern order 2 contributes, on free indices 0 and 1.
    f0, f1 = int(free_cols[0]), int(free_cols[1])
    col01 = u[:, f0] ^ u[:, f1]
    delta_double = float(v[pivot_cols] @ col01.astype(np.float64) + v[f0] + v[f1])
    l_null_double = l_null[:, f0] ^ l_null[:, f1]

    # Upstream order: all singles, then the doubles. It replaces the incumbent
    # only on a strict improvement, so the first index attaining the maximum
    # improvement wins, and OSD-0 survives if no improvement is positive.
    scores = np.concatenate([delta_single, [delta_double]])
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    if best_score <= 0.0:
        winner_type, winner_rank = WINNER_OSD0, -1
    elif best_idx < delta_single.size:
        winner_type, winner_rank = WINNER_SINGLE, best_idx
    else:
        winner_type, winner_rank = WINNER_DOUBLE, -1

    return {
        "e0": e0,
        "free_cols": free_cols,
        "delta_single": delta_single,
        "delta_double": delta_double,
        "l_null_single": l_null_single,
        "l_null_double": l_null_double,
        "reduced": u,
        "winner_type": winner_type,
        "winner_rank": winner_rank,
        "winner_score": max(best_score, 0.0),
        "n_candidates": int(delta_single.size + 1),
        "double_indices": (f0, f1),
    }


def winner_error_vector(
    land: Dict[str, np.ndarray], n_cols: int, pivot_cols: np.ndarray
) -> np.ndarray:
    """Rebuild the OSD-CS winning error vector from the landscape."""

    e = land["e0"].copy()
    if land["winner_type"] == WINNER_OSD0:
        return e
    u = land["reduced"]
    if land["winner_type"] == WINNER_SINGLE:
        j = int(land["free_cols"][land["winner_rank"]])
        col = u[:, j]
        flips = [j]
    else:
        f0, f1 = land["double_indices"]
        col = u[:, f0] ^ u[:, f1]
        flips = [f0, f1]
    e[pivot_cols] ^= col
    for j in flips:
        e[j] ^= 1
    return e


def truncated_winner(
    land: Dict[str, np.ndarray], k_singles: int
) -> tuple:
    """Winner if the sweep stopped after the first `k_singles` single candidates.

    The double-flip pattern is scored last upstream, so a truncated sweep that
    stops inside the singles never reaches it.
    """

    k = min(k_singles, land["delta_single"].size)
    if k <= 0:
        return WINNER_OSD0, -1, 0.0
    head = land["delta_single"][:k]
    idx = int(np.argmax(head))
    if head[idx] <= 0.0:
        return WINNER_OSD0, -1, 0.0
    return WINNER_SINGLE, idx, float(head[idx])
