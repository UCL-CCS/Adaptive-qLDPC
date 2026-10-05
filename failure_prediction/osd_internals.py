"""Faithful re-derivation of the OSD-0 information set used by `ldpc` 2.4.1.

The `ldpc` package keeps the OSD column ordering and pivot set in C++
(`ldpc::osd::OsdDecoder::column_ordering` and `RowReduce::cols/pivots`) and does
not expose either to Python, so any structural quantity defined on the
information-set boundary has to be re-derived here. Everything below mirrors the
upstream implementation rather than a textbook description of OSD:

* the columns are ordered by *ascending BP log-probability ratio*, via
  `ldpc::sort::soft_decision_col_sort(log_prob_ratios, column_ordering, n)`.
  It sorts the raw LPR, not |LPR| and not the channel probability, so the
  columns BP considers most likely to carry an error come first;
* the information set is the first `rank(H)` linearly independent columns in
  that order, taken greedily left to right (`RowReduce::rref`);
* OSD-0 solves for those columns and leaves every non-pivot column at zero
  (`lu_solve` starts from an all-zero vector and only writes pivot entries);
* OSD-CS scores a candidate by `sum_{i: x_i=1} log(1/p_i)` with the *channel*
  probabilities, not the BP posteriors.

The reconstruction is verified bit-for-bit against the decoder's own
`osd0_decoding` before any feature built on it is trusted; see
`verify_against_decoder`.

Note on row tie-breaking: upstream picks the sparsest available pivot row, which
changes which row is used but not the reduced row echelon form, and hence not
the solution. The pivot *column* set is fixed by independence in the sorted
order, so it is reproducible here. Exact ties in the LPR are the one genuine
ambiguity, since C `qsort` is not stable; `tie_fraction` reports how often that
could bite.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

_ONE = np.uint64(1)


def pack_matrix(h, n_cols: int, extra_cols: int = 1) -> np.ndarray:
    """Bit-pack a sparse GF(2) matrix into uint64 words, with spare columns.

    The spare columns hold the augmented syndrome during elimination.
    """

    h = h.tocoo()
    m = h.shape[0]
    n_words = (n_cols + extra_cols + 63) // 64
    packed = np.zeros((m, n_words), dtype=np.uint64)
    rows = h.row.astype(np.int64)
    cols = h.col.astype(np.int64)
    words = cols >> 6
    bits = (_ONE << (cols & 63).astype(np.uint64)).astype(np.uint64)
    np.bitwise_xor.at(packed, (rows, words), bits)
    return packed


def gf2_rank_packed(packed: np.ndarray, n_cols: int) -> int:
    """Rank of the bit-packed matrix, scanning columns in natural order."""

    work = packed.copy()
    m = work.shape[0]
    rank = 0
    for j in range(n_cols):
        w, b = j >> 6, np.uint64(j & 63)
        col = (work[rank:, w] >> b) & _ONE
        nz = np.flatnonzero(col)
        if nz.size == 0:
            continue
        pr = rank + int(nz[0])
        if pr != rank:
            work[[rank, pr]] = work[[pr, rank]]
        hits = ((work[:, w] >> b) & _ONE).astype(bool)
        hits[rank] = False
        if hits.any():
            work[hits] ^= work[rank]
        rank += 1
        if rank == m:
            break
    return rank


def osd0_information_set(
    packed_h: np.ndarray,
    syndrome: np.ndarray,
    llr: np.ndarray,
    n_cols: int,
    rank: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Reproduce the OSD-0 pivot set and solution for one shot.

    Columns are visited in ascending-LPR order and taken as pivots whenever they
    are independent of those already taken, which is exactly the upstream greedy
    scan. Returns the pivot columns in visit order, the sorted ordering, the
    index in that ordering at which the information set closes, and the OSD-0
    solution rebuilt with all non-pivot bits at zero.
    """

    order = np.argsort(llr, kind="stable").astype(np.int64)
    work = packed_h.copy()
    aug = n_cols
    aug_w, aug_b = aug >> 6, np.uint64(aug & 63)
    hit_rows = np.flatnonzero(syndrome.astype(bool))
    if hit_rows.size:
        work[hit_rows, aug_w] ^= (_ONE << aug_b)

    m = work.shape[0]
    pivot_cols = np.empty(rank, dtype=np.int64)
    pivot_pos = np.empty(rank, dtype=np.int64)
    r = 0
    closed_at = -1
    for pos in range(order.shape[0]):
        j = int(order[pos])
        w, b = j >> 6, np.uint64(j & 63)
        col = (work[r:, w] >> b) & _ONE
        nz = np.flatnonzero(col)
        if nz.size == 0:
            continue
        pr = r + int(nz[0])
        if pr != r:
            work[[r, pr]] = work[[pr, r]]
        hits = ((work[:, w] >> b) & _ONE).astype(bool)
        hits[r] = False
        if hits.any():
            work[hits] ^= work[r]
        pivot_cols[r] = j
        pivot_pos[r] = pos
        r += 1
        if r == rank:
            closed_at = pos
            break
        if r == m:
            break

    if r < rank:
        return None

    # With every non-pivot bit held at zero, each reduced row states the value
    # of its own pivot bit directly.
    solution = np.zeros(n_cols, dtype=np.uint8)
    rhs = ((work[:rank, aug_w] >> aug_b) & _ONE).astype(np.uint8)
    solution[pivot_cols] = rhs

    # Row operations touched whole rows, so every column of `work` -- including
    # the ones the scan never reached -- now expresses that column in the pivot
    # basis. That is what makes the OSD-CS candidate landscape recoverable from a
    # single elimination; see `osd_cs_landscape`.
    return {
        "order": order,
        "pivot_cols": pivot_cols,
        "pivot_pos": pivot_pos,
        "closed_at": closed_at,
        "solution": solution,
        "reduced": work[:rank],
    }


def extract_columns(packed: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Pull selected bit-packed columns as a dense uint8 matrix (n_rows x |cols|)."""

    cols = np.asarray(cols, dtype=np.int64)
    if cols.size == 0:
        return np.zeros((packed.shape[0], 0), dtype=np.uint8)
    words = cols >> 6
    shifts = np.asarray(cols & 63, dtype=np.uint64)
    return ((packed[:, words] >> shifts) & _ONE).astype(np.uint8)


def free_columns(info: Dict[str, np.ndarray], n_cols: int) -> np.ndarray:
    """Free columns in OSD-CS visit order: ascending LPR, non-pivots first."""

    is_pivot = np.zeros(n_cols, dtype=bool)
    is_pivot[info["pivot_cols"]] = True
    order = info["order"]
    return order[~is_pivot[order]]


def unpack_columns(packed_rows: np.ndarray, n_cols: int) -> np.ndarray:
    """Expand bit-packed uint64 rows into a dense uint8 matrix of `n_cols`.

    The packing puts global bit j at word j>>6, bit j&63. On a little-endian
    machine a uint8 view places that at byte (j>>6)*8 + ((j&63)>>3), bit j&7,
    which is exactly what `unpackbits(..., bitorder="little")` reverses.
    """

    as_bytes = np.ascontiguousarray(packed_rows).view(np.uint8)
    bits = np.unpackbits(as_bytes, axis=1, bitorder="little")
    return bits[:, :n_cols]


def tie_fraction(llr: np.ndarray) -> float:
    """Fraction of columns sharing an LPR value with another column."""

    vals, counts = np.unique(llr, return_counts=True)
    return float(counts[counts > 1].sum() / llr.size)


def verify_against_decoder(
    packed_h: np.ndarray,
    n_cols: int,
    rank: int,
    syndromes: np.ndarray,
    decoder,
) -> Dict[str, float]:
    """Check the reconstruction reproduces the decoder's own OSD-0 output.

    Only shots where BP fails to converge are meaningful: upstream short-circuits
    to the BP hard decision when it converges, and never enters OSD at all.
    """

    n_checked = 0
    n_match = 0
    n_converged = 0
    n_degenerate = 0
    ties = []
    for syndrome in syndromes:
        decoder.decode(syndrome.astype(np.uint8))
        if bool(decoder.converge):
            n_converged += 1
            continue
        llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
        ref = np.asarray(decoder.osd0_decoding, dtype=np.uint8) % 2
        info = osd0_information_set(packed_h, syndrome, llr, n_cols, rank)
        if info is None:
            n_degenerate += 1
            continue
        n_checked += 1
        n_match += int(np.array_equal(info["solution"], ref))
        ties.append(tie_fraction(llr))
    return {
        "n_checked": n_checked,
        "n_match": n_match,
        "match_rate": (n_match / n_checked) if n_checked else float("nan"),
        "n_bp_converged": n_converged,
        "n_rank_deficient": n_degenerate,
        "mean_tie_fraction": float(np.mean(ties)) if ties else float("nan"),
    }
