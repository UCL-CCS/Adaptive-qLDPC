"""qLDPC code construction utilities for failure prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class CSSCode:
    """Sparse CSS parity-check matrices and basic metadata."""

    hx: "csr_matrix"
    hz: "csr_matrix"
    name: str
    n: int
    k: int | None = None
    d: int | None = None


def _require_scipy_sparse():
    try:
        from scipy import sparse
    except ModuleNotFoundError as exc:
        raise RuntimeError("scipy is required for qLDPC code construction") from exc
    return sparse


def _shift_matrix(rows: int, cols: int, dx: int = 0, dy: int = 0):
    """Permutation matrix for x/y shifts on a rows-by-cols torus."""

    sparse = _require_scipy_sparse()
    n = rows * cols
    source = np.arange(n, dtype=np.int64)
    r = source // cols
    c = source % cols
    rr = (r + dx) % rows
    cc = (c + dy) % cols
    target = rr * cols + cc
    data = np.ones(n, dtype=np.uint8)
    return sparse.csr_matrix((data, (target, source)), shape=(n, n), dtype=np.uint8)


def _bb_polynomial_matrix(
    rows: int,
    cols: int,
    terms: Iterable[Tuple[int, int]],
):
    """Binary matrix for a bivariate bicycle polynomial on a torus."""

    sparse = _require_scipy_sparse()
    n = rows * cols
    mat = sparse.csr_matrix((n, n), dtype=np.uint8)
    for dx, dy in terms:
        mat = mat + _shift_matrix(rows, cols, dx=dx, dy=dy)
    mat.data %= 2
    mat.eliminate_zeros()
    return mat.astype(np.uint8)


def bivariate_bicycle_144_12_12() -> CSSCode:
    """
    Construct the standard [[144,12,12]] bivariate bicycle CSS code.

    The commonly used Bravyi et al. 2024 instance has a 12-by-6 torus, n=2lm,
    and polynomials A=x^3+y+y^2, B=y^3+x+x^2. The matrices are:
      H_X = [A, B]
      H_Z = [B^T, A^T]
    """

    sparse = _require_scipy_sparse()
    rows, cols = 12, 6
    a_terms = [(3, 0), (0, 1), (0, 2)]
    b_terms = [(0, 3), (1, 0), (2, 0)]
    a = _bb_polynomial_matrix(rows, cols, a_terms)
    b = _bb_polynomial_matrix(rows, cols, b_terms)
    hx = sparse.hstack([a, b], format="csr", dtype=np.uint8)
    hz = sparse.hstack([b.T, a.T], format="csr", dtype=np.uint8)
    return CSSCode(hx=hx, hz=hz, name="bb_144_12_12", n=144, k=12, d=12)


def bivariate_bicycle_72_12_6() -> CSSCode:
    """
    Construct the smaller [[72,12,6]] bivariate bicycle CSS code.

    This is the 6-by-6 torus version of the same BB polynomial family used by
    the [[144,12,12]] code:
      A=x^3+y+y^2, B=y^3+x+x^2
      H_X=[A,B], H_Z=[B^T,A^T]
    """

    sparse = _require_scipy_sparse()
    rows, cols = 6, 6
    a_terms = [(3, 0), (0, 1), (0, 2)]
    b_terms = [(0, 3), (1, 0), (2, 0)]
    a = _bb_polynomial_matrix(rows, cols, a_terms)
    b = _bb_polynomial_matrix(rows, cols, b_terms)
    hx = sparse.hstack([a, b], format="csr", dtype=np.uint8)
    hz = sparse.hstack([b.T, a.T], format="csr", dtype=np.uint8)
    return CSSCode(hx=hx, hz=hz, name="bb_72_12_6", n=72, k=12, d=6)


def parity_check_rank_mod2(matrix) -> int:
    """Compute GF(2) rank for small parity-check sanity checks."""

    arr = matrix.toarray().astype(np.uint8) % 2 if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=np.uint8) % 2
    m, n = arr.shape
    rank = 0
    col = 0
    while rank < m and col < n:
        pivot = np.where(arr[rank:, col] == 1)[0]
        if len(pivot) == 0:
            col += 1
            continue
        pivot_row = rank + int(pivot[0])
        if pivot_row != rank:
            arr[[rank, pivot_row]] = arr[[pivot_row, rank]]
        rows = np.where(arr[:, col] == 1)[0]
        rows = rows[rows != rank]
        arr[rows] ^= arr[rank]
        rank += 1
        col += 1
    return int(rank)


def css_commutes(code: CSSCode) -> bool:
    """Return whether H_X H_Z^T = 0 over GF(2)."""

    product = code.hx @ code.hz.T
    if product.nnz == 0:
        return True
    return bool(np.all((product.data % 2) == 0))

