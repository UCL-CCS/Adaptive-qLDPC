"""Literature quantum radial / lifted-product CSS code.

Instance: Scruby, Hillmann, Roffe (arXiv:2406.14445) companion PCMs
`90_8_10` with (r, s) = (3, 5), from tRowans/radial-codes-public.

This is not a BB code and not the Hamming-HGP instance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from failure_prediction.qldpc_codes import CSSCode, _require_scipy_sparse, parity_check_rank_mod2

DATA = Path(__file__).resolve().parent / "data" / "radial_90_8_10"


def _load_csv(name: str) -> np.ndarray:
    return np.loadtxt(DATA / name, delimiter=",", dtype=np.uint8)


def radial_90_8() -> CSSCode:
    """Load the published (r, s)=(3, 5) radial / lifted-product CSS code.

    Family: n = 2 r^2 s = 90, k = 2 (r-1)^2 = 8, d ≤ 2s = 10.
    Distance is not independently verified here.
    """

    sparse = _require_scipy_sparse()
    hx = sparse.csr_matrix(_load_csv("hx.csv"), dtype=np.uint8)
    hz = sparse.csr_matrix(_load_csv("hz.csv"), dtype=np.uint8)
    n = int(hx.shape[1])
    return CSSCode(hx=hx, hz=hz, name="radial_90_8", n=n, k=8, d=None)


def published_logicals() -> Dict[str, np.ndarray]:
    return {
        "x_logicals": _load_csv("lx.csv"),
        "z_logicals": _load_csv("lz.csv"),
    }


def construction_meta() -> dict:
    return {
        "family": "quantum_radial_lifted_product",
        "literature": "Scruby, Hillmann, Roffe, arXiv:2406.14445 / PRX Quantum",
        "reference_implementation": "https://github.com/tRowans/radial-codes-public",
        "reference_commit": "12e67584db0fae3dbf8b4447596066e26ad2514c",
        "pcm_dir": "PCMs/90_8_10",
        "r": 3,
        "s": 5,
        "n_formula": 2 * 3 * 3 * 5,
        "k_formula": 2 * (3 - 1) ** 2,
        "d_family_bound": 2 * 5,
        "d": "unknown",
        "d_note": (
            "Source directory is named 90_8_10 and the family bound is d≤2s=10. "
            "Distance is not independently verified in this repository."
        ),
        "selection_reason": (
            "Smallest published PCM in the paper companion repository. "
            "Chosen before any decoder or headroom evaluation."
        ),
    }


def code_weight_stats(code: CSSCode) -> dict:
    hx, hz = code.hx.tocsr(), code.hz.tocsr()

    def row_w(m):
        return np.diff(m.indptr).astype(int)

    def col_w(m):
        return np.asarray(m.sum(axis=0)).ravel().astype(int)

    def summ(w):
        return {
            "min": int(w.min()) if w.size else None,
            "max": int(w.max()) if w.size else None,
            "mean": float(w.mean()) if w.size else None,
            "unique": sorted(int(x) for x in np.unique(w)),
        }

    return {
        "hx_shape": list(hx.shape),
        "hz_shape": list(hz.shape),
        "hx_rank": parity_check_rank_mod2(hx),
        "hz_rank": parity_check_rank_mod2(hz),
        "hx_row_weights": summ(row_w(hx)),
        "hz_row_weights": summ(row_w(hz)),
        "hx_col_degrees": summ(col_w(hx)),
        "hz_col_degrees": summ(col_w(hz)),
    }


def validate_logicals(code: CSSCode, x_logs: np.ndarray, z_logs: np.ndarray) -> dict:
    hx = code.hx.toarray().astype(np.uint8) % 2
    hz = code.hz.toarray().astype(np.uint8) % 2
    issues = []
    kx, kz = int(x_logs.shape[0]), int(z_logs.shape[0])
    if kx != kz:
        issues.append(f"k_X={kx} != k_Z={kz}")
    if code.k is not None and kx != code.k:
        issues.append(f"k_X={kx} != formula k={code.k}")
    if np.any((hz @ x_logs.T) % 2):
        issues.append("some X logicals fail Hz X^T = 0")
    if np.any((hx @ z_logs.T) % 2):
        issues.append("some Z logicals fail Hx Z^T = 0")
    pair = (x_logs.astype(np.uint8) @ z_logs.astype(np.uint8).T) % 2
    rank_pair = parity_check_rank_mod2(pair)
    if rank_pair != min(kx, kz):
        issues.append(f"X/Z pairing matrix rank {rank_pair} != {min(kx, kz)}")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "k_x": kx,
        "k_z": kz,
        "pairing_rank": int(rank_pair),
        "n_anticommuting_pairs": int(pair.sum()),
    }


def cnot_schedule(code: CSSCode) -> dict:
    hx, hz = code.hx.tocsr(), code.hz.tocsr()
    n = code.n
    z_checks = []
    for i in range(hz.shape[0]):
        z_checks.append(
            {
                "z_check": i,
                "ancilla": n + i,
                "data": [int(q) for q in hz.indices[hz.indptr[i] : hz.indptr[i + 1]]],
            }
        )
    x_checks = []
    for i in range(hx.shape[0]):
        x_checks.append(
            {
                "x_check": i,
                "ancilla": n + hz.shape[0] + i,
                "data": [int(q) for q in hx.indices[hx.indptr[i] : hx.indptr[i + 1]]],
            }
        )
    return {
        "z_memory_builder": "failure_prediction.qldpc_circuit.build_z_memory_circuit",
        "full_css_builder": "failure_prediction.h2_full_css.build_css_memory_circuit",
        "note": (
            "CNOT order is CSR column-index order of H_X / H_Z, matching the "
            "Adaptive V1 CSS/Z-memory extractors. The authors' overlapping radial "
            "CX schedule (radial-codes-public/decoding/circuit_stuff/circuit.py) "
            "is a different circuit family and is not used here."
        ),
        "n_data": int(n),
        "n_z_anc": int(hz.shape[0]),
        "n_x_anc_fullcss": int(hx.shape[0]),
        "z_checks": z_checks,
        "x_checks": x_checks,
    }
