"""Zero-HQC screen: H2-compatible codes for Adaptive Decoder V1.

Local Stim only. Does not submit Nexus / H2 hardware jobs.

HQC estimate uses the published formula already in this repo:
    HQC = 5 + (shots/5000) * (N1q + 10*N2q + 5*Nm)
with CX counted as one 2q gate and Rz excluded from N1q.

Usage:
    python -m failure_prediction.h2_adaptive_screen
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from failure_prediction.adaptive_v1 import evaluate_batch
from failure_prediction.adaptive_v1_analysis import auroc, wilson
from failure_prediction.osd_internals import gf2_rank_packed, pack_matrix
from failure_prediction.qldpc_circuit import (
    build_z_memory_circuit,
    detector_error_model_to_matrices,
)
from failure_prediction.qldpc_codes import (
    CSSCode,
    _bb_polynomial_matrix,
    css_commutes,
    parity_check_rank_mod2,
)

# H2-1 two-qubit fault probability (2025-04-30 hardware noise_specs).
P_H2 = 0.00105
OUT_DIR = "outputs/failure_prediction/h2_screen"
SHOTS = 4000
SEED = 11


def _row_basis_csr(mat):
    arr = mat.toarray().astype(np.uint8) % 2
    m, n = arr.shape
    rank = 0
    for col in range(n):
        pivot = np.where(arr[rank:, col] == 1)[0]
        if len(pivot) == 0:
            continue
        pr = rank + int(pivot[0])
        if pr != rank:
            arr[[rank, pr]] = arr[[pr, rank]]
        hits = np.where(arr[:, col] == 1)[0]
        hits = hits[hits != rank]
        arr[hits] ^= arr[rank]
        rank += 1
        if rank == m:
            break
    return sparse.csr_matrix(arr[:rank], dtype=np.uint8)


def steane() -> CSSCode:
    h = np.array(
        [[0, 0, 0, 1, 1, 1, 1], [0, 1, 1, 0, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1]],
        dtype=np.uint8,
    )
    hx = sparse.csr_matrix(h)
    return CSSCode(hx=hx, hz=hx.copy(), name="steane_7_1_3", n=7, k=1, d=3)


def rep3() -> CSSCode:
    hz = sparse.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
    hx = sparse.csr_matrix((0, 3), dtype=np.uint8)
    return CSSCode(hx=hx, hz=hz, name="rep3_bitflip", n=3, k=1, d=3)


def bb_12_2() -> CSSCode:
    a = _bb_polynomial_matrix(3, 2, [(1, 0), (0, 1)])
    b = _bb_polynomial_matrix(3, 2, [(0, 1), (1, 0)])
    hx = _row_basis_csr(sparse.hstack([a, b], format="csr", dtype=np.uint8))
    hz = _row_basis_csr(sparse.hstack([b.T, a.T], format="csr", dtype=np.uint8))
    n = hx.shape[1]
    k = n - parity_check_rank_mod2(hx) - parity_check_rank_mod2(hz)
    return CSSCode(hx=hx, hz=hz, name="bb_12_2", n=n, k=k)


def bb_18_4() -> CSSCode:
    a = _bb_polynomial_matrix(3, 3, [(0, 0), (1, 0), (0, 1)])
    b = _bb_polynomial_matrix(3, 3, [(0, 0), (2, 0), (0, 2)])
    hx = _row_basis_csr(sparse.hstack([a, b], format="csr", dtype=np.uint8))
    hz = _row_basis_csr(sparse.hstack([b.T, a.T], format="csr", dtype=np.uint8))
    n = hx.shape[1]
    k = n - parity_check_rank_mod2(hx) - parity_check_rank_mod2(hz)
    return CSSCode(hx=hx, hz=hz, name="bb_18_4", n=n, k=k)


def hqc(n1: int, n2: int, nm: int, shots: int) -> float:
    return 5.0 + (shots / 5000.0) * (n1 + 10 * n2 + 5 * nm)


def count_stim_gates(circuit) -> Dict[str, int]:
    n1 = n2 = nm = n_reset = 0
    oneq = {"H", "S", "S_DAG", "X", "Y", "Z", "SQRT_X", "SQRT_X_DAG", "I"}
    skip = {
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "TICK",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "DEPOLARIZE1",
        "DEPOLARIZE2",
        "X_ERROR",
        "Z_ERROR",
        "E",
        "ELSE_CORRELATED_ERROR",
    }
    for op in circuit.flattened():
        name = op.name
        if name in skip:
            continue
        n_t = len(op.targets_copy())
        if name in ("M", "MR", "MX", "MY", "MZ"):
            nm += max(n_t, 1)
        elif name in ("R", "RX", "RY"):
            n_reset += max(n_t, 1)
        elif name in ("CX", "CY", "CZ", "SWAP", "ISWAP", "CNOT"):
            n2 += max(n_t // 2, 1)
        elif name == "RZ":
            continue
        elif name in oneq:
            n1 += max(n_t, 1)
    return {"n1": n1, "n2": n2, "nm": nm, "n_reset": n_reset}


def surface_d3(p: float, rounds: int):
    import stim

    kwargs = dict(distance=3, rounds=rounds)
    if p:
        kwargs.update(
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
        )
    return stim.Circuit.generated("surface_code:rotated_memory_z", **kwargs)


def css_meta(code: CSSCode) -> dict:
    n_anc = int(code.hz.shape[0])
    nq = code.n + n_anc
    return {
        "name": code.name,
        "kind": "css_z_memory",
        "parameters": f"[[{code.n},{code.k},{code.d if code.d else '?'}]]",
        "n_data": int(code.n),
        "n_z_checks": n_anc,
        "n_qubits_simultaneous": nq,
        "fits_h2_emulator_26": nq <= 26,
        "fits_h2_hardware_56": nq <= 56,
        "commutes": bool(css_commutes(code)),
    }


def decode_circuit(circuit, shots: int, seed: int) -> dict:
    from ldpc import BpOsdDecoder

    dem = circuit.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True
    )
    matrices = detector_error_model_to_matrices(dem)
    h = matrices.h
    n_cols = h.shape[1]
    packed = pack_matrix(h, n_cols=n_cols, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n_cols)
    log_inv_p = np.log(1.0 / matrices.error_probs)
    logicals = matrices.logicals.astype(np.int64)
    decoder = BpOsdDecoder(
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=20,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    syn, actual = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    table = evaluate_batch(
        decoder,
        syn.astype(np.uint8),
        actual.astype(np.uint8),
        packed,
        log_inv_p,
        logicals,
        n_cols,
        rank,
        progress_every=0,
    )
    n = shots
    fail0 = table["fail_osd0"]
    fail_full = table["fail_full"]
    useful = (fail0 == 1) & (fail_full == 0)
    k0, kfull = int(fail0.sum()), int(fail_full.sum())
    lo0, hi0 = wilson(k0, n)
    lof, hif = wilson(kfull, n)
    ler0 = k0 / n
    ler_full = kfull / n
    gap = ler0 / ler_full if ler_full > 0 else float("inf")
    return {
        "n": n,
        "n_detectors": int(h.shape[0]),
        "n_error_mechs": int(n_cols),
        "osd_rank": int(rank),
        "n_free": int(n_cols - rank),
        "bp_converged": float(table["bp_converged"].mean()),
        "ler_osd0": ler0,
        "ler_osd0_ci": [lo0, hi0],
        "ler_k500": float(table["fail_k500"].mean()),
        "ler_k1000": float(table["fail_k1000"].mean()),
        "ler_full_1fv": ler_full,
        "ler_full_ci": [lof, hif],
        "gap_osd0_over_full": gap,
        "n_fail_osd0": k0,
        "n_fail_full": kfull,
        "n_beneficial": int(useful.sum()),
        "auroc_fail": auroc(fail0, table["d_h"]),
        "auroc_beneficial": auroc(useful, table["d_h"]),
        "d_h_mean": float(table["d_h"].mean()),
        "d_h_mean_fail": float(table["d_h"][fail0.astype(bool)].mean()) if k0 else float("nan"),
        "d_h_mean_ok": float(table["d_h"][~fail0.astype(bool)].mean()) if k0 < n else float("nan"),
    }


def eval_css(code: CSSCode, p: float, rounds: int, shots: int, seed: int) -> dict:
    noisy = build_z_memory_circuit(code, p=p, rounds=rounds)
    quiet = build_z_memory_circuit(code, p=0.0, rounds=rounds)
    gates = count_stim_gates(quiet)
    meta = css_meta(code)
    dec = decode_circuit(noisy, shots, seed)
    row = {
        **meta,
        "p": p,
        "rounds": rounds,
        "noise": "stim_circuit_level_single_p",
        **gates,
        **dec,
    }
    row["hqc"] = {
        str(s): hqc(gates["n1"], gates["n2"], gates["nm"], s) for s in (100, 300, 500, 1000)
    }
    return row


def eval_surface(p: float, rounds: int, shots: int, seed: int) -> dict:
    noisy = surface_d3(p, rounds)
    quiet = surface_d3(0.0, rounds)
    nq = quiet.num_qubits
    gates = count_stim_gates(quiet)
    dec = decode_circuit(noisy, shots, seed)
    row = {
        "name": "surface_d3",
        "kind": "stim_rotated_memory_z",
        "parameters": "[[9,1,3]]",
        "n_data": 9,
        "n_z_checks": 8,
        "n_qubits_simultaneous": int(nq),
        "fits_h2_emulator_26": nq <= 26,
        "fits_h2_hardware_56": nq <= 56,
        "commutes": True,
        "p": p,
        "rounds": rounds,
        "noise": "stim_generated_si_style_single_p",
        **gates,
        **dec,
    }
    row["hqc"] = {
        str(s): hqc(gates["n1"], gates["n2"], gates["nm"], s) for s in (100, 300, 500, 1000)
    }
    return row


def fmt_row(r: dict) -> str:
    gap = r["gap_osd0_over_full"]
    gap_s = f"{gap:.2f}x" if np.isfinite(gap) else "inf"
    return (
        f"{r['name']:14s} r={r['rounds']:2d} nq={r['n_qubits_simultaneous']:2d}  "
        f"OSD0={r['ler_osd0']:.4f}[{r['n_fail_osd0']}]  "
        f"1FV={r['ler_full_1fv']:.4f}[{r['n_fail_full']}]  gap={gap_s:6s}  "
        f"AUROC_fail={r['auroc_fail']:.3f}  ben={r['auroc_beneficial']:.3f}  "
        f"k_free={r['n_free']}  HQC100={r['hqc']['100']:.0f} HQC1000={r['hqc']['1000']:.0f}"
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    codes = [rep3(), steane(), bb_12_2(), bb_18_4()]
    print("=== code constructions ===")
    for c in codes:
        m = css_meta(c)
        print(
            f"  {m['name']:14s} {m['parameters']:12s} nq={m['n_qubits_simultaneous']:2d} "
            f"emu26={m['fits_h2_emulator_26']} k={c.k} commute={m['commutes']}"
        )
    print("  surface_d3      [[9,1,3]]     nq=17 emu26=True  (stim generated)")
    print(f"\nH2-like p={P_H2}  shots={SHOTS}  seed={SEED}  (local Stim, 0 HQC)\n")

    rows: List[dict] = []
    schedule = [
        ("rep3_bitflip", 6),
        ("rep3_bitflip", 12),
        ("steane_7_1_3", 6),
        ("steane_7_1_3", 12),
        ("steane_7_1_3", 18),
        ("steane_7_1_3", 24),
        ("bb_12_2", 6),
        ("bb_12_2", 12),
        ("bb_12_2", 18),
        ("bb_18_4", 6),
        ("bb_18_4", 12),
        ("bb_18_4", 18),
        ("surface_d3", 6),
        ("surface_d3", 12),
        ("surface_d3", 18),
        ("surface_d3", 24),
    ]
    lookup = {c.name: c for c in codes}
    for name, rounds in schedule:
        print(f"-- {name}  r={rounds}", flush=True)
        if name == "surface_d3":
            row = eval_surface(P_H2, rounds, SHOTS, SEED)
        else:
            row = eval_css(lookup[name], P_H2, rounds, SHOTS, SEED)
        rows.append(row)
        print("   " + fmt_row(row), flush=True)

    path = os.path.join(OUT_DIR, "screen_results.json")
    with open(path, "w") as fh:
        json.dump({"p": P_H2, "shots": SHOTS, "seed": SEED, "rows": rows}, fh, indent=2)
        fh.write("\n")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
