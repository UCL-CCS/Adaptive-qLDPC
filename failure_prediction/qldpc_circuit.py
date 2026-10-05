"""Stim circuit-level BB-code memory experiments and DEM decoding."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import OUTPUT_DIR
from failure_prediction.qldpc_codes import (
    CSSCode,
    bivariate_bicycle_72_12_6,
    bivariate_bicycle_144_12_12,
)

BP_FEATURE_NAMES = [
    "bp_converged",
    "bp_iterations",
    "osd_triggered",
    "llr_min",
    "llr_mean",
    "llr_std",
    "llr_entropy",
]


def _binary_entropy(prob: np.ndarray) -> float:
    prob = np.clip(np.asarray(prob, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return float(np.mean(-(prob * np.log2(prob) + (1.0 - prob) * np.log2(1.0 - prob))))


@dataclass
class DemMatrices:
    h: "csr_matrix"
    logicals: np.ndarray
    error_probs: np.ndarray


def _require_stim():
    try:
        import stim
    except ModuleNotFoundError as exc:
        raise RuntimeError("stim is required for circuit-level qLDPC experiments") from exc
    return stim


def _require_scipy_sparse():
    try:
        from scipy import sparse
    except ModuleNotFoundError as exc:
        raise RuntimeError("scipy is required for DEM matrix conversion") from exc
    return sparse


def _gf2_row_basis(rows: np.ndarray) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.uint8).copy() % 2
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    m, n = arr.shape
    rank = 0
    for col in range(n):
        pivot = np.where(arr[rank:, col] == 1)[0]
        if len(pivot) == 0:
            continue
        pivot_row = rank + int(pivot[0])
        if pivot_row != rank:
            arr[[rank, pivot_row]] = arr[[pivot_row, rank]]
        rows_to_clear = np.where(arr[:, col] == 1)[0]
        rows_to_clear = rows_to_clear[rows_to_clear != rank]
        arr[rows_to_clear] ^= arr[rank]
        rank += 1
        if rank == m:
            break
    return arr[:rank].copy()


def _gf2_rank(rows: np.ndarray) -> int:
    return int(_gf2_row_basis(rows).shape[0])


def _gf2_nullspace(matrix) -> np.ndarray:
    arr = matrix.toarray().astype(np.uint8) % 2 if hasattr(matrix, "toarray") else np.asarray(matrix, dtype=np.uint8) % 2
    m, n = arr.shape
    rref = arr.copy()
    pivots: List[int] = []
    rank = 0
    for col in range(n):
        pivot = np.where(rref[rank:, col] == 1)[0]
        if len(pivot) == 0:
            continue
        pivot_row = rank + int(pivot[0])
        if pivot_row != rank:
            rref[[rank, pivot_row]] = rref[[pivot_row, rank]]
        rows_to_clear = np.where(rref[:, col] == 1)[0]
        rows_to_clear = rows_to_clear[rows_to_clear != rank]
        rref[rows_to_clear] ^= rref[rank]
        pivots.append(col)
        rank += 1
        if rank == m:
            break

    free_cols = [c for c in range(n) if c not in set(pivots)]
    basis = []
    for free_col in free_cols:
        v = np.zeros(n, dtype=np.uint8)
        v[free_col] = 1
        for row, pivot_col in enumerate(pivots):
            if rref[row, free_col]:
                v[pivot_col] = 1
        basis.append(v)
    return np.asarray(basis, dtype=np.uint8)


def z_logical_basis(code: CSSCode) -> np.ndarray:
    """Return k representatives of Z logicals: ker(H_X) / row(H_Z)."""

    null_basis = _gf2_nullspace(code.hx)
    stabilizers = code.hz.toarray().astype(np.uint8) % 2
    current = _gf2_row_basis(stabilizers)
    logicals = []
    current_rank = _gf2_rank(current)
    for v in null_basis:
        trial = np.vstack([current, v])
        trial_rank = _gf2_rank(trial)
        if trial_rank > current_rank:
            logicals.append(v.copy())
            current = _gf2_row_basis(trial)
            current_rank = trial_rank
        if code.k is not None and len(logicals) == code.k:
            break
    return np.asarray(logicals, dtype=np.uint8)


def x_logical_basis(code: CSSCode) -> np.ndarray:
    """Return k representatives of X logicals: ker(H_Z) / row(H_X)."""

    null_basis = _gf2_nullspace(code.hz)
    stabilizers = code.hx.toarray().astype(np.uint8) % 2
    current = _gf2_row_basis(stabilizers)
    logicals = []
    current_rank = _gf2_rank(current)
    for v in null_basis:
        trial = np.vstack([current, v])
        trial_rank = _gf2_rank(trial)
        if trial_rank > current_rank:
            logicals.append(v.copy())
            current = _gf2_row_basis(trial)
            current_rank = trial_rank
        if code.k is not None and len(logicals) == code.k:
            break
    return np.asarray(logicals, dtype=np.uint8)


def build_z_memory_circuit(code: CSSCode, p: float, rounds: int):
    """
    Build a circuit-level noisy Z-basis memory experiment.

    Noise model includes reset errors, CNOT depolarization, measurement errors,
    and one idle depolarization layer on data qubits per syndrome round.
    """

    stim = _require_stim()
    circuit = stim.Circuit()
    n = code.n
    z_checks = code.hz.tocsr()
    z_ancillas = list(range(n, n + z_checks.shape[0]))
    data_qubits = list(range(n))
    meas_count = 0
    z_round_measurements: List[List[int]] = []

    def rec(abs_index: int):
        return stim.target_rec(abs_index - meas_count)

    circuit.append("R", data_qubits)
    if p:
        circuit.append("X_ERROR", data_qubits, p)
    circuit.append("TICK")

    for r in range(rounds):
        circuit.append("R", z_ancillas)
        if p:
            circuit.append("X_ERROR", z_ancillas, p)
        circuit.append("TICK")

        for check, anc in enumerate(z_ancillas):
            support = z_checks.indices[z_checks.indptr[check] : z_checks.indptr[check + 1]]
            for q in support:
                circuit.append("CX", [int(q), anc])
                if p:
                    circuit.append("DEPOLARIZE2", [int(q), anc], p)
            circuit.append("TICK")

        round_measurements = []
        for anc in z_ancillas:
            circuit.append("M", [anc], p)
            round_measurements.append(meas_count)
            meas_count += 1
        z_round_measurements.append(round_measurements)

        for check, abs_m in enumerate(round_measurements):
            if r == 0:
                circuit.append("DETECTOR", [rec(abs_m)])
            else:
                circuit.append("DETECTOR", [rec(abs_m), rec(z_round_measurements[r - 1][check])])

        if p:
            circuit.append("DEPOLARIZE1", data_qubits, p)
        circuit.append("TICK")

    final_data_measurements = []
    for q in data_qubits:
        circuit.append("M", [q], p)
        final_data_measurements.append(meas_count)
        meas_count += 1

    for check, last_abs_m in enumerate(z_round_measurements[-1]):
        support = z_checks.indices[z_checks.indptr[check] : z_checks.indptr[check + 1]]
        targets = [rec(last_abs_m)] + [rec(final_data_measurements[int(q)]) for q in support]
        circuit.append("DETECTOR", targets)

    logicals = z_logical_basis(code)
    if code.k is not None and logicals.shape[0] != code.k:
        raise ValueError(f"Expected {code.k} Z logicals, found {logicals.shape[0]}")
    for obs_idx, logical in enumerate(logicals):
        support = np.flatnonzero(logical)
        targets = [rec(final_data_measurements[int(q)]) for q in support]
        circuit.append("OBSERVABLE_INCLUDE", targets, obs_idx)
    return circuit


def detector_error_model_to_matrices(dem) -> DemMatrices:
    """Convert a Stim detector error model to BP+OSD parity-check matrices."""

    sparse = _require_scipy_sparse()
    dem = dem.flattened()
    det_rows: List[int] = []
    err_cols: List[int] = []
    logicals = []
    probs = []
    col = 0
    for inst in dem:
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        dets = []
        obs = []
        for target in inst.targets_copy():
            if target.is_relative_detector_id():
                dets.append(int(target.val))
            elif target.is_logical_observable_id():
                obs.append(int(target.val))
        for det in dets:
            det_rows.append(det)
            err_cols.append(col)
        logical_row = np.zeros(dem.num_observables, dtype=np.uint8)
        logical_row[obs] = 1
        logicals.append(logical_row)
        probs.append(min(max(p, 1e-12), 1.0 - 1e-12))
        col += 1
    h = sparse.csr_matrix(
        (np.ones(len(det_rows), dtype=np.uint8), (det_rows, err_cols)),
        shape=(dem.num_detectors, col),
        dtype=np.uint8,
    )
    return DemMatrices(
        h=h,
        logicals=np.asarray(logicals, dtype=np.uint8).T,
        error_probs=np.asarray(probs, dtype=np.float64),
    )


def _decode_one(decoder, logical_matrix: np.ndarray, syndrome: np.ndarray) -> Tuple[np.ndarray, dict, np.ndarray]:
    correction = np.asarray(decoder.decode(syndrome), dtype=np.uint8) % 2
    predicted = np.asarray(logical_matrix @ correction, dtype=np.uint8).ravel() % 2
    llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
    llr_abs = np.abs(llr).astype(np.float32)
    posterior_error_prob = 1.0 / (1.0 + np.exp(np.clip(np.abs(llr), -60, 60)))
    features = {
        "bp_converged": float(bool(decoder.converge)),
        "bp_iterations": float(decoder.iter),
        "osd_triggered": float(not bool(decoder.converge)),
        "llr_min": float(np.min(llr_abs)),
        "llr_mean": float(np.mean(llr_abs)),
        "llr_std": float(np.std(llr_abs)),
        "llr_entropy": _binary_entropy(posterior_error_prob),
    }
    return predicted, features, llr_abs


def _get_code(name: str) -> CSSCode:
    if name == "bb_72_12_6":
        return bivariate_bicycle_72_12_6()
    if name == "bb_144_12_12":
        return bivariate_bicycle_144_12_12()
    raise ValueError(f"Unknown code: {name}")


def generate_circuit_level_shot_table(
    code_name: str,
    p: float,
    shots: int,
    rounds: int,
    seed: int,
    output: str | None = None,
    max_iter: int = 50,
    osd_order: int = 2,
) -> str:
    """Sample a noisy Stim memory circuit and decode with DEM BP+OSD."""

    try:
        from ldpc import BpOsdDecoder
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install qLDPC dependencies with `pip install -r requirements-qldpc.txt`") from exc

    code = _get_code(code_name)
    circuit = build_z_memory_circuit(code, p=p, rounds=rounds)
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matrices = detector_error_model_to_matrices(dem)
    decoder = BpOsdDecoder(
        matrices.h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_cs",
        osd_order=osd_order,
    )

    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, actual_obs = sampler.sample(shots, separate_observables=True)
    syndromes = syndromes.astype(np.uint8)
    actual_obs = actual_obs.astype(np.uint8)
    predictions = np.zeros_like(actual_obs, dtype=np.uint8)
    feature_rows = np.zeros((shots, len(BP_FEATURE_NAMES)), dtype=np.float32)
    llr_vectors = None
    for i, syndrome in enumerate(syndromes):
        pred, features, llr_abs = _decode_one(decoder, matrices.logicals, syndrome)
        if llr_vectors is None:
            llr_vectors = np.zeros((shots, len(llr_abs)), dtype=np.float16)
        predictions[i] = pred
        feature_rows[i] = [features[name] for name in BP_FEATURE_NAMES]
        llr_vectors[i] = llr_abs.astype(np.float16)
    y_fail = np.any(predictions != actual_obs, axis=1).astype(np.uint8)

    if output is None:
        out_dir = os.path.join(OUTPUT_DIR, "failure_prediction", "qldpc_phase2", code.name)
        os.makedirs(out_dir, exist_ok=True)
        output = os.path.join(out_dir, f"circuit_zmem_p{p:g}_r{rounds}_n{shots}_seed{seed}.npz")
    else:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    dem_path = output.replace(".npz", ".dem")
    circuit_path = output.replace(".npz", ".stim")
    with open(dem_path, "w") as f:
        f.write(str(dem))
    with open(circuit_path, "w") as f:
        f.write(str(circuit))
    np.savez_compressed(
        output,
        syndromes=syndromes,
        y_fail=y_fail,
        actual_observables=actual_obs,
        predicted_observables=predictions,
        bp_features=feature_rows,
        bp_feature_names=np.asarray(BP_FEATURE_NAMES),
        bp_llr_vectors=llr_vectors,
        code=code.name,
        noise_model="stim_circuit_level_z_memory",
        p=float(p),
        rounds=int(rounds),
        seed=int(seed),
        dem_path=dem_path,
        circuit_path=circuit_path,
        num_observables=int(actual_obs.shape[1]),
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="bb_72_12_6", choices=["bb_72_12_6", "bb_144_12_12"])
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--osd-order", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = generate_circuit_level_shot_table(
        code_name=args.code,
        p=args.p,
        shots=args.shots,
        rounds=args.rounds,
        seed=args.seed,
        output=args.output,
        max_iter=args.max_iter,
        osd_order=args.osd_order,
    )
    print(path)

