"""Zero-HQC full-CSS bb_12_2 feasibility preflight.

Local Stim decode + codespace checks. Does NOT submit H2 hardware.
Nexus compile/cost lives in quantinuum/bb12_full_css_cost.py.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from failure_prediction.adaptive_v1 import evaluate_batch
from failure_prediction.adaptive_v1_analysis import auroc
from failure_prediction.h2_adaptive_screen import P_H2, SEED, bb_12_2, count_stim_gates, hqc
from failure_prediction.h2_full_css import (
    build_css_memory_circuit,
    measurement_layout,
    peek_codespace,
    stim_to_pytket_css,
)
from failure_prediction.osd_internals import gf2_rank_packed, pack_matrix
from failure_prediction.qldpc_circuit import (
    build_z_memory_circuit,
    detector_error_model_to_matrices,
    x_logical_basis,
    z_logical_basis,
)
from failure_prediction.qldpc_codes import css_commutes

OUT_DIR = "outputs/failure_prediction/h2_full_css_preflight"
ROUNDS = 12
SHOTS = 4000
NOISELESS_SHOTS = 256


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if np.isnan(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _decode(noisy, shots: int, seed: int, measurements_first: bool = True) -> dict:
    from ldpc import BpOsdDecoder

    if measurements_first:
        meas = noisy.compile_sampler(seed=seed).sample(shots).astype(np.uint8)
        conv = noisy.compile_m2d_converter()
        det, obs = conv.convert(measurements=meas.astype(bool), separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)
        det2, obs2 = noisy.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True
        )
        m2d_ok = bool(
            np.array_equal(det, det2.astype(np.uint8))
            and np.array_equal(obs, obs2.astype(np.uint8))
        )
    else:
        meas = None
        det, obs = noisy.compile_detector_sampler(seed=seed).sample(
            shots, separate_observables=True
        )
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)
        m2d_ok = None

    dem = noisy.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True
    )
    matrices = detector_error_model_to_matrices(dem)
    h = matrices.h
    n_cols = h.shape[1]
    packed = pack_matrix(h, n_cols=n_cols, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n_cols)
    decoder = BpOsdDecoder(
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=20,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    table = evaluate_batch(
        decoder,
        det,
        obs,
        packed,
        np.log(1.0 / matrices.error_probs),
        matrices.logicals.astype(np.int64),
        n_cols,
        rank,
        progress_every=0,
    )
    fail0 = table["fail_osd0"].astype(bool)
    failf = table["fail_full"].astype(bool)
    dh = table["d_h"]
    n_fail = int(fail0.sum())
    auc = auroc(fail0.astype(int), dh) if n_fail and n_fail < shots else float("nan")
    return {
        "n_detectors": int(h.shape[0]),
        "n_observables": int(obs.shape[1]),
        "n_meas": int(noisy.num_measurements),
        "n_qubits": int(noisy.num_qubits),
        "n_error_mechs": int(n_cols),
        "osd_rank": int(rank),
        "n_free": int(n_cols - rank),
        "ler_osd0": float(fail0.mean()),
        "ler_full_1fv": float(failf.mean()),
        "n_fail_osd0": n_fail,
        "n_fail_full": int(failf.sum()),
        "n_beneficial": int((fail0 & ~failf).sum()),
        "auroc_fail": float(auc) if auc == auc else None,
        "d_h_mean_ok": float(dh[~fail0].mean()) if (~fail0).any() else None,
        "d_h_mean_fail": float(dh[fail0].mean()) if n_fail else None,
        "bp_converged": float(table["bp_converged"].mean()),
        "m2d_matches_detector_sampler": m2d_ok,
        "labels_from_measurements_only": True,
        "expected_fail_count_300": float(fail0.mean() * 300),
        "n_meas_bits_sampled": None if meas is None else int(meas.shape[1]),
    }


def noiseless_check(quiet, shots: int = NOISELESS_SHOTS, seed: int = 0) -> dict:
    det, obs = quiet.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    meas = quiet.compile_sampler(seed=seed).sample(shots)
    n_hx = 5
    n_hz = 5
    first_x = meas[:, :n_hx]
    first_z = meas[:, n_hx : n_hx + n_hz]
    return {
        "shots": shots,
        "all_detectors_zero": bool(np.all(det == 0)),
        "all_observables_zero": bool(np.all(obs == 0)),
        "n_nonzero_det_shots": int(np.any(det, axis=1).sum()),
        "n_nonzero_obs_shots": int(np.any(obs, axis=1).sum()),
        "first_x_check_mean": first_x.mean(axis=0).tolist(),
        "first_z_check_mean": first_z.mean(axis=0).tolist(),
        "first_x_not_identically_zero": bool(np.any(first_x)),
        "first_z_not_identically_zero": bool(np.any(first_z)),
    }


def eval_design(code, basis: str, layout: str, shots: int, seed: int) -> dict:
    quiet = build_css_memory_circuit(
        code, p=0.0, rounds=ROUNDS, basis=basis, ancilla_layout=layout
    )
    noisy = build_css_memory_circuit(
        code, p=P_H2, rounds=ROUNDS, basis=basis, ancilla_layout=layout
    )
    gates = count_stim_gates(quiet)
    peek = peek_codespace(code, basis=basis)
    nl = noiseless_check(quiet)
    print(
        f"  noiseless det0={nl['all_detectors_zero']} obs0={nl['all_observables_zero']} "
        f"codespace={peek['after_extract_full_codespace']}",
        flush=True,
    )
    print(f"  decoding n={shots} ...", flush=True)
    dec = _decode(noisy, shots, seed)
    n_per = int(code.hx.shape[0]) + int(code.hz.shape[0])
    tket_ok, tket_err, tket_nq, tket_nb = True, None, None, None
    try:
        tket = stim_to_pytket_css(quiet, n_per_round=n_per, n_data=int(code.n), rounds=ROUNDS)
        tket_nq, tket_nb = tket.n_qubits, tket.n_bits
    except Exception as exc:
        tket_ok, tket_err = False, str(exc)
    mid_meas = gates["nm"] - int(code.n)
    return {
        "design": f"full_css_{basis.lower()}mem_{layout}",
        "basis": basis,
        "ancilla_layout": layout,
        "encoded_state": "|00>_L" if basis == "Z" else "|++>_L",
        "prep": "stabilizer measurement + Pauli-frame tracking",
        "x_checks": True,
        "z_checks": True,
        "n_x_checks": int(code.hx.shape[0]),
        "n_z_checks": int(code.hz.shape[0]),
        "rounds": ROUNDS,
        "p": P_H2,
        "shots": shots,
        "seed": seed,
        "n_qubits": int(quiet.num_qubits),
        "fits_h2_emulator_26": int(quiet.num_qubits) <= 26,
        "fits_h2_hardware_56": int(quiet.num_qubits) <= 56,
        **gates,
        "n_mid_circuit_meas": mid_meas,
        "formula_hqc": {
            str(s): hqc(gates["n1"], gates["n2"], gates["nm"], s)
            for s in (100, 300, 500, 1000)
        },
        "noiseless": nl,
        "codespace": {
            "data_reset_in_determined_eigenspace": peek["data_reset_in_determined_eigenspace"],
            "after_extract_full_codespace": peek["after_extract_full_codespace"],
            "before_extract": peek["before_extract"],
            "after_one_round": peek["after_one_round"],
        },
        "pytket_ok": tket_ok,
        "pytket_error": tket_err,
        "pytket_n_qubits": tket_nq,
        "pytket_n_bits": tket_nb,
        "layout": measurement_layout(code, ROUNDS),
        **dec,
        "dh_sector": "d_H^X (X-error / Z-logical DEM)" if basis == "Z" else "d_H^Z (Z-error / X-logical DEM)",
    }


def eval_z_only(code, shots: int, seed: int) -> dict:
    quiet = build_z_memory_circuit(code, p=0.0, rounds=ROUNDS)
    noisy = build_z_memory_circuit(code, p=P_H2, rounds=ROUNDS)
    gates = count_stim_gates(quiet)
    print("  decoding Z-only baseline ...", flush=True)
    dec = _decode(noisy, shots, seed)
    return {
        "design": "z_only",
        "basis": "Z",
        "ancilla_layout": "z_ancilla_only",
        "encoded_state": "|0>^12 (not full CSS codespace)",
        "prep": "data reset only",
        "x_checks": False,
        "z_checks": True,
        "n_x_checks": 0,
        "n_z_checks": int(code.hz.shape[0]),
        "rounds": ROUNDS,
        "p": P_H2,
        "shots": shots,
        "seed": seed,
        "n_qubits": int(quiet.num_qubits),
        "fits_h2_emulator_26": True,
        "fits_h2_hardware_56": True,
        **gates,
        "n_mid_circuit_meas": gates["nm"] - int(code.n),
        "formula_hqc": {
            str(s): hqc(gates["n1"], gates["n2"], gates["nm"], s)
            for s in (100, 300, 500, 1000)
        },
        "official_hqc_300": 221.0,
        "dh_sector": "d_H^X (Z-check DEM only)",
        **dec,
    }


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    code = bb_12_2()
    zlog = z_logical_basis(code)
    xlog = x_logical_basis(code)
    meta = {
        "code": "bb_12_2",
        "n": int(code.n),
        "k": int(code.k),
        "hx_rank": int(code.hx.shape[0]),
        "hz_rank": int(code.hz.shape[0]),
        "hx_equals_hz": bool(np.array_equal(code.hx.toarray() % 2, code.hz.toarray() % 2)),
        "css_commutes": bool(css_commutes(code)),
        "z_logical_supports": [np.flatnonzero(r).tolist() for r in zlog],
        "x_logical_supports": [np.flatnonzero(r).tolist() for r in xlog],
        "prep_choice": (
            "Stabilizer-measurement + Pauli-frame tracking. A Clifford encoder "
            "would add a dense encoding block before the 12 QEC rounds; the first "
            "X-extract (Z-mem) or Z-extract (X-mem) already projects into the "
            "codespace, so the encoder is strictly more gates for the same state."
        ),
        "p": P_H2,
        "rounds": ROUNDS,
        "shots": SHOTS,
        "seed": SEED,
        "hardware_submitted": False,
        "hqc_spent": 0,
    }
    print("bb_12_2 full-CSS preflight  p=%g r=%d n=%d" % (P_H2, ROUNDS, SHOTS), flush=True)
    print("hx==hz", meta["hx_equals_hz"], "commute", meta["css_commutes"], flush=True)

    rows: Dict[str, dict] = {}
    print("-- z_only baseline", flush=True)
    rows["z_only"] = eval_z_only(code, SHOTS, SEED)
    print(
        f"   OSD0={rows['z_only']['ler_osd0']:.4f} AUROC={rows['z_only']['auroc_fail']}",
        flush=True,
    )

    for layout in ("reuse", "separate"):
        for basis in ("Z", "X"):
            key = f"{basis}_{layout}"
            print(f"-- full CSS {basis}-memory layout={layout}", flush=True)
            rows[key] = eval_design(code, basis, layout, SHOTS, SEED)
            r = rows[key]
            print(
                f"   nq={r['n_qubits']} n2={r['n2']} nm={r['nm']} depth_stim_ticks=n/a "
                f"OSD0={r['ler_osd0']:.4f}[{r['n_fail_osd0']}] "
                f"1FV={r['ler_full_1fv']:.4f} AUROC={r['auroc_fail']} "
                f"m2d={r['m2d_matches_detector_sampler']}",
                flush=True,
            )

    out = {"meta": meta, "rows": rows}
    path = os.path.join(OUT_DIR, "full_css_sim.json")
    with open(path, "w") as fh:
        json.dump(_jsonable(out), fh, indent=2)
        fh.write("\n")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
