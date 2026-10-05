"""Zero-HQC preflight for native-H2 d_H validation on bb_12_2.

Local Stim only. Does not submit H2-1 / H2-1E / H2-2 / H2-2E jobs.

The hardware experiment (if later approved) records syndromes + final data
and labels OSD-0 success/failure offline. This script is the exact labelling
pipeline, using Stim measurement records instead of hardware bitstrings.

Usage (conda base, with stim+ldpc):
    python -m failure_prediction.h2_dh_preflight
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from failure_prediction.adaptive_v1 import evaluate_batch
from failure_prediction.adaptive_v1_analysis import auroc
from failure_prediction.h2_adaptive_screen import (
    P_H2,
    SEED,
    bb_12_2,
    count_stim_gates,
    css_meta,
    hqc,
)
from failure_prediction.osd_internals import gf2_rank_packed, pack_matrix
from failure_prediction.qldpc_circuit import (
    build_z_memory_circuit,
    detector_error_model_to_matrices,
    z_logical_basis,
)

OUT_DIR = "outputs/failure_prediction/h2_dh_preflight"
ROUNDS = 12
SHOTS = 4000


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


def stim_to_pytket(stim_circ, n_anc: int, n_data: int, rounds: int):
    """H2-legal pytket translation: one creg per syndrome round (≤64 bits)."""
    from pytket.circuit import Bit, Circuit

    nq = stim_circ.num_qubits
    n_meas = stim_circ.num_measurements
    circ = Circuit(nq)
    for r in range(rounds):
        circ.add_c_register(f"s{r}", n_anc)
    circ.add_c_register("d", n_data)
    bit_i = 0
    initialized = set()
    n_syn = n_anc * rounds

    def meas_bit(idx: int):
        if idx < n_syn:
            return Bit(f"s{idx // n_anc}", idx % n_anc)
        return Bit("d", idx - n_syn)

    for op in stim_circ:
        name = op.name
        tgts = [t.value for t in op.targets_copy()]
        if name in ("TICK", "DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS"):
            continue
        if name == "R":
            for q in tgts:
                if q in initialized:
                    circ.Reset(q)
                initialized.add(q)
            continue
        if name == "H":
            for q in tgts:
                circ.H(q)
        elif name == "CX":
            for i in range(0, len(tgts), 2):
                circ.CX(tgts[i], tgts[i + 1])
        elif name == "M":
            for q in tgts:
                circ.Measure(q, meas_bit(bit_i))
                bit_i += 1
        elif name == "MR":
            for q in tgts:
                circ.Measure(q, meas_bit(bit_i))
                circ.Reset(q)
                bit_i += 1
                initialized.add(q)
        else:
            raise ValueError(f"unsupported stim op for pytket translate: {name}")
    if bit_i != n_meas:
        raise RuntimeError(f"bit count mismatch {bit_i} vs {n_meas}")
    return circ


def logical_label_spec(code, quiet, noisy) -> dict:
    """Document how an OSD-0 success/failure label is computed from a shot."""
    z_log = z_logical_basis(code)
    n_anc = int(code.hz.shape[0])
    n_data = int(code.n)
    n_meas = quiet.num_measurements
    n_syn = n_anc * ROUNDS
    return {
        "encoded_state": "|0...0> on data (Stim R); Z-logicals ideally 0",
        "correction": "virtual/offline — no recovery gates in the physical circuit",
        "measurement_layout": {
            "ancilla_syndrome_bits": f"first {n_syn} bits: {ROUNDS} rounds × {n_anc} Z-checks",
            "final_data_bits": f"last {n_data} bits: Z-basis data readout",
            "n_meas": n_meas,
        },
        "detectors": (
            "Stim DETECTORs: round-0 = raw Z-checks; later rounds = consecutive "
            "Z-check XOR; final layer = last Z-checks XOR data support of each check. "
            "Reconstructed on hardware via stim.compile_m2d_converter()."
        ),
        "actual_logical": (
            "OBSERVABLE_INCLUDE = Z-logicals on final data bits. "
            "actual[k] = XOR of measured data bits on z_logical_basis[k]. "
            "Because the encoded state is |0>_L, a 1 is a logical Z flip."
        ),
        "predicted_logical": (
            "DEM BP@20+OSD-0 produces e_OSD0 on error mechanisms. "
            "pred = (DEM_logicals @ e_OSD0) mod 2."
        ),
        "osd0_fail": "any(pred != actual) over the k logical observables",
        "d_H": "Hamming distance between BP hard decision and OSD-0 on the DEM columns",
        "z_logical_supports": [np.flatnonzero(row).tolist() for row in z_log],
        "n_data": n_data,
        "n_ancilla": n_anc,
        "n_qubits": n_data + n_anc,
        "k": int(code.k),
        "rounds": ROUNDS,
        "dem_from": f"Stim circuit-level DEM at p={P_H2} (same as H2 screen)",
    }


def decode_from_measurements(noisy, measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    conv = noisy.compile_m2d_converter()
    det, obs = conv.convert(measurements=measurements.astype(bool), separate_observables=True)
    return det.astype(np.uint8), obs.astype(np.uint8)


def run_preflight(shots: int, seed: int) -> dict:
    from ldpc import BpOsdDecoder

    code = bb_12_2()
    quiet = build_z_memory_circuit(code, p=0.0, rounds=ROUNDS)
    noisy = build_z_memory_circuit(code, p=P_H2, rounds=ROUNDS)
    spec = logical_label_spec(code, quiet, noisy)
    gates = count_stim_gates(quiet)

    sampler = noisy.compile_sampler(seed=seed)
    measurements = sampler.sample(shots).astype(np.uint8)
    det, obs = decode_from_measurements(noisy, measurements)

    # Cross-check against the detector sampler used in the original screen.
    det2, obs2 = noisy.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    m2d_matches_detector_sampler = bool(
        np.array_equal(det, det2.astype(np.uint8)) and np.array_equal(obs, obs2.astype(np.uint8))
    )

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
    n_ben = int((fail0 & ~failf).sum())
    rng = np.random.default_rng(seed + 7)
    auroc_vals = np.empty(800)
    y = fail0.astype(int)
    for b in range(800):
        ix = rng.integers(0, shots, shots)
        auroc_vals[b] = auroc(y[ix], dh[ix])
    auc = auroc(y, dh)

    pytket_ok = True
    pytket_err = None
    nq_tket = None
    try:
        tket = stim_to_pytket(quiet, n_anc=int(code.hz.shape[0]), n_data=int(code.n), rounds=ROUNDS)
        nq_tket = tket.n_qubits
    except Exception as exc:  # pragma: no cover
        pytket_ok = False
        pytket_err = str(exc)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, "preflight_shots.npz"),
        measurements=measurements,
        detectors=det,
        actual_logical=obs,
        d_h=dh,
        fail_osd0=table["fail_osd0"],
        fail_full=table["fail_full"],
        bp_converged=table["bp_converged"],
        p=np.array(P_H2),
        seed=np.array(seed),
        rounds=np.array(ROUNDS),
    )

    fig_path = os.path.join(OUT_DIR, "fig_dh_osd0.png")
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt

        mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        xmax = max(8.0, float(np.quantile(dh[fail0], 0.98)) if n_fail else 8.0)
        bins = np.arange(0, xmax + 2) - 0.5
        ax.hist(dh[~fail0], bins=bins, density=True, alpha=0.4, color="#8a8a8a", label="OSD-0 correct")
        ax.hist(dh[fail0], bins=bins, density=True, alpha=0.45, color="#c44e52", label="OSD-0 fail")
        ax.set_xlabel(r"$d_H(\hat e_{\mathrm{BP}},\hat e_{\mathrm{OSD0}})$")
        ax.set_ylabel("density")
        ax.set_title(f"bb_12_2  r=12  Stim p={P_H2}  n={shots}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    except Exception:
        fig_path = None

    row = {
        **css_meta(code),
        "p": P_H2,
        "rounds": ROUNDS,
        "shots": shots,
        "seed": seed,
        "noise": "stim_circuit_level_single_p",
        "note": (
            "Preflight uses the H2-screen scalar p=0.00105. Native H2-Emulator "
            "noise is a separate Nexus Simulation job (0 HQC) attempted by "
            "quantinuum/bb12_compile_cost.py."
        ),
        **gates,
        "n_detectors": int(h.shape[0]),
        "n_error_mechs": int(n_cols),
        "osd_rank": int(rank),
        "n_free": int(n_cols - rank),
        "ler_osd0": float(fail0.mean()),
        "ler_full_1fv": float(failf.mean()),
        "n_fail_osd0": n_fail,
        "n_beneficial": n_ben,
        "auroc_fail": float(auc),
        "auroc_fail_ci95": [float(np.quantile(auroc_vals, 0.025)), float(np.quantile(auroc_vals, 0.975))],
        "d_h_mean_ok": float(dh[~fail0].mean()) if (~fail0).any() else None,
        "d_h_mean_fail": float(dh[fail0].mean()) if n_fail else None,
        "bp_converged": float(table["bp_converged"].mean()),
        "m2d_matches_detector_sampler": m2d_matches_detector_sampler,
        "expected_fail_count_300": float(fail0.mean() * 300),
        "formula_hqc": {str(s): hqc(gates["n1"], gates["n2"], gates["nm"], s) for s in (100, 300, 500, 1000)},
        "pytket_translate_ok": pytket_ok,
        "pytket_n_qubits": nq_tket,
        "pytket_error": pytket_err,
        "label_spec": spec,
        "figure": fig_path,
    }
    return row


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(
        f"bb_12_2 r={ROUNDS} p={P_H2} n={SHOTS} seed={SEED}  (local Stim, 0 HQC)",
        flush=True,
    )
    row = run_preflight(SHOTS, SEED)
    path = os.path.join(OUT_DIR, "preflight_results.json")
    with open(path, "w") as fh:
        json.dump(_jsonable(row), fh, indent=2)
        fh.write("\n")
    print(
        f"OSD0 LER={row['ler_osd0']:.4f} [{row['n_fail_osd0']}]  "
        f"1FV LER={row['ler_full_1fv']:.4f}  "
        f"AUROC={row['auroc_fail']:.3f} "
        f"[{row['auroc_fail_ci95'][0]:.3f},{row['auroc_fail_ci95'][1]:.3f}]  "
        f"m2d_ok={row['m2d_matches_detector_sampler']}  "
        f"E[fails @ 300]={row['expected_fail_count_300']:.1f}",
        flush=True,
    )
    print("formula HQC", row["formula_hqc"])
    print("wrote", path, row.get("figure"))


if __name__ == "__main__":
    main()
