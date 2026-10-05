"""Offline analysis of the frozen 300-shot full-CSS H2 pilot.

Does not submit hardware. Reads raw named-register reconstruction and
decodes with the frozen BP@20 + OSD-0 pipeline.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from failure_prediction.adaptive_v1 import evaluate_batch
from failure_prediction.adaptive_v1_analysis import auroc, wilson
from failure_prediction.h2_adaptive_screen import P_H2, SEED, bb_12_2
from failure_prediction.h2_full_css import build_css_memory_circuit
from failure_prediction.osd_internals import gf2_rank_packed, pack_matrix
from failure_prediction.qldpc_circuit import detector_error_model_to_matrices

OUT_DIR = "outputs/failure_prediction/h2_full_css_hardware"
ROUNDS = 12
N_DET_EXPECTED = 120
N_OBS_EXPECTED = 2
N_MEAS_EXPECTED = 132
SHOTS_EXPECTED = 300
BOOT = 800


def _jsonable(obj: Any):
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


def main() -> None:
    from ldpc import BpOsdDecoder

    meta_path = os.path.join(OUT_DIR, "job_meta.json")
    rec_path = os.path.join(OUT_DIR, "stim_record_named.npy")
    with open(meta_path) as fh:
        job = json.load(fh)
    meas = np.load(rec_path).astype(np.uint8)
    n, n_bits = meas.shape
    issues = []
    if n_bits != N_MEAS_EXPECTED:
        issues.append(f"stim record width {n_bits} != {N_MEAS_EXPECTED}")
    if n == 0:
        issues.append("zero shots")

    code = bb_12_2()
    quiet = build_css_memory_circuit(
        code, p=0.0, rounds=ROUNDS, basis="Z", ancilla_layout="separate"
    )
    noisy = build_css_memory_circuit(
        code, p=P_H2, rounds=ROUNDS, basis="Z", ancilla_layout="separate"
    )

    # Noiseless mapping has not changed
    det0, obs0 = quiet.compile_detector_sampler(seed=0).sample(32, separate_observables=True)
    noiseless_ok = bool(np.all(det0 == 0) and np.all(obs0 == 0))
    if not noiseless_ok:
        issues.append("noiseless Stim mapping changed: detectors/obs not all zero")
    if quiet.num_detectors != N_DET_EXPECTED or quiet.num_observables != N_OBS_EXPECTED:
        issues.append(
            f"quiet circuit det/obs {quiet.num_detectors}/{quiet.num_observables} "
            f"!= {N_DET_EXPECTED}/{N_OBS_EXPECTED}"
        )

    try:
        conv = noisy.compile_m2d_converter()
        det, obs = conv.convert(measurements=meas.astype(bool), separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)
        m2d_ok = True
        m2d_err = None
    except Exception as exc:
        m2d_ok = False
        m2d_err = f"{type(exc).__name__}: {exc}"
        det = obs = None
        issues.append(f"m2d failed: {m2d_err}")

    if det is not None:
        if det.shape != (n, N_DET_EXPECTED):
            issues.append(f"detector shape {det.shape} != ({n},{N_DET_EXPECTED})")
        if obs.shape != (n, N_OBS_EXPECTED):
            issues.append(f"observable shape {obs.shape} != ({n},{N_OBS_EXPECTED})")

    first_x = float(meas[:, 0:5].mean()) if n else None
    first_z = float(meas[:, 5:10].mean()) if n else None
    if first_z is not None and first_z > 0.40 and abs(first_z - first_x) < 0.08:
        issues.append(
            f"first-round Z-checks look random (mean={first_z:.3f}, X={first_x:.3f}); "
            "possible named-register mapping failure"
        )

    sanity = {
        "n_shots": n,
        "n_bits": n_bits,
        "m2d_ok": m2d_ok,
        "m2d_error": m2d_err,
        "noiseless_mapping_ok": noiseless_ok,
        "detector_shape": None if det is None else list(det.shape),
        "observable_shape": None if obs is None else list(obs.shape),
        "labels_from_measurements_only": True,
        "unpack": "s0||...||s11||d by name",
        "first_x_check_mean": first_x,
        "first_z_check_mean": first_z,
        "issues": issues,
    }
    if issues:
        out = {
            "status": "STOP_BEFORE_SCIENCE",
            "reason": issues,
            "sanity": sanity,
            "job": job,
        }
        path = os.path.join(OUT_DIR, "hardware_analysis.json")
        with open(path, "w") as fh:
            json.dump(_jsonable(out), fh, indent=2)
            fh.write("\n")
        print("STOP BEFORE SCIENCE:", issues, flush=True)
        print("wrote", path, flush=True)
        return

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
    n_full = int(failf.sum())
    n_ben = int((fail0 & ~failf).sum())
    ler = float(fail0.mean())
    lo, hi = wilson(n_fail, n)
    auc = auroc(fail0.astype(int), dh)
    rng = np.random.default_rng(SEED + 7)
    boot = np.empty(BOOT)
    y = fail0.astype(int)
    for b in range(BOOT):
        ix = rng.integers(0, n, n)
        boot[b] = auroc(y[ix], dh[ix])
    ci = [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]

    np.savez_compressed(
        os.path.join(OUT_DIR, "decoded_shots.npz"),
        measurements=meas,
        detectors=det,
        actual_logical=obs,
        d_h=dh,
        fail_osd0=table["fail_osd0"],
        fail_full=table["fail_full"],
        bp_converged=table["bp_converged"],
    )

    fig_path = os.path.join(OUT_DIR, "fig_dh_osd0_hardware.png")
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
        ax.set_title(f"H2-1 full-CSS Z-mem bb_12_2  n={n}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    except Exception:
        fig_path = None

    # Preflight (Stim p=0.00105, separate Z, n=4000) frozen numbers
    stim_pre = {
        "p": P_H2,
        "shots": 4000,
        "ler_osd0": 0.2655,
        "n_fail_osd0": 1062,
        "auroc_fail": 0.8860856636655347,
        "ler_full_1fv": 0.2545,
        "expected_fail_count_300": 79.65,
        "d_h_mean_ok": None,
        "d_h_mean_fail": None,
        "source": "outputs/failure_prediction/h2_full_css_preflight/full_css_sim.json Z_separate",
    }

    predictive = bool(auc == auc and auc > 0.5 and ci[0] > 0.5)
    informative = bool(n_fail >= 10 and (n - n_fail) >= 10)
    wasm = bool(predictive and informative)

    out = {
        "status": "OK",
        "backend": job.get("backend"),
        "execute_job_id": job.get("execute_job_id"),
        "actual_hqc": job.get("actual_hqc"),
        "shots_requested": SHOTS_EXPECTED,
        "n_valid_shots": n,
        "n_fail_osd0": n_fail,
        "ler_osd0": ler,
        "ler_osd0_wilson95": [lo, hi],
        "n_fail_full_1fv": n_full,
        "ler_full_1fv": float(failf.mean()),
        "n_osd0_rescued_by_full_1fv": n_ben,
        "d_h_mean_ok": float(dh[~fail0].mean()) if (~fail0).any() else None,
        "d_h_median_ok": float(np.median(dh[~fail0])) if (~fail0).any() else None,
        "d_h_mean_fail": float(dh[fail0].mean()) if n_fail else None,
        "d_h_median_fail": float(np.median(dh[fail0])) if n_fail else None,
        "auroc_fail": float(auc) if auc == auc else None,
        "auroc_fail_ci95": ci if auc == auc else None,
        "bp_converged": float(table["bp_converged"].mean()),
        "stim_preflight": stim_pre,
        "dh_remains_predictive_native_h2": predictive,
        "statistically_informative": informative,
        "wasm_later_justified": wasm,
        "wasm_note": (
            "Wasm is justified only as a later real-time d_H routing primitive "
            "if AUROC is predictive. This job is not an adaptive-hardware experiment."
        ),
        "sanity": sanity,
        "figure": fig_path,
        "decoder": "BP@20 + OSD-0 on Stim DEM at p=0.00105; d_H = wt(e_BP XOR e_OSD0)",
        "no_parameter_tuning": True,
    }
    path = os.path.join(OUT_DIR, "hardware_analysis.json")
    with open(path, "w") as fh:
        json.dump(_jsonable(out), fh, indent=2)
        fh.write("\n")
    print(
        f"n={n} OSD0_fail={n_fail} LER={ler:.4f} "
        f"AUROC={auc:.3f} CI={ci} 1FV_rescue={n_ben} "
        f"predictive={predictive} informative={informative}",
        flush=True,
    )
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
