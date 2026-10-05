"""Formal radial_90_8 production shots. Does not touch BB Adaptive V1 or the pilot dir.

Usage:
    python -m failure_prediction.radial_formal_run --p 0.008 --seed 9100 --shots 500 \\
        --split test --out outputs/failure_prediction/nonbb_radial_formal/test/p0.008_seed9100.npz
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from failure_prediction.adaptive_v1 import (
    K_SENSITIVITY,
    pick_one_free,
    score_one_free_prefix,
)
from failure_prediction.osd_internals import gf2_rank_packed, osd0_information_set, pack_matrix
from failure_prediction.qldpc_circuit import (
    build_z_memory_circuit,
    detector_error_model_to_matrices,
)
from failure_prediction.radial_code import radial_90_8
from failure_prediction.radial_headroom import gf2_residual, _logical_fail

ROUNDS = 12


def evaluate_record(decoder, syndrome, actual, packed, log_inv_p, logicals, h, n_cols, rank):
    syndrome = np.asarray(syndrome, dtype=np.uint8)
    actual = np.asarray(actual, dtype=np.uint8).ravel()
    decoder.decode(syndrome)
    conv = bool(decoder.converge)
    e_bp = np.asarray(decoder.bp_decoding, dtype=np.uint8) % 2
    r_bp = int(np.count_nonzero(gf2_residual(h, e_bp, syndrome)))
    out = {
        "d_h": 0.0,
        "r_bp": float(r_bp),
        "syn_weight": float(np.count_nonzero(syndrome)),
        "bp_converged": float(conv),
        "bp_syndrome_ok": float(r_bp == 0),
        "fail_osd0": 0.0,
        "fail_k1000": 0.0,
        "fail_full": 0.0,
        "he_osd0_ok": 1.0,
        "e_bp": e_bp,
        "e_osd0": e_bp.copy(),
    }
    if r_bp == 0:
        fail = float(_logical_fail(logicals, e_bp, actual))
        out["fail_osd0"] = out["fail_k1000"] = out["fail_full"] = fail
        return out
    llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
    info = osd0_information_set(packed, syndrome, llr, n_cols=n_cols, rank=rank)
    if info is None:
        e0 = np.asarray(decoder.osd0_decoding, dtype=np.uint8) % 2
        out["e_osd0"] = e0
        out["d_h"] = float(np.count_nonzero(e_bp ^ e0))
        out["he_osd0_ok"] = float(np.count_nonzero(gf2_residual(h, e0, syndrome)) == 0)
        fail = float(_logical_fail(logicals, e0, actual))
        out["fail_osd0"] = out["fail_k1000"] = out["fail_full"] = fail
        return out
    e0 = np.asarray(info["solution"], dtype=np.uint8) % 2
    out["e_osd0"] = e0
    out["d_h"] = float(np.count_nonzero(e_bp ^ e0))
    out["he_osd0_ok"] = float(np.count_nonzero(gf2_residual(h, e0, syndrome)) == 0)
    fail0 = float(_logical_fail(logicals, e0, actual))
    out["fail_osd0"] = fail0
    delta, free_p, rf = score_one_free_prefix(info, log_inv_p, n_cols, n_cols)
    e1000 = pick_one_free(info, delta, free_p, rf, K_SENSITIVITY)
    e_full = pick_one_free(info, delta, free_p, rf, int(delta.size))
    out["fail_k1000"] = float(_logical_fail(logicals, e1000, actual))
    out["fail_full"] = float(_logical_fail(logicals, e_full, actual))
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--p", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--shots", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test", choices=["calib", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed == 21:
        raise SystemExit("pilot seed 21 is frozen out of calibration and test")
    from ldpc import BpOsdDecoder

    code = radial_90_8()
    circuit = build_z_memory_circuit(code, p=args.p, rounds=ROUNDS)
    dem = circuit.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True
    )
    matrices = detector_error_model_to_matrices(dem)
    h = matrices.h
    n_cols = h.shape[1]
    packed = pack_matrix(h, n_cols=n_cols, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n_cols)
    k_free = n_cols - rank
    decoder = BpOsdDecoder(
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=20,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    log_inv_p = np.log(1.0 / matrices.error_probs)
    logicals = matrices.logicals.astype(np.int64)
    syn, actual = circuit.compile_detector_sampler(seed=args.seed).sample(
        args.shots, separate_observables=True
    )
    syn = syn.astype(np.uint8)
    actual = actual.astype(np.uint8)
    t0 = time.time()
    rows = [
        evaluate_record(
            decoder, syn[i], actual[i], packed, log_inv_p, logicals, h, n_cols, rank
        )
        for i in range(args.shots)
    ]
    scalar = [
        k
        for k in rows[0]
        if k not in ("e_bp", "e_osd0")
    ]
    table = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in scalar}
    table["e_bp"] = np.stack([r["e_bp"] for r in rows]).astype(np.uint8)
    table["e_osd0"] = np.stack([r["e_osd0"] for r in rows]).astype(np.uint8)
    table["syndrome"] = syn
    table["actual_obs"] = actual
    table["p"] = np.full(args.shots, args.p)
    table["seed"] = np.full(args.shots, args.seed)
    table["k_free"] = np.full(args.shots, k_free)
    table["K_eff"] = np.full(args.shots, min(K_SENSITIVITY, k_free))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **table)
    print(
        f"split={args.split} p={args.p:g} seed={args.seed} n={args.shots} "
        f"k_free={k_free} LER osd0={table['fail_osd0'].mean():.5f} "
        f"K1000={table['fail_k1000'].mean():.5f} full={table['fail_full'].mean():.5f} "
        f"{time.time()-t0:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
