"""Generate matched Adaptive V1 records: OSD-0, truncated CS, full one-free CS.

One (code, p, seed, shots) block. All decoder variants see the same Stim shots.

Usage:
    python -m failure_prediction.adaptive_v1_run --p 0.008 --seed 2000 --shots 500 \\
        --out outputs/failure_prediction/adaptive_v1/parts/p0.008_seed2000.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from failure_prediction.adaptive_v1 import evaluate_batch
from failure_prediction.osd_internals import gf2_rank_packed, pack_matrix
from failure_prediction.qldpc_circuit import (
    _get_code,
    build_z_memory_circuit,
    detector_error_model_to_matrices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default="bb_144_12_12")
    p.add_argument("--p", type=float, required=True)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--shots", type=int, default=500)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ldpc import BpOsdDecoder

    code = _get_code(args.code)
    circuit = build_z_memory_circuit(code, p=args.p, rounds=args.rounds)
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
        max_iter=args.max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    syndromes, actual = circuit.compile_detector_sampler(seed=args.seed).sample(
        args.shots, separate_observables=True
    )
    table = evaluate_batch(
        decoder,
        syndromes.astype(np.uint8),
        actual.astype(np.uint8),
        packed,
        log_inv_p,
        logicals,
        n_cols,
        rank,
        progress_every=args.progress_every,
    )
    table["p"] = np.full(args.shots, args.p)
    table["seed"] = np.full(args.shots, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **table)
    n = args.shots
    print(
        f"p={args.p:g} seed={args.seed} n={n}  "
        f"LER osd0={table['fail_osd0'].mean():.5f}  "
        f"K500={table['fail_k500'].mean():.5f}  "
        f"K1000={table['fail_k1000'].mean():.5f}  "
        f"full={table['fail_full'].mean():.5f}",
        flush=True,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
