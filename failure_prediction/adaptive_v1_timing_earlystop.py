"""True early-stop serial timing for truncated 1FV at K=500 and K=1000.

Unlike adaptive_v1_timing / evaluate_shot, this scores only the first K free
columns (score_one_free_prefix(..., k_max=K)). It does not prorate a full sweep.
Does not overwrite serial_timing_n1000.json.

Usage:
    python -m failure_prediction.adaptive_v1_timing_earlystop \
        --p 0.007 --shots 1000 --seed 9001 \
        --out outputs/failure_prediction/adaptive_v1/serial_timing_earlystop_n1000.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from failure_prediction.adaptive_v1 import (
    K_PRIMARY,
    K_SENSITIVITY,
    pick_one_free,
    score_one_free_prefix,
)
from failure_prediction.adaptive_v1_timing import boot_mean_ci
from failure_prediction.osd_internals import (
    gf2_rank_packed,
    osd0_information_set,
    pack_matrix,
)
from failure_prediction.qldpc_circuit import (
    _get_code,
    build_z_memory_circuit,
    detector_error_model_to_matrices,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default="bb_144_12_12")
    p.add_argument("--p", type=float, default=0.007)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--shots", type=int, default=1000)
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--out", required=True)
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
    decoder = BpOsdDecoder(
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=args.max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    syndromes, _ = circuit.compile_detector_sampler(seed=args.seed).sample(
        args.shots, separate_observables=True
    )
    syndromes = syndromes.astype(np.uint8)

    t_osd0: List[float] = []
    t_ge: List[float] = []
    t_k500: List[float] = []
    t_k1000: List[float] = []
    t_score500: List[float] = []
    t_score1000: List[float] = []
    d_h: List[float] = []
    n_free_used = []

    print(
        f"[earlystop] host serial  p={args.p:g} n={args.shots} rank={rank} "
        f"k_free={n_cols - rank}  true k_max in {{{K_PRIMARY},{K_SENSITIVITY}}}",
        flush=True,
    )
    for i in range(args.shots):
        s = syndromes[i]
        t0 = time.perf_counter()
        decoder.decode(s)
        t_fast = time.perf_counter() - t0
        conv = bool(decoder.converge)
        e_bp = np.asarray(decoder.bp_decoding, dtype=np.uint8) % 2
        e0 = np.asarray(decoder.osd0_decoding, dtype=np.uint8) % 2
        d_h.append(float(np.count_nonzero(e_bp ^ e0)))
        t_osd0.append(1e3 * t_fast)
        extra_ge = extra500 = extra1000 = sc500 = sc1000 = 0.0
        if not conv:
            llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
            t1 = time.perf_counter()
            info = osd0_information_set(packed, s, llr, n_cols=n_cols, rank=rank)
            extra_ge = time.perf_counter() - t1
            if info is not None:
                t2 = time.perf_counter()
                d500, f500, r500 = score_one_free_prefix(
                    info, log_inv_p, n_cols, K_PRIMARY
                )
                pick_one_free(info, d500, f500, r500, K_PRIMARY)
                sc500 = time.perf_counter() - t2
                extra500 = extra_ge + sc500

                t3 = time.perf_counter()
                d1000, f1000, r1000 = score_one_free_prefix(
                    info, log_inv_p, n_cols, K_SENSITIVITY
                )
                pick_one_free(info, d1000, f1000, r1000, K_SENSITIVITY)
                sc1000 = time.perf_counter() - t3
                extra1000 = extra_ge + sc1000
                n_free_used.append(int(d1000.size))
        t_ge.append(1e3 * extra_ge)
        t_score500.append(1e3 * sc500)
        t_score1000.append(1e3 * sc1000)
        t_k500.append(1e3 * (t_fast + extra500))
        t_k1000.append(1e3 * (t_fast + extra1000))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.shots}", flush=True)

    rng = np.random.default_rng(args.seed + 17)
    n_used = args.shots
    t0a = np.asarray(t_osd0)
    tk = np.asarray(t_k1000)
    dh = np.asarray(d_h)
    out: Dict[str, dict] = {
        "meta": {
            "code": args.code,
            "p": args.p,
            "shots": args.shots,
            "seed": args.seed,
            "early_stop": True,
            "note": (
                "true truncated scoring: score_one_free_prefix(k_max=K) plus pick. "
                "GE shared in the always-K totals. Not a prorated full 1FV sweep. "
                "Does not replace serial_timing_n1000.json."
            ),
            "n_free_scored_k1000_mean": float(np.mean(n_free_used)) if n_free_used else None,
        },
        "bp20_osd0": boot_mean_ci(t_osd0, rng),
        "ge_only_ms": boot_mean_ci(t_ge, rng),
        "score_k500_only_ms": boot_mean_ci(t_score500, rng),
        "score_k1000_only_ms": boot_mean_ci(t_score1000, rng),
        "always_trunc_k500_earlystop": boot_mean_ci(t_k500, rng),
        "always_trunc_k1000_earlystop": boot_mean_ci(t_k1000, rng),
    }
    for f, key in (
        (0.10, "adapt_k1000_f10_earlystop"),
        (0.20, "adapt_k1000_f20_earlystop"),
        (0.30, "adapt_k1000_f30_earlystop"),
    ):
        k = max(int(round(f * n_used)), 0)
        order = np.argsort(-dh, kind="mergesort")
        esc = np.zeros(n_used, dtype=bool)
        esc[order[:k]] = True
        t_ad = t0a + esc * (tk - t0a)
        out[key] = boot_mean_ci(t_ad.tolist(), rng)
        out[key]["f_esc"] = float(esc.mean())
        if k:
            out[key]["E_delta_ms_given_escalated"] = float((tk[esc] - t0a[esc]).mean())
        out[key]["E_delta_ms_unconditional"] = float((tk - t0a).mean())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
