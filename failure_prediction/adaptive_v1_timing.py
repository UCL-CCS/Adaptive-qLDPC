"""Serial wall-clock of OSD-0, truncated CS, full 1FV CS, and optional package OSD-CS.

One process, one node, no worker contention. Candidate counts are not used as
a speed-up claim.

Usage:
    python -m failure_prediction.adaptive_v1_timing --p 0.007 --shots 1000 \\
        --skip-package --out outputs/failure_prediction/adaptive_v1/serial_timing_n1000.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np

from failure_prediction.adaptive_v1 import evaluate_shot
from failure_prediction.osd_internals import gf2_rank_packed, pack_matrix
from failure_prediction.qldpc_circuit import (
    _get_code,
    build_z_memory_circuit,
    detector_error_model_to_matrices,
)


def _stats(xs: List[float]) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()) if a.size else float("nan"),
        "median_ms": float(np.median(a)) if a.size else float("nan"),
        "p10_ms": float(np.quantile(a, 0.10)) if a.size else float("nan"),
        "p90_ms": float(np.quantile(a, 0.90)) if a.size else float("nan"),
    }


def boot_mean_ci(xs, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    a = np.asarray(xs, dtype=np.float64)
    st = _stats(a.tolist())
    if a.size == 0:
        st["mean_ci95"] = [float("nan"), float("nan")]
        return st
    n = a.size
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = a[rng.integers(0, n, n)].mean()
    st["mean_ci95"] = [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]
    return st


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--code", default="bb_144_12_12")
    p.add_argument("--p", type=float, default=0.007)
    p.add_argument("--rounds", type=int, default=24)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument("--seed", type=int, default=9001)
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument(
        "--skip-package",
        action="store_true",
        help="Skip package OSD-CS timing (not the primary baseline)",
    )
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
    logicals = matrices.logicals.astype(np.int64)
    common = dict(
        error_channel=matrices.error_probs.tolist(),
        max_iter=args.max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
    )
    dec0 = BpOsdDecoder(h, osd_method="osd_0", **common)
    dec_cs = None
    if not args.skip_package:
        dec_cs = BpOsdDecoder(h, osd_method="osd_cs", osd_order=2, **common)
    syndromes, actual = circuit.compile_detector_sampler(seed=args.seed).sample(
        args.shots, separate_observables=True
    )
    syndromes = syndromes.astype(np.uint8)
    actual = actual.astype(np.uint8)

    t_osd0, t_k500, t_k1000, t_full, t_pkg = [], [], [], [], []
    d_h: List[float] = []
    fail0 = fail_full = fail_pkg = 0
    agree = 0
    print(
        f"[timing] host serial  p={args.p:g} n={args.shots} rank={rank} "
        f"k={n_cols - rank} skip_package={args.skip_package}",
        flush=True,
    )
    for i in range(args.shots):
        row = evaluate_shot(
            dec0,
            syndromes[i],
            actual[i],
            packed,
            log_inv_p,
            logicals,
            n_cols,
            rank,
        )
        t_osd0.append(row["t_osd0_ms"])
        t_k500.append(row["t_osd0_ms"] + row["t_k500_ms"])
        t_k1000.append(row["t_osd0_ms"] + row["t_k1000_ms"])
        t_full.append(row["t_osd0_ms"] + row["t_full_ms"])
        d_h.append(row["d_h"])
        fail0 += int(row["fail_osd0"])
        fail_full += int(row["fail_full"])
        if dec_cs is not None:
            t0 = time.perf_counter()
            e_pkg = np.asarray(dec_cs.decode(syndromes[i]), dtype=np.uint8) % 2
            t_pkg.append(1e3 * (time.perf_counter() - t0))
            pred = (logicals @ e_pkg.astype(np.int64)) % 2
            fp = int(np.any(pred != actual[i]))
            fail_pkg += fp
            agree += int(fp == int(row["fail_full"]))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.shots}", flush=True)

    rng = np.random.default_rng(args.seed + 17)
    n_used = args.shots
    out: Dict[str, dict] = {
        "meta": {
            "code": args.code,
            "p": args.p,
            "shots": args.shots,
            "seed": args.seed,
            "skip_package": bool(args.skip_package),
            "note": "serial, one process; package OSD-CS is not the primary baseline",
        },
        "bp20_osd0": boot_mean_ci(t_osd0, rng),
        "always_trunc_k500": boot_mean_ci(t_k500, rng),
        "always_trunc_k1000": boot_mean_ci(t_k1000, rng),
        "always_full_1fv": boot_mean_ci(t_full, rng),
        "subsample_ler": {
            "n": n_used,
            "osd0": fail0 / n_used,
            "full_1fv": fail_full / n_used,
        },
    }
    if t_pkg:
        out["package_osd_cs_order2"] = boot_mean_ci(t_pkg, rng)
        out["subsample_ler"]["package_cs"] = fail_pkg / n_used
        out["subsample_ler"]["full_1fv_vs_package_logical_agree"] = agree / n_used

    dh = np.asarray(d_h)
    t0a = np.asarray(t_osd0)
    tk = np.asarray(t_k1000)
    for f, key in (
        (0.10, "adapt_k1000_f10"),
        (0.20, "adapt_k1000_f20"),
        (0.30, "adapt_k1000_f30"),
    ):
        k = max(int(round(f * n_used)), 0)
        order = np.argsort(-dh, kind="mergesort")
        esc = np.zeros(n_used, dtype=bool)
        esc[order[:k]] = True
        t_ad = t0a + esc * (tk - t0a)
        out[key] = boot_mean_ci(t_ad, rng)
        out[key]["f_esc"] = float(esc.mean())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
