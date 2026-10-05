"""Coset-aware aggregation of the existing OSD-CS surrogate, no new search.

OSD-CS picks

    e* = argmin_{e in C} S(e),    S(e) = sum_{i: e_i=1} log(1/p_i)

i.e. the single *physical* candidate with the best surrogate. The logical
objective is the total mass of a *coset*. This module does not enlarge the
candidate family and does not claim that S is a negative log probability. It
only re-groups the same CS candidates (OSD-0 + k one-free-variable
perturbations + the order-2 double) by Stim-observable action and compares

    min-S selection          vs         W(L) = sum_{e in C_L} exp[-S(e)]

which is equivalently S_L^eff = -log sum exp[-S]. Call this *coset-aware
aggregation of the OSD surrogate*, not Bayesian decoding.

The 20k Monte Carlo is not rerun. Shots are the exact (seed, shot) pairs already
in `trajectories.npz`; Stim is replayed only to recover syndromes and
observables.

A "single perturbation" here is one free variable of the OSD information set.
Pivot bits are then re-solved from He = s, so the full correction can flip many
physical mechanisms. It is not a single-qubit error.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from failure_prediction.osd_cs_landscape import candidate_landscape
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

CC, WC, CW, WW = 0, 1, 2, 3


def pack_obs(obs: np.ndarray) -> np.ndarray:
    """Pack a 12-bit (or n_obs-bit) observable vector into uint16/uint32 keys."""

    bits = np.asarray(obs, dtype=np.uint32)
    n_obs = bits.shape[0]
    weights = (1 << np.arange(n_obs, dtype=np.uint32))
    if bits.ndim == 1:
        return np.array([int(bits @ weights)], dtype=np.uint32)
    return (bits.T @ weights).astype(np.uint32)


def _logsumexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.max(x))
    return m + float(np.log(np.sum(np.exp(x - m))))


def aggregate_cosets(
    land: Dict[str, np.ndarray],
    pred0: np.ndarray,
    actual: np.ndarray,
) -> Dict[str, float]:
    """min-S vs coset aggregate on one shot's CS family, including OSD-0."""

    pred0 = np.asarray(pred0, dtype=np.uint8).ravel()
    actual = np.asarray(actual, dtype=np.uint8).ravel()
    pred_single = (pred0[:, None] ^ land["l_null_single"].astype(np.uint8)) % 2
    pred_double = (pred0 ^ land["l_null_double"].astype(np.uint8)) % 2

    keys = np.concatenate(
        [
            pack_obs(pred0),
            pack_obs(pred_single),
            pack_obs(pred_double),
        ]
    )
    deltas = np.concatenate(
        [
            np.array([0.0], dtype=np.float64),
            np.asarray(land["delta_single"], dtype=np.float64),
            np.array([float(land["delta_double"])], dtype=np.float64),
        ]
    )
    true_key = int(pack_obs(actual)[0])

    # min-S: maximum ΔS, OSD-0 on a non-positive maximum (strict improvement).
    imax = int(np.argmax(deltas))
    if deltas[imax] <= 0.0:
        imax = 0
    mins_key = int(keys[imax])

    uniq, inv = np.unique(keys, return_inverse=True)
    n_cosets = int(uniq.size)
    logW = np.empty(n_cosets, dtype=np.float64)
    counts = np.empty(n_cosets, dtype=np.int32)
    best_d = np.empty(n_cosets, dtype=np.float64)
    for t in range(n_cosets):
        d = deltas[inv == t]
        logW[t] = _logsumexp(d)
        counts[t] = int(d.size)
        best_d[t] = float(d.max())
    i_agg = int(np.argmax(logW))
    agg_key = int(uniq[i_agg])

    true_present = bool(np.any(uniq == true_key))
    if true_present:
        i_true = int(np.flatnonzero(uniq == true_key)[0])
        logW_true = float(logW[i_true])
        n_true = int(counts[i_true])
        best_true = float(best_d[i_true])
    else:
        logW_true = float("-inf")
        n_true = 0
        best_true = float("nan")

    i_mins_c = int(np.flatnonzero(uniq == mins_key)[0])
    return {
        "mins_correct": float(mins_key == true_key),
        "agg_correct": float(agg_key == true_key),
        "true_present": float(true_present),
        "n_cosets": float(n_cosets),
        "n_true": float(n_true),
        "logW_true": logW_true,
        "logW_mins": float(logW[i_mins_c]),
        "logW_agg": float(logW[i_agg]),
        "delta_mins": float(deltas[imax]),
        "delta_best_true": best_true,
        "delta_gap": (
            float(deltas[imax] - best_true) if true_present else float("nan")
        ),
        "n_eff_true": (
            float(np.exp(logW_true - best_true)) if true_present and np.isfinite(best_true) else 0.0
        ),
        "agg_equals_mins": float(agg_key == mins_key),
        "osd0_correct": float(int(pack_obs(pred0)[0]) == true_key),
    }


def run_from_trajectories(
    traj_path: str,
    shard_dir: str,
    decoder,
    packed,
    matrices,
    log_inv_p,
    rank: int,
    n_cols: int,
    circuit,
    seed_min: Optional[int] = None,
    seed_max: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Replay only the (seed, shot) pairs already stored in trajectories.npz."""

    traj = np.load(traj_path)
    seeds = traj["seed"].astype(int)
    shots = traj["shot"].astype(int)
    outcomes = traj["outcome"].astype(int)
    logicals = matrices.logicals.astype(np.int64)

    by_seed: Dict[int, List[int]] = {}
    for i, (sd, sh) in enumerate(zip(seeds, shots)):
        if seed_min is not None and sd < seed_min:
            continue
        if seed_max is not None and sd >= seed_max:
            continue
        by_seed.setdefault(int(sd), []).append(i)

    rows: List[Dict[str, float]] = []
    t0 = time.time()
    for n_done, (sd, idxs) in enumerate(sorted(by_seed.items()), 1):
        shard = os.path.join(
            shard_dir, f"bb_144_12_12_p0.008_r24_n500_seed{sd}.npz"
        )
        if not os.path.isfile(shard):
            raise FileNotFoundError(shard)
        meta = np.load(shard)
        n_sample = int(meta["shots"])
        syndromes, actual_obs = circuit.compile_detector_sampler(seed=int(sd)).sample(
            n_sample, separate_observables=True
        )
        syndromes = syndromes.astype(np.uint8)
        actual_obs = actual_obs.astype(np.uint8)

        for i in idxs:
            sh = int(shots[i])
            decoder.decode(syndromes[sh])
            if bool(decoder.converge):
                continue
            llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
            info = osd0_information_set(
                packed, syndromes[sh], llr, n_cols=n_cols, rank=rank
            )
            if info is None:
                continue
            land = candidate_landscape(
                info, log_inv_p, matrices.logicals, n_cols=n_cols, rank=rank
            )
            if land is None:
                continue
            e0 = land["e0"]
            pred0 = (logicals @ e0.astype(np.int64)) % 2
            agg = aggregate_cosets(land, pred0, actual_obs[sh])
            agg["seed"] = float(sd)
            agg["shot"] = float(sh)
            agg["outcome"] = float(outcomes[i])
            rows.append(agg)

        print(
            f"  seed {sd}  {n_done}/{len(by_seed)}  "
            f"{time.time() - t0:.0f}s  n={len(rows)}",
            flush=True,
        )

    if not rows:
        raise SystemExit("no shots aggregated")
    keys = list(rows[0].keys())
    return {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trajectories",
        default="outputs/failure_prediction/osd_cs_mechanism/trajectories.npz",
    )
    p.add_argument(
        "--shards",
        default="outputs/failure_prediction/arbitration_mechanism/shards",
    )
    p.add_argument("--seed-min", type=int, default=None)
    p.add_argument("--seed-max", type=int, default=None)
    p.add_argument("--code", default="bb_144_12_12")
    p.add_argument("--p", type=float, default=0.008)
    p.add_argument("--rounds", type=int, default=24)
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
    table = run_from_trajectories(
        args.trajectories,
        args.shards,
        decoder,
        packed,
        matrices,
        log_inv_p,
        rank,
        n_cols,
        circuit,
        seed_min=args.seed_min,
        seed_max=args.seed_max,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **table)
    print(f"wrote {args.out}  n={table['outcome'].size}", flush=True)


if __name__ == "__main__":
    main()
