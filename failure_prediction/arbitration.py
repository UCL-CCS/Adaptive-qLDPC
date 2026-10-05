"""Paired fast/strong decoder outcomes with fast-path-only structural features.

The question here is not "is this syndrome hard" but "would switching decoder
help this instance", so every shot is placed in one of four classes:

    CC  both correct
    WC  fast wrong, strong correct   -> beneficial escalation
    CW  fast correct, strong wrong   -> harmful escalation
    WW  both wrong

Fast is BP@20 + OSD-0, strong is BP@20 + OSD-CS-2. Both are read from a *single*
`BpOsdDecoder(osd_method="osd_cs", osd_order=2)` call, which exposes
`osd0_decoding` and `osdw_decoding` from the same decode. That is verified to
agree bit-for-bit with a standalone OSD-0 decoder by
`failure_prediction.verify_osd_reconstruction`, and it makes the pairing exact by
construction instead of relying on matched sampler seeds.

Every recorded feature is computable *before* the OSD-CS sweep runs, so anything
found here could in principle drive a real-time router. Features are tied to the
actual upstream OSD mechanism (see `osd_internals` for the exact correspondence)
rather than being generic engineered statistics:

* OSD-CS scores candidates by `sum_{i: x_i=1} log(1/p_i)` over the *channel*
  priors, so `osd0_weight_prior` is literally the baseline weight the sweep has
  to beat;
* the information set is the first `rank(H)` independent columns in ascending
  BP-LPR order, so the boundary quantities describe where that greedy scan
  closed and how contested that point was;
* the sweep flips free columns in ascending-LPR order and its single
  double-flip pattern hits free indices 0 and 1, so `llr_free_0` / `llr_free_1`
  are exactly the columns OSD-CS attacks first.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np

from config import OUTPUT_DIR
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

CLASS_CC, CLASS_WC, CLASS_CW, CLASS_WW = 0, 1, 2, 3
CLASS_NAMES = ["CC", "WC", "CW", "WW"]

LOWCONF_TAUS = (0.5, 1.0, 2.0)
BOUNDARY_DELTAS = (0.5, 1.0, 2.0)


def feature_names() -> List[str]:
    names = [
        # OSD-0 candidate itself
        "osd0_weight_prior",
        "osd0_hamming",
        "osd0_mean_logp",
        "hamming_bp_osd0",
        "n_bp_negative",
        # information-set boundary
        "boundary_llr",
        "boundary_pos",
        "n_dep_rejected",
        "frac_dep_rejected",
        "gap_after_boundary",
        "dep_rejected_min_llr",
        "llr_free_0",
        "llr_free_1",
        "llr_free_2",
        "free01_span",
        # BP LPR distribution
        "llr_min",
        "llr_p01",
        "llr_p05",
        "llr_median",
        "llr_mean",
        "llr_std",
        "frac_llr_negative",
        "llr_entropy",
        # context
        "syndrome_weight",
        "bp_converged",
        "bp_iter",
    ]
    names += [f"n_lowconf_{tau:g}".replace(".", "p") for tau in LOWCONF_TAUS]
    names += [f"boundary_density_{d:g}".replace(".", "p") for d in BOUNDARY_DELTAS]
    return names


def _extract_features(
    llr: np.ndarray,
    syndrome: np.ndarray,
    osd0: np.ndarray,
    log_inv_p: np.ndarray,
    info,
    converged: bool,
    bp_iter: int,
    rank: int,
) -> Dict[str, float]:
    support = osd0.astype(bool)
    hamming = int(support.sum())
    weight_prior = float(log_inv_p[support].sum())
    bp_hard = llr < 0.0

    abs_llr = np.abs(llr)
    posterior = 1.0 / (1.0 + np.exp(np.clip(abs_llr, -60.0, 60.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(
            posterior * np.log2(np.clip(posterior, 1e-12, 1.0))
            + (1 - posterior) * np.log2(np.clip(1 - posterior, 1e-12, 1.0))
        )

    feats: Dict[str, float] = {
        "osd0_weight_prior": weight_prior,
        "osd0_hamming": float(hamming),
        "osd0_mean_logp": weight_prior / max(hamming, 1),
        "hamming_bp_osd0": float(np.count_nonzero(bp_hard ^ support)),
        "n_bp_negative": float(np.count_nonzero(bp_hard)),
        "llr_min": float(llr.min()),
        "llr_p01": float(np.quantile(llr, 0.01)),
        "llr_p05": float(np.quantile(llr, 0.05)),
        "llr_median": float(np.median(llr)),
        "llr_mean": float(llr.mean()),
        "llr_std": float(llr.std()),
        "frac_llr_negative": float(np.count_nonzero(bp_hard) / llr.size),
        "llr_entropy": float(np.mean(ent)),
        "syndrome_weight": float(np.count_nonzero(syndrome)),
        "bp_converged": float(bool(converged)),
        "bp_iter": float(bp_iter),
    }
    for tau in LOWCONF_TAUS:
        feats[f"n_lowconf_{tau:g}".replace(".", "p")] = float(np.count_nonzero(abs_llr < tau))

    boundary_keys = [
        "boundary_llr",
        "boundary_pos",
        "n_dep_rejected",
        "frac_dep_rejected",
        "gap_after_boundary",
        "dep_rejected_min_llr",
        "llr_free_0",
        "llr_free_1",
        "llr_free_2",
        "free01_span",
    ] + [f"boundary_density_{d:g}".replace(".", "p") for d in BOUNDARY_DELTAS]
    if info is None:
        # BP converged, so upstream never entered OSD and no information set exists.
        for key in boundary_keys:
            feats[key] = float("nan")
        return feats

    order = info["order"]
    closed_at = int(info["closed_at"])
    boundary_pos = closed_at + 1
    boundary_llr = float(llr[order[closed_at]])
    n_dep_rejected = boundary_pos - rank

    is_pivot = np.zeros(llr.size, dtype=bool)
    is_pivot[info["pivot_cols"]] = True
    free_in_order = order[~is_pivot[order]]

    feats["boundary_llr"] = boundary_llr
    feats["boundary_pos"] = float(boundary_pos)
    feats["n_dep_rejected"] = float(n_dep_rejected)
    feats["frac_dep_rejected"] = float(n_dep_rejected / boundary_pos)
    feats["gap_after_boundary"] = (
        float(llr[order[closed_at + 1]] - boundary_llr)
        if closed_at + 1 < order.size
        else float("nan")
    )
    rejected = order[:boundary_pos][~is_pivot[order[:boundary_pos]]]
    feats["dep_rejected_min_llr"] = (
        float(llr[rejected].min()) if rejected.size else float("nan")
    )
    for idx in range(3):
        feats[f"llr_free_{idx}"] = (
            float(llr[free_in_order[idx]]) if free_in_order.size > idx else float("nan")
        )
    feats["free01_span"] = (
        feats["llr_free_1"] - feats["llr_free_0"] if free_in_order.size > 1 else float("nan")
    )
    for delta in BOUNDARY_DELTAS:
        key = f"boundary_density_{delta:g}".replace(".", "p")
        feats[key] = float(np.count_nonzero(np.abs(llr - boundary_llr) <= delta))
    return feats


def generate(
    code_name: str,
    p: float,
    rounds: int,
    shots: int,
    seed: int,
    max_iter: int,
    output: str,
    progress_every: int = 100,
) -> str:
    from ldpc import BpOsdDecoder

    code = _get_code(code_name)
    circuit = build_z_memory_circuit(code, p=p, rounds=rounds)
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matrices = detector_error_model_to_matrices(dem)
    h = matrices.h
    m, n = h.shape
    log_inv_p = np.log(1.0 / matrices.error_probs)
    # uint8 matmul over ~12k terms wraps mod 256, which happens to preserve the
    # parity we want, but widening keeps that from being load-bearing.
    logicals = matrices.logicals.astype(np.int64)

    packed = pack_matrix(h, n_cols=n, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n)
    print(f"H: {m} x {n}, rank={rank}, free k={n - rank}", flush=True)

    decoder = BpOsdDecoder(
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_cs",
        osd_order=2,
    )

    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, actual_obs = sampler.sample(shots, separate_observables=True)
    syndromes = syndromes.astype(np.uint8)
    actual_obs = actual_obs.astype(np.uint8)

    names = feature_names()
    features = np.zeros((shots, len(names)), dtype=np.float32)
    outcome = np.zeros(shots, dtype=np.uint8)
    fast_wrong = np.zeros(shots, dtype=np.uint8)
    strong_wrong = np.zeros(shots, dtype=np.uint8)

    t_start = time.time()
    for i, syndrome in enumerate(syndromes):
        decoder.decode(syndrome)
        converged = bool(decoder.converge)
        bp_iter = int(decoder.iter)
        llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
        e_fast = np.asarray(decoder.osd0_decoding, dtype=np.uint8) % 2
        e_strong = np.asarray(decoder.osdw_decoding, dtype=np.uint8) % 2

        pred_fast = (logicals @ e_fast.astype(np.int64)) % 2
        pred_strong = (logicals @ e_strong.astype(np.int64)) % 2
        fw = int(np.any(pred_fast != actual_obs[i]))
        sw = int(np.any(pred_strong != actual_obs[i]))
        fast_wrong[i] = fw
        strong_wrong[i] = sw
        outcome[i] = (CLASS_WW if sw else CLASS_WC) if fw else (CLASS_CW if sw else CLASS_CC)

        info = None
        if not converged:
            info = osd0_information_set(packed, syndrome, llr, n_cols=n, rank=rank)
        feats = _extract_features(
            llr, syndrome, e_fast, log_inv_p, info, converged, bp_iter, rank
        )
        features[i] = [feats[name] for name in names]

        if progress_every and (i + 1) % progress_every == 0:
            rate = (time.time() - t_start) / (i + 1)
            counts = np.bincount(outcome[: i + 1], minlength=4)
            print(
                f"  {i + 1}/{shots}  {rate:.2f}s/shot  "
                f"CC={counts[0]} WC={counts[1]} CW={counts[2]} WW={counts[3]}",
                flush=True,
            )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        feature_names=np.asarray(names),
        outcome=outcome,
        outcome_names=np.asarray(CLASS_NAMES),
        fast_wrong=fast_wrong,
        strong_wrong=strong_wrong,
        syndrome_weight=np.count_nonzero(syndromes, axis=1).astype(np.int32),
        code=code.name,
        p=float(p),
        rounds=int(rounds),
        seed=int(seed),
        shots=int(shots),
        max_iter=int(max_iter),
        rank=int(rank),
        n_cols=int(n),
        fast_decoder="bp%d_osd0" % max_iter,
        strong_decoder="bp%d_osdcs2" % max_iter,
    )
    counts = np.bincount(outcome, minlength=4)
    print(
        f"Done in {time.time() - t_start:.0f}s  "
        + "  ".join(f"{CLASS_NAMES[c]}={counts[c]}" for c in range(4)),
        flush=True,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="bb_144_12_12")
    parser.add_argument("--p", type=float, default=0.008)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir or os.path.join(
        OUTPUT_DIR, "failure_prediction", "arbitration_mechanism", "shards"
    )
    out = os.path.join(
        out_dir,
        f"{args.code}_p{args.p:g}_r{args.rounds}_n{args.shots}_seed{args.seed}.npz",
    )
    print(generate(
        code_name=args.code,
        p=args.p,
        rounds=args.rounds,
        shots=args.shots,
        seed=args.seed,
        max_iter=args.max_iter,
        output=out,
        progress_every=args.progress_every,
    ))
