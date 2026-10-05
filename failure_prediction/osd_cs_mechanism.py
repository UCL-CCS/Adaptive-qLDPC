"""Extract OSD-CS search trajectories for the WC / WW / CW populations.

The 20k paired dataset stored labels and features but not syndromes. It does not
need to: the shots come from a deterministic Stim sampler, so replaying each
shard's seed reproduces exactly the same syndromes and observables. Only the
selected shots are reprocessed, and the OSD-CS sweep itself is never rerun --
`osd_cs_landscape` reconstructs the whole candidate landscape from the single
OSD-0 elimination at about 47 ms/shot against the sweep's 1300 ms, and gives the
score and logical action of *every* candidate rather than just the winner.

Recorded per shot, all with respect to the scoring function the package actually
uses, S(x) = sum_{i : x_i = 1} log(1/p_i) over the channel priors:

* which candidate won (OSD-0 itself, a single flip, or the order-2 double flip)
  and its rank in the free-variable ordering;
* Delta S = S(OSD-0) - S(winner);
* whether the OSD-0-to-winner difference changes the logical coset;
* whether a *correct* candidate exists anywhere in the sweep's family, where it
  sits in the ordering, and how its score compares to the winner's. This is what
  separates "the answer is not in the candidate set" from "the answer is in the
  set but the score ranks it below a wrong one";
* the prefix-maximum record sequence, which is enough to reconstruct the winner
  under any truncation of the sweep without rerunning anything.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np

from config import OUTPUT_DIR
from failure_prediction.osd_cs_landscape import (
    WINNER_DOUBLE,
    WINNER_OSD0,
    WINNER_SINGLE,
    candidate_landscape,
    winner_error_vector,
)
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
SEVERITY_CUT = 26.0  # top tercile of hamming_bp_osd0 in the 20k dataset


def _select(outcome: np.ndarray, features: np.ndarray, names: List[str], n_cc: int, rng):
    """All discordant shots, plus a modest high-severity CC control."""

    idx = {
        "WC": np.flatnonzero(outcome == WC),
        "WW": np.flatnonzero(outcome == WW),
        "CW": np.flatnonzero(outcome == CW),
    }
    ham = features[:, names.index("hamming_bp_osd0")]
    cc_pool = np.flatnonzero((outcome == CC) & (ham >= SEVERITY_CUT))
    take = min(n_cc, cc_pool.size)
    idx["CC"] = rng.choice(cc_pool, size=take, replace=False) if take else cc_pool[:0]
    return idx


def run_shard(
    shard_path: str,
    decoder,
    packed,
    matrices,
    log_inv_p,
    rank: int,
    n_cols: int,
    circuit,
    n_cc: int,
    rng,
) -> Dict[str, np.ndarray]:
    d = np.load(shard_path, allow_pickle=True)
    outcome = d["outcome"]
    features = d["features"]
    names = [str(x) for x in d["feature_names"]]
    seed = int(d["seed"])
    shots = int(d["shots"])
    stored_fast_wrong = d["fast_wrong"]
    stored_strong_wrong = d["strong_wrong"]

    syndromes, actual_obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    syndromes = syndromes.astype(np.uint8)
    actual_obs = actual_obs.astype(np.uint8)

    picks = _select(outcome, features, names, n_cc, rng)
    logicals = matrices.logicals.astype(np.int64)

    rows: List[Dict[str, float]] = []
    rec_idx: List[np.ndarray] = []
    rec_score: List[np.ndarray] = []
    rec_correct: List[np.ndarray] = []
    n_label_agree = 0
    n_label_checked = 0

    for cls_name, indices in picks.items():
        for i in indices:
            i = int(i)
            syndrome = syndromes[i]
            decoder.decode(syndrome)
            if bool(decoder.converge):
                continue
            llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
            info = osd0_information_set(packed, syndrome, llr, n_cols=n_cols, rank=rank)
            if info is None:
                continue
            land = candidate_landscape(
                info, log_inv_p, matrices.logicals, n_cols=n_cols, rank=rank
            )
            if land is None:
                continue

            e0 = land["e0"]
            e_cs = winner_error_vector(land, n_cols, info["pivot_cols"])
            delta_e = (e0 ^ e_cs).astype(np.uint8)
            h_delta_zero = int(not np.any((matrices.h @ delta_e) % 2))
            obs = actual_obs[i]
            pred0 = (logicals @ e0.astype(np.int64)) % 2
            osd0_correct = int(not np.any(pred0 != obs))

            # Logical action of every candidate: L e_j = L e_0 XOR L n_j.
            pred_single = (pred0[:, None].astype(np.uint8) ^ land["l_null_single"]) % 2
            correct_single = ~np.any(pred_single != obs[:, None], axis=0)
            pred_double = (pred0.astype(np.uint8) ^ land["l_null_double"]) % 2
            double_correct = int(not np.any(pred_double != obs))

            delta = land["delta_single"]
            wtype, wrank = land["winner_type"], land["winner_rank"]
            if wtype == WINNER_OSD0:
                cs_correct = osd0_correct
                winner_nontrivial = 0
            elif wtype == WINNER_SINGLE:
                cs_correct = int(correct_single[wrank])
                winner_nontrivial = int(np.any(land["l_null_single"][:, wrank]))
            else:
                cs_correct = double_correct
                winner_nontrivial = int(np.any(land["l_null_double"]))

            n_label_checked += 1
            n_label_agree += int(
                (1 - osd0_correct) == int(stored_fast_wrong[i])
                and (1 - cs_correct) == int(stored_strong_wrong[i])
            )

            # Prefix maxima over the singles: enough to replay any truncation.
            running = np.maximum.accumulate(delta)
            is_record = np.empty(delta.size, dtype=bool)
            is_record[0] = True
            is_record[1:] = delta[1:] > running[:-1]
            is_record &= delta > 0.0
            r_idx = np.flatnonzero(is_record)
            rec_idx.append(r_idx.astype(np.int32))
            rec_score.append(delta[r_idx].astype(np.float32))
            rec_correct.append(correct_single[r_idx].astype(np.uint8))

            n_correct = int(correct_single.sum())
            if n_correct:
                first_correct = int(np.argmax(correct_single))
                cand_scores = np.where(correct_single, delta, -np.inf)
                best_correct_rank = int(np.argmax(cand_scores))
                best_correct_score = float(cand_scores[best_correct_rank])
            else:
                first_correct = -1
                best_correct_rank = -1
                best_correct_score = float("nan")

            rows.append(
                {
                    "seed": seed,
                    "shot": i,
                    "outcome": int(outcome[i]),
                    "winner_type": int(wtype),
                    "winner_rank": int(wrank),
                    "winner_score": float(land["winner_score"]),
                    "winner_logical_nontrivial": winner_nontrivial,
                    "osd0_correct": osd0_correct,
                    "cs_correct": cs_correct,
                    "n_candidates": int(land["n_candidates"]),
                    "n_correct_candidates": n_correct,
                    "first_correct_rank": first_correct,
                    "best_correct_rank": best_correct_rank,
                    "best_correct_score": best_correct_score,
                    "delta_max": float(delta.max()),
                    "n_delta_positive": int((delta > 0).sum()),
                    "double_score": float(land["delta_double"]),
                    "double_correct": double_correct,
                    "osd0_score": float(log_inv_p[e0.astype(bool)].sum()),
                    "s_winner": float(
                        log_inv_p[e0.astype(bool)].sum() - land["winner_score"]
                    ),
                    "h_delta_zero": h_delta_zero,
                    "delta_hamming": float(delta_e.sum()),
                    "n_correct_plus_double": n_correct + double_correct,
                    "hamming_bp_osd0": float(
                        np.count_nonzero((llr < 0) ^ e0.astype(bool))
                    ),
                }
            )

    if not rows:
        return {
            "_rec_offsets": np.zeros(1, dtype=np.int64),
            "_rec_idx": np.zeros(0, dtype=np.int32),
            "_rec_score": np.zeros(0, dtype=np.float32),
            "_rec_correct": np.zeros(0, dtype=np.uint8),
            "_label_agree": np.asarray([n_label_agree, n_label_checked], dtype=np.int64),
        }

    keys = list(rows[0].keys())
    table = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys}
    offsets = np.cumsum([0] + [a.size for a in rec_idx]).astype(np.int64)
    table["_rec_offsets"] = offsets
    table["_rec_idx"] = (
        np.concatenate(rec_idx) if rec_idx else np.zeros(0, dtype=np.int32)
    )
    table["_rec_score"] = (
        np.concatenate(rec_score) if rec_score else np.zeros(0, dtype=np.float32)
    )
    table["_rec_correct"] = (
        np.concatenate(rec_correct) if rec_correct else np.zeros(0, dtype=np.uint8)
    )
    table["_label_agree"] = np.asarray([n_label_agree, n_label_checked], dtype=np.int64)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", default="outputs/failure_prediction/arbitration_mechanism/shards"
    )
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-stop", type=int, default=-1)
    parser.add_argument("--n-cc-per-shard", type=int, default=10)
    parser.add_argument("--code", default="bb_144_12_12")
    parser.add_argument("--p", type=float, default=0.008)
    parser.add_argument("--rounds", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import glob

    from ldpc import BpOsdDecoder

    paths = sorted(glob.glob(os.path.join(args.shards, "*.npz")))
    stop = args.shard_stop if args.shard_stop >= 0 else len(paths)
    paths = paths[args.shard_start : stop]
    print(f"{len(paths)} shards", flush=True)

    code = _get_code(args.code)
    circuit = build_z_memory_circuit(code, p=args.p, rounds=args.rounds)
    dem = circuit.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True
    )
    matrices = detector_error_model_to_matrices(dem)
    h = matrices.h
    n_cols = h.shape[1]
    log_inv_p = np.log(1.0 / matrices.error_probs)
    packed = pack_matrix(h, n_cols=n_cols, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n_cols)

    # OSD-0 is all that is needed: it supplies the BP posteriors and lets the
    # reconstruction be cross-checked against the package's own OSD-0 output.
    decoder = BpOsdDecoder(
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=args.max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )

    rng = np.random.default_rng(args.seed)
    tables = []
    t0 = time.time()
    for j, path in enumerate(paths):
        tables.append(
            run_shard(
                path, decoder, packed, matrices, log_inv_p, rank, n_cols, circuit,
                args.n_cc_per_shard, rng,
            )
        )
        print(
            f"  shard {j + 1}/{len(paths)}  {time.time() - t0:.0f}s elapsed", flush=True
        )

    nonempty = [t for t in tables if any(k for k in t if not k.startswith("_"))]
    if not nonempty:
        raise SystemExit("no shots selected")
    merged: Dict[str, np.ndarray] = {}
    scalar_keys = [
        k for k in nonempty[0] if not k.startswith("_")
    ]
    for key in scalar_keys:
        merged[key] = np.concatenate([t[key] for t in nonempty])
    rec_idx, rec_score, rec_correct, offsets = [], [], [], [0]
    for t in nonempty:
        base = offsets[-1]
        rec_idx.append(t["_rec_idx"])
        rec_score.append(t["_rec_score"])
        rec_correct.append(t["_rec_correct"])
        offsets.extend((t["_rec_offsets"][1:] + base).tolist())
    merged["rec_idx"] = np.concatenate(rec_idx) if rec_idx else np.zeros(0, dtype=np.int32)
    merged["rec_score"] = (
        np.concatenate(rec_score) if rec_score else np.zeros(0, dtype=np.float32)
    )
    merged["rec_correct"] = (
        np.concatenate(rec_correct) if rec_correct else np.zeros(0, dtype=np.uint8)
    )
    merged["rec_offsets"] = np.asarray(offsets, dtype=np.int64)
    agree = np.sum([t["_label_agree"] for t in tables], axis=0)
    merged["label_agreement"] = agree
    merged["rank"] = np.asarray([rank])
    merged["n_cols"] = np.asarray([n_cols])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **merged)
    print(
        f"replayed labels agree with the stored dataset on {int(agree[0])}/{int(agree[1])}"
        " shots",
        flush=True,
    )
    print(f"wrote {args.out}  ({merged['outcome'].size} shots)", flush=True)


if __name__ == "__main__":
    main()
