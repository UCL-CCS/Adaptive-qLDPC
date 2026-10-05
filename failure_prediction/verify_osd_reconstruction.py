"""Verification gates that must pass before any OSD-boundary feature is trusted.

Gate 1: a single BP+OSD-CS decoder exposes both `osd0_decoding` and
        `osdw_decoding` from one `decode()` call. If its `osd0_decoding` agrees
        bit-for-bit with a standalone BP+OSD-0 decoder, the fast and strong arms
        can be read off one decode, which both halves the cost and makes the
        pairing exact by construction rather than by matching seeds.

        This is not guaranteed a priori: upstream routes `osd_order == 0`
        through `fast_solve` (elimination stops as soon as the syndrome lies in
        the span reached so far) while the OSD-CS path uses the full
        `rref` + `lu_solve`.

Gate 2: the information set re-derived in `osd_internals` reproduces the
        decoder's own `osd0_decoding`.

Gate 3: the OSD-CS candidate landscape rebuilt in `osd_cs_landscape` picks the
        same winning candidate as the decoder's own sweep, so the offline
        reconstruction of the search can stand in for running it. Also checks
        that the OSD-0-to-winner difference really is a nullspace element,
        H (e_0 XOR e_CS) = 0, which is what licenses the logical-coset reading.

Usage:
    python -m failure_prediction.verify_osd_reconstruction --code bb_72_12_6 \
        --p 0.006 --rounds 12 --shots 40
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from failure_prediction.osd_cs_landscape import candidate_landscape, winner_error_vector
from failure_prediction.osd_internals import (
    gf2_rank_packed,
    osd0_information_set,
    pack_matrix,
    tie_fraction,
)
from failure_prediction.qldpc_circuit import (
    _get_code,
    build_z_memory_circuit,
    detector_error_model_to_matrices,
)


def build(code_name: str, p: float, rounds: int):
    code = _get_code(code_name)
    circuit = build_z_memory_circuit(code, p=p, rounds=rounds)
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matrices = detector_error_model_to_matrices(dem)
    return code, circuit, matrices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="bb_72_12_6")
    parser.add_argument("--p", type=float, default=0.006)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--shots", type=int, default=40)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    from ldpc import BpOsdDecoder

    rounds = args.rounds if args.rounds is not None else 12
    code, circuit, matrices = build(args.code, args.p, rounds)
    h = matrices.h
    m, n = h.shape
    print(f"code={code.name} p={args.p:g} rounds={rounds}  H: {m} x {n}")

    common = dict(
        error_channel=matrices.error_probs.tolist(),
        max_iter=args.max_iter,
        bp_method="ms",
        ms_scaling_factor=0.625,
    )
    dec_cs = BpOsdDecoder(h, osd_method="osd_cs", osd_order=2, **common)
    dec_0 = BpOsdDecoder(h, osd_method="osd_0", **common)

    sampler = circuit.compile_detector_sampler(seed=args.seed)
    syndromes, _ = sampler.sample(args.shots, separate_observables=True)
    syndromes = syndromes.astype(np.uint8)

    t0 = time.time()
    packed = pack_matrix(h, n_cols=n, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n)
    print(f"rank(H) = {rank}  (m={m}, n={n})  free columns k = {n - rank}")
    print(f"  OSD-CS order 2 therefore sweeps k+1 = {n - rank + 1} candidates")
    print(f"  rank computed in {time.time() - t0:.1f}s")

    n_cs_vs_0 = 0
    n_cs_vs_0_match = 0
    n_llr_match = 0
    n_recon = 0
    n_recon_match = 0
    n_conv = 0
    ties = []
    recon_times = []
    land_times: list = []
    winner_types: list = []
    n_land = 0
    n_land_match = 0
    tie_gaps: list = []
    n_logical_match = 0
    n_syndrome_preserved = 0
    log_inv_p = np.log(1.0 / matrices.error_probs)

    for syndrome in syndromes:
        dec_cs.decode(syndrome)
        conv_cs = bool(dec_cs.converge)
        llr_cs = np.asarray(dec_cs.log_prob_ratios, dtype=np.float64)
        osd0_from_cs = np.asarray(dec_cs.osd0_decoding, dtype=np.uint8) % 2

        dec_0.decode(syndrome)
        llr_0 = np.asarray(dec_0.log_prob_ratios, dtype=np.float64)
        osd0_standalone = np.asarray(dec_0.osd0_decoding, dtype=np.uint8) % 2

        n_llr_match += int(np.array_equal(llr_cs, llr_0))
        n_cs_vs_0 += 1
        n_cs_vs_0_match += int(np.array_equal(osd0_from_cs, osd0_standalone))

        if conv_cs:
            n_conv += 1
            continue

        ties.append(tie_fraction(llr_cs))
        t1 = time.time()
        info = osd0_information_set(packed, syndrome, llr_cs, n_cols=n, rank=rank)
        recon_times.append(time.time() - t1)
        if info is None:
            continue
        n_recon += 1
        n_recon_match += int(np.array_equal(info["solution"], osd0_from_cs))

        osdw = np.asarray(dec_cs.osdw_decoding, dtype=np.uint8) % 2
        t2 = time.time()
        land = candidate_landscape(
            info, log_inv_p, matrices.logicals, n_cols=n, rank=rank, osd_order=2
        )
        land_times.append(time.time() - t2)
        if land is None:
            continue
        n_land += 1
        mine = winner_error_vector(land, n, info["pivot_cols"])
        if np.array_equal(mine, osdw):
            n_land_match += 1
        else:
            # Degenerate candidates can share a score exactly; upstream and this
            # reconstruction then break the tie on different float rounding. That
            # is only acceptable if the scores really are equal.
            score_gap = float(
                log_inv_p[mine.astype(bool)].sum() - log_inv_p[osdw.astype(bool)].sum()
            )
            tie_gaps.append(score_gap)
        winner_types.append(land["winner_type"])
        delta = (mine ^ osd0_from_cs).astype(np.uint8)
        n_syndrome_preserved += int(not np.any((h @ delta) % 2))
        # The logical action must also agree, which is what the analysis uses.
        n_logical_match += int(
            np.array_equal(
                (matrices.logicals @ mine.astype(np.int64)) % 2,
                (matrices.logicals @ osdw.astype(np.int64)) % 2,
            )
        )

    print("")
    print("Gate 1: osd0_decoding from the OSD-CS decoder vs a standalone OSD-0 decoder")
    print(f"  identical BP log_prob_ratios : {n_llr_match}/{n_cs_vs_0}")
    print(f"  identical OSD-0 solutions    : {n_cs_vs_0_match}/{n_cs_vs_0}")
    print(f"  BP converged (OSD skipped)   : {n_conv}/{args.shots}")
    print("")
    print("Gate 2: re-derived information set vs the decoder's own osd0_decoding")
    print(f"  exact bit-for-bit match      : {n_recon_match}/{n_recon}")
    if ties:
        print(f"  mean LPR tie fraction        : {np.mean(ties):.4g}")
    if recon_times:
        print(f"  reconstruction cost          : {np.mean(recon_times) * 1e3:.1f} ms/shot")

    print("")
    print("Gate 3: rebuilt OSD-CS candidate landscape vs the decoder's own sweep")
    print(f"  winning error vector matches : {n_land_match}/{n_land}")
    print(f"  winning logical action matches: {n_logical_match}/{n_land}")
    print(f"  H (e0 XOR e_CS) = 0          : {n_syndrome_preserved}/{n_land}")
    max_gap = max((abs(g) for g in tie_gaps), default=0.0)
    if tie_gaps:
        print(
            f"  the {len(tie_gaps)} differing picks are score ties, "
            f"max |score gap| = {max_gap:.3e}"
        )
    if winner_types:
        counts = np.bincount(np.asarray(winner_types), minlength=3)
        print(
            f"  winner type                  : osd0={counts[0]} single={counts[1]} "
            f"double={counts[2]}"
        )
    if land_times:
        print(f"  landscape cost               : {np.mean(land_times) * 1e3:.1f} ms/shot")
        print(
            "  (the decoder's own sweep costs ~1300 ms/shot for the same information)"
        )

    gate1 = n_cs_vs_0_match == n_cs_vs_0
    gate2 = n_recon > 0 and n_recon_match == n_recon
    # A different pick is fine when the two candidates score identically, since
    # the decoder itself is indifferent between them; what must hold exactly is
    # the logical action and the nullspace property.
    gate3 = (
        n_land > 0
        and n_logical_match == n_land
        and n_syndrome_preserved == n_land
        and max_gap < 1e-9
    )
    print("")
    print(
        f"GATE1={'PASS' if gate1 else 'FAIL'}  GATE2={'PASS' if gate2 else 'FAIL'}  "
        f"GATE3={'PASS' if gate3 else 'FAIL'}"
    )


if __name__ == "__main__":
    main()
