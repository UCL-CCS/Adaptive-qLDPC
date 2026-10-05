"""Survey the (cost, logical error rate) frontier of qLDPC decoders.

Motivation
----------
Adaptive decoding only buys something if the reference decoder is worth its
price. The Stage B calibration found that BP@20 + OSD-0 reaches the same LER as
BP+OSD-CS order 2/4/8 at a sixth of the cost, so routing hard syndromes to the
expensive decoder saved nothing: a third decoder dominated both endpoints of the
pair.

This module therefore measures every candidate decoder on the same shots and
asks two questions per noise point:

  1. which decoders lie on the Pareto frontier of (mean ms/shot, LER)?
  2. for each frontier pair (cheap `fast`, expensive `strong`), what is the best
     possible adaptive speedup, using an oracle that escalates exactly the shots
     the fast decoder gets wrong?

The oracle is the upper bound over all hardness predictors, so a pair that
fails here cannot be rescued by better machine learning. The screening
criterion is

    adaptive_cost = cost_fast + LER_fast * cost_strong  <  cost_strong

which requires LER_fast < 1 - cost_fast / cost_strong.
"""

from __future__ import annotations

import argparse
import json
import os
from time import perf_counter
from typing import Any, Dict, List, Tuple

import numpy as np

from config import OUTPUT_DIR
from failure_prediction.qldpc_circuit import (
    _get_code,
    build_z_memory_circuit,
    detector_error_model_to_matrices,
)

BP_METHOD = "ms"
MS_SCALING_FACTOR = 0.625

# (name, kind, kwargs) where kind selects the ldpc class.
DECODER_SPECS: List[Tuple[str, str, Dict[str, Any]]] = [
    ("bp10", "bp", {"max_iter": 10}),
    ("bp20", "bp", {"max_iter": 20}),
    ("bp50", "bp", {"max_iter": 50}),
    ("bp10_osd0", "bposd", {"max_iter": 10, "osd_method": "osd_0"}),
    ("bp20_osd0", "bposd", {"max_iter": 20, "osd_method": "osd_0"}),
    ("bp50_osd0", "bposd", {"max_iter": 50, "osd_method": "osd_0"}),
    # Matched BP effort for the OSD-0 / OSD-CS pair, used by the BP-iteration control.
    ("bp10_osdcs2", "bposd", {"max_iter": 10, "osd_method": "osd_cs", "osd_order": 2}),
    ("bp20_osdcs2", "bposd", {"max_iter": 20, "osd_method": "osd_cs", "osd_order": 2}),
    ("bp50_osdcs2", "bposd", {"max_iter": 50, "osd_method": "osd_cs", "osd_order": 2}),
    ("bp50_osdcs8", "bposd", {"max_iter": 50, "osd_method": "osd_cs", "osd_order": 8}),
    ("bp20_lsd", "bplsd", {"max_iter": 20}),
    ("bp50_lsd", "bplsd", {"max_iter": 50}),
]


def select_specs(names: List[str] | None):
    if not names:
        return DECODER_SPECS
    known = {spec[0] for spec in DECODER_SPECS}
    unknown = set(names) - known
    if unknown:
        raise ValueError(f"unknown decoders {sorted(unknown)}; known: {sorted(known)}")
    return [spec for spec in DECODER_SPECS if spec[0] in set(names)]


def build_decoders(matrices, specs=DECODER_SPECS):
    from ldpc import BpDecoder, BpOsdDecoder
    from ldpc.bplsd_decoder import BpLsdDecoder

    classes = {"bp": BpDecoder, "bposd": BpOsdDecoder, "bplsd": BpLsdDecoder}
    error_channel = matrices.error_probs.tolist()
    built = []
    for name, kind, kwargs in specs:
        built.append(
            (
                name,
                classes[kind](
                    matrices.h,
                    error_channel=error_channel,
                    bp_method=BP_METHOD,
                    ms_scaling_factor=MS_SCALING_FACTOR,
                    **kwargs,
                ),
            )
        )
    return built


def _pareto_frontier(rows: List[Dict[str, Any]]) -> List[str]:
    """Names of decoders not dominated on both cost and LER (lower is better)."""

    frontier = []
    for a in rows:
        dominated = any(
            b["decoder"] != a["decoder"]
            and b["mean_ms"] <= a["mean_ms"]
            and b["ler"] <= a["ler"]
            and (b["mean_ms"] < a["mean_ms"] or b["ler"] < a["ler"])
            for b in rows
        )
        if not dominated:
            frontier.append(a["decoder"])
    return frontier


def _mcnemar(fast_only: int, strong_only: int) -> Dict[str, float]:
    """Exact-ish paired test on discordant shots for two decoders on the same data.

    Aggregate LER differences are misleading here: the decoders see identical
    syndromes, so the informative quantity is the split of the shots where
    exactly one of them fails.
    """

    n = fast_only + strong_only
    if n == 0:
        return {"n_discordant": 0, "z": 0.0, "p_value_two_sided": 1.0}
    z = (fast_only - strong_only) / np.sqrt(n)
    from math import erfc, sqrt

    return {
        "n_discordant": int(n),
        "z": float(z),
        "p_value_two_sided": float(erfc(abs(z) / sqrt(2.0))),
    }


def _wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def run_point(
    code_name: str,
    p: float,
    shots: int,
    rounds: int | None,
    seed: int,
    progress_every: int = 0,
    decoder_names: List[str] | None = None,
) -> Dict[str, Any]:
    code = _get_code(code_name)
    if rounds is None:
        rounds = 2 * (code.d if getattr(code, "d", None) else 6)
    circuit = build_z_memory_circuit(code, p=p, rounds=rounds)
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True)
    matrices = detector_error_model_to_matrices(dem)
    print(
        f"DEM: {matrices.h.shape[0]} detectors x {matrices.h.shape[1]} mechanisms",
        flush=True,
    )
    decoders = build_decoders(matrices, select_specs(decoder_names))
    logicals = matrices.logicals.astype(np.int32)

    sampler = circuit.compile_detector_sampler(seed=seed)
    syndromes, actual = sampler.sample(shots, separate_observables=True)
    syndromes = syndromes.astype(np.uint8)
    actual = actual.astype(np.uint8)

    wrong: Dict[str, np.ndarray] = {}
    predictions: Dict[str, np.ndarray] = {}
    times: Dict[str, np.ndarray] = {}
    converged: Dict[str, np.ndarray] = {}

    for name, decoder in decoders:
        pred = np.zeros_like(actual)
        elapsed = np.zeros(shots, dtype=np.float64)
        conv = np.zeros(shots, dtype=bool)
        for i, syndrome in enumerate(syndromes):
            t0 = perf_counter()
            correction = np.asarray(decoder.decode(syndrome), dtype=np.int32) % 2
            elapsed[i] = perf_counter() - t0
            pred[i] = (logicals @ correction) % 2
            conv[i] = bool(decoder.converge)
            if progress_every and (i + 1) % progress_every == 0:
                print(f"    {name}: {i + 1}/{shots}", flush=True)
        predictions[name] = pred
        wrong[name] = np.any(pred != actual, axis=1)
        times[name] = elapsed
        converged[name] = conv
        print(
            f"  {name:14s} LER={wrong[name].mean():.6f} "
            f"({int(wrong[name].sum())})  {elapsed.mean() * 1e3:8.3f} ms/shot  "
            f"conv={conv.mean():.4f}",
            flush=True,
        )

    rows = []
    for name, _ in decoders:
        n_fail = int(wrong[name].sum())
        lo, hi = _wilson(n_fail, shots)
        rows.append(
            {
                "decoder": name,
                "ler": float(wrong[name].mean()),
                "n_fail": n_fail,
                "ler_ci95": [lo, hi],
                "mean_ms": float(times[name].mean() * 1e3),
                "median_ms": float(np.median(times[name]) * 1e3),
                "converged_rate": float(converged[name].mean()),
            }
        )
    frontier = _pareto_frontier(rows)
    by_name = {r["decoder"]: r for r in rows}

    # Pairs are formed over all decoders, not just the frontier: a control that
    # compares two decoders of equal LER would otherwise lose its pair record,
    # since the cheaper one dominates and drops the other off the frontier.
    all_names = [name for name, _ in decoders]
    pairs = []
    for fast in all_names:
        for strong in all_names:
            if by_name[strong]["mean_ms"] <= by_name[fast]["mean_ms"]:
                continue
            ler_fast = by_name[fast]["ler"]
            cost_fast = by_name[fast]["mean_ms"]
            cost_strong = by_name[strong]["mean_ms"]

            # Oracle: escalate exactly the shots the fast decoder gets wrong.
            escalate = wrong[fast]
            oracle_fail = int(np.sum(escalate & wrong[strong]))
            oracle_cost = cost_fast + float(escalate.mean()) * cost_strong

            # Realistic cheap policy: escalate whenever the fast decoder did not converge.
            flag = ~converged[fast]
            flag_fail = int(np.sum(wrong[fast] & ~flag)) + int(np.sum(wrong[strong] & flag))
            flag_cost = cost_fast + float(flag.mean()) * cost_strong

            fast_only = int(np.sum(wrong[fast] & ~wrong[strong]))
            strong_only = int(np.sum(wrong[strong] & ~wrong[fast]))

            pairs.append(
                {
                    "fast": fast,
                    "strong": strong,
                    "both_on_frontier": bool(fast in frontier and strong in frontier),
                    "ler_fast": ler_fast,
                    "ler_strong": by_name[strong]["ler"],
                    "n_fast_only_wrong": fast_only,
                    "n_strong_only_wrong": strong_only,
                    "mcnemar": _mcnemar(fast_only, strong_only),
                    "cost_fast_ms": cost_fast,
                    "cost_strong_ms": cost_strong,
                    "cost_ratio": cost_strong / max(cost_fast, 1e-12),
                    "oracle_escalated_fraction": float(escalate.mean()),
                    "oracle_ler": oracle_fail / shots,
                    "oracle_cost_ms": oracle_cost,
                    "oracle_speedup_vs_strong": cost_strong / max(oracle_cost, 1e-12),
                    "flag_escalated_fraction": float(flag.mean()),
                    "flag_ler": flag_fail / shots,
                    "flag_cost_ms": flag_cost,
                    "flag_speedup_vs_strong": cost_strong / max(flag_cost, 1e-12),
                    "viable": bool(
                        oracle_cost < cost_strong
                        and by_name[strong]["ler"] < by_name[fast]["ler"]
                    ),
                }
            )
    pairs.sort(key=lambda r: (not r["both_on_frontier"], -r["oracle_speedup_vs_strong"]))

    return {
        "code": code.name,
        "p": float(p),
        "rounds": int(rounds),
        "shots": int(shots),
        "seed": int(seed),
        "n_detectors": int(matrices.h.shape[0]),
        "n_error_mechanisms": int(matrices.h.shape[1]),
        "decoders": rows,
        "pareto_frontier": frontier,
        "pairs": pairs,
    }


def format_point(result: Dict[str, Any]) -> str:
    lines = [
        f"Decoder survey: {result['code']}, p={result['p']:g}, rounds={result['rounds']}, "
        f"shots={result['shots']}",
        f"  DEM: {result['n_detectors']} detectors x {result['n_error_mechanisms']} mechanisms",
        "",
        f"  {'decoder':14s} {'LER':>10s} {'n_fail':>7s} {'ms/shot':>9s} {'conv':>7s} {'pareto':>7s}",
    ]
    frontier = set(result["pareto_frontier"])
    for row in sorted(result["decoders"], key=lambda r: r["mean_ms"]):
        mark = "*" if row["decoder"] in frontier else ""
        lines.append(
            f"  {row['decoder']:14s} {row['ler']:10.6f} {row['n_fail']:7d} "
            f"{row['mean_ms']:9.3f} {row['converged_rate']:7.4f} {mark:>7s}"
        )
    lines.append("")
    lines.append("  Pairs, frontier pairs first then best adaptive speedup (oracle = upper bound):")
    if not result["pairs"]:
        lines.append("    none: fewer than two decoders with distinct cost")
    header = (
        f"    {'fast':14s} {'strong':14s} {'LERfast':>9s} {'LERstr':>9s} "
        f"{'cost x':>7s} {'orcl esc':>9s} {'orcl LER':>9s} {'orcl spd':>9s} "
        f"{'fOnly':>6s} {'sOnly':>6s} {'McN p':>9s} {'front':>6s} {'viable':>7s}"
    )
    lines.append(header)
    for row in result["pairs"][:12]:
        mcn = row.get("mcnemar") or {}
        lines.append(
            f"    {row['fast']:14s} {row['strong']:14s} {row['ler_fast']:9.5f} "
            f"{row['ler_strong']:9.5f} {row['cost_ratio']:7.2f} "
            f"{row['oracle_escalated_fraction']:9.4f} {row['oracle_ler']:9.5f} "
            f"{row['oracle_speedup_vs_strong']:9.2f} "
            f"{row.get('n_fast_only_wrong', 0):6d} {row.get('n_strong_only_wrong', 0):6d} "
            f"{mcn.get('p_value_two_sided', float('nan')):9.3g} "
            f"{str(row.get('both_on_frontier', False)):>6s} {str(row['viable']):>7s}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", default="bb_72_12_6", choices=["bb_72_12_6", "bb_144_12_12"])
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument(
        "--decoders",
        default=None,
        help="comma-separated subset of decoder names (default: all)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_point(
        code_name=args.code,
        p=args.p,
        shots=args.shots,
        rounds=args.rounds,
        seed=args.seed,
        progress_every=args.progress_every,
        decoder_names=args.decoders.split(",") if args.decoders else None,
    )
    out_dir = args.out_dir or os.path.join(OUTPUT_DIR, "failure_prediction", "decoder_survey")
    os.makedirs(out_dir, exist_ok=True)
    stem = f"{result['code']}_p{result['p']:g}_r{result['rounds']}_n{result['shots']}_seed{result['seed']}"
    with open(os.path.join(out_dir, stem + ".json"), "w") as f:
        json.dump(result, f, indent=2)
    text = format_point(result)
    with open(os.path.join(out_dir, stem + ".txt"), "w") as f:
        f.write(text)
    print()
    print(text)
