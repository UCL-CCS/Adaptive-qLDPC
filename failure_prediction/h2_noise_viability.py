"""Zero-HQC H2 viability test: amplify native Stim noise on bb_36.

Does not submit Nexus / H2 hardware jobs. Does not redesign Adaptive V1.

Noise model (same as h2_adaptive_screen): a single circuit-level depolarizing
parameter p, applied by build_z_memory_circuit. Native operating point is
P_H2 = 0.00105 (H2-1 two-qubit fault probability, 2025-04-30). Amplification
is p(lambda) = lambda * P_H2. This is not a full H2 noise_specs model.

Usage:
    python -m failure_prediction.h2_noise_viability
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import sparse

from failure_prediction.adaptive_v1_analysis import auroc
from failure_prediction.h2_adaptive_screen import (
    P_H2,
    SEED,
    _row_basis_csr,
    decode_circuit,
    eval_css,
)
from failure_prediction.qldpc_codes import (
    CSSCode,
    _bb_polynomial_matrix,
    parity_check_rank_mod2,
)

OUT_DIR = "outputs/failure_prediction/h2_noise_viability"
ROUNDS = 12
LAMBDAS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
SHOTS_PASS1 = 4000
SHOTS_PASS2 = 10000
ESC = (0.10, 0.20, 0.30)

# Screening gates (not theoretical claims).
LER_LO, LER_HI = 0.03, 0.20
R_MIN = 2.0
AUROC_MIN = 0.80
# Near-miss: warrant a 10k confirmation, not a GO by themselves.
NEAR_LER_LO, NEAR_LER_HI = 0.02, 0.25
NEAR_R, NEAR_AUROC = 1.70, 0.75


def bb_36() -> CSSCode:
    """Largest already-screened H2-fit BB code: 6x3 torus, bb144 polynomials."""
    a = _bb_polynomial_matrix(6, 3, [(3, 0), (0, 1), (0, 2)])
    b = _bb_polynomial_matrix(6, 3, [(0, 3), (1, 0), (2, 0)])
    hx = _row_basis_csr(sparse.hstack([a, b], format="csr", dtype=np.uint8))
    hz = _row_basis_csr(sparse.hstack([b.T, a.T], format="csr", dtype=np.uint8))
    n = hx.shape[1]
    k = n - parity_check_rank_mod2(hx) - parity_check_rank_mod2(hz)
    return CSSCode(hx=hx, hz=hz, name="bb_36", n=n, k=k)


def _jsonable(obj):
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


def classify(row: dict) -> str:
    ler0 = row["ler_osd0"]
    r = row["gap_osd0_over_full"]
    auc = row["auroc_fail"]
    if not np.isfinite(r) or np.isnan(auc):
        return "invalid"
    if LER_LO <= ler0 <= LER_HI and r >= R_MIN and auc >= AUROC_MIN:
        return "window"
    if NEAR_LER_LO <= ler0 <= NEAR_LER_HI and r >= NEAR_R and auc >= NEAR_AUROC:
        return "near"
    if ler0 > LER_HI:
        return "saturated"
    if ler0 < LER_LO:
        return "too_quiet"
    return "no_gap"


def topk_mask(score: np.ndarray, f: float) -> np.ndarray:
    n = score.size
    k = int(round(f * n))
    esc = np.zeros(n, dtype=bool)
    if k <= 0:
        return esc
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    esc[order[:k]] = True
    return esc


def mix_ler(fail0, failk, esc) -> float:
    return float(np.where(esc, failk, fail0).mean())


def random_lers(fail0, failk, f, rng, n_rep=100) -> Tuple[float, float, float]:
    n = fail0.size
    k = int(round(f * n))
    out = np.empty(n_rep)
    idx = np.arange(n)
    for i in range(n_rep):
        esc = np.zeros(n, dtype=bool)
        pick = rng.choice(idx, size=k, replace=False)
        esc[pick] = True
        out[i] = mix_ler(fail0, failk, esc)
    return float(out.mean()), float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def decode_tables(code: CSSCode, p: float, shots: int, seed: int) -> Tuple[dict, dict]:
    """Return (summary row, per-shot table) using identical Stim shots."""
    from ldpc import BpOsdDecoder

    from failure_prediction.osd_internals import gf2_rank_packed, pack_matrix
    from failure_prediction.qldpc_circuit import (
        build_z_memory_circuit,
        detector_error_model_to_matrices,
    )
    from failure_prediction.adaptive_v1 import evaluate_batch

    circuit = build_z_memory_circuit(code, p=p, rounds=ROUNDS)
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
        max_iter=20,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    syn, actual = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )
    table = evaluate_batch(
        decoder,
        syn.astype(np.uint8),
        actual.astype(np.uint8),
        packed,
        log_inv_p,
        logicals,
        n_cols,
        rank,
        progress_every=max(shots // 4, 0),
    )
    fail0 = table["fail_osd0"]
    fail_full = table["fail_full"]
    useful = (fail0 == 1) & (fail_full == 0)
    n = shots
    k0, kfull = int(fail0.sum()), int(fail_full.sum())
    ler0, ler_full = k0 / n, kfull / n
    gap = ler0 / ler_full if ler_full > 0 else float("inf")
    row = {
        "lambda": None,
        "p": p,
        "rounds": ROUNDS,
        "n": n,
        "seed": seed,
        "n_free": int(n_cols - rank),
        "osd_rank": int(rank),
        "n_error_mechs": int(n_cols),
        "ler_osd0": ler0,
        "ler_k1000": float(table["fail_k1000"].mean()),
        "ler_full_1fv": ler_full,
        "gap_osd0_over_full": gap,
        "n_fail_osd0": k0,
        "n_fail_full": kfull,
        "n_beneficial": int(useful.sum()),
        "auroc_fail": auroc(fail0, table["d_h"]),
        "auroc_beneficial": auroc(useful, table["d_h"]),
        "bp_converged": float(table["bp_converged"].mean()),
    }
    return row, table


def adaptive_at(table: dict, rng) -> List[dict]:
    fail0 = table["fail_osd0"]
    failk = table["fail_k1000"]
    failf = table["fail_full"]
    dh = table["d_h"]
    ler0 = float(fail0.mean())
    lerf = float(failf.mean())
    gap = ler0 - lerf
    rows = []
    for f in ESC:
        esc = topk_mask(dh, f)
        ler_dh = mix_ler(fail0, failk, esc)
        rnd_m, rnd_lo, rnd_hi = random_lers(fail0, failk, f, rng)
        rec = (ler0 - ler_dh) / gap if gap > 0 else float("nan")
        rec_r = (ler0 - rnd_m) / gap if gap > 0 else float("nan")
        rows.append(
            {
                "f": f,
                "ler_dh": ler_dh,
                "ler_rnd": rnd_m,
                "ler_rnd_ci": [rnd_lo, rnd_hi],
                "rec_dh": rec,
                "rec_rnd": rec_r,
            }
        )
    return rows


def fig_viability(rows: List[dict]) -> str:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    lam = np.array([r["lambda"] for r in rows], dtype=float)
    ler0 = np.array([r["ler_osd0"] for r in rows])
    lerf = np.array([r["ler_full_1fv"] for r in rows])
    R = np.array([r["gap_osd0_over_full"] for r in rows])
    auc = np.array([r["auroc_fail"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55))
    ax = axes[0]
    ax.axhspan(LER_LO, LER_HI, color="#d9ead3", alpha=0.55, zorder=0, label="usable LER band")
    ax.plot(lam, ler0, "o-", color="#6b6b6b", lw=1.8, ms=6, label="OSD-0")
    ax.plot(lam, lerf, "s--", color="#2a9d8f", lw=1.6, ms=5.5, label="full 1FV")
    ax.set_xlabel(r"noise amplification  $\lambda$")
    ax.set_ylabel("logical error rate")
    ax.set_title(r"bb_36  $r{=}12$  (Stim $p=\lambda\,p_{\mathrm{H2}}$)")
    ax.set_xticks(lam)
    ax.set_ylim(0.0, min(1.0, max(ler0.max() * 1.15, 0.25)))
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.axhline(R_MIN, color="#1f4e79", ls=":", lw=1.0, label=r"$R{=}2$")
    ax.axhline(AUROC_MIN, color="#c44e52", ls=":", lw=1.0, label="AUROC 0.8")
    ax.plot(lam, R, "o-", color="#1f4e79", lw=1.8, ms=6, label="headroom $R$")
    ax.plot(lam, auc, "D--", color="#c44e52", lw=1.6, ms=5.5, label=r"$d_H$ fail AUROC")
    ax.set_xlabel(r"noise amplification  $\lambda$")
    ax.set_ylabel("headroom  /  AUROC")
    ax.set_title("Decoder gap vs routing signal")
    ax.set_xticks(lam)
    ymax = max(2.4, float(np.nanmax(R)) * 1.12, 1.05)
    ax.set_ylim(0.4, ymax)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "fig_h2_noise_viability.png")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fmt_table(rows: List[dict]) -> str:
    lines = [
        "lambda | p_eff     | OSD0 LER | full-1FV | headroom | "
        "d_H AUROC | OSD0 fails | rescues | class",
        "-" * 102,
    ]
    for r in rows:
        gap = r["gap_osd0_over_full"]
        gap_s = f"{gap:.2f}x" if np.isfinite(gap) else "inf"
        auc = r["auroc_fail"]
        auc_s = f"{auc:.3f}" if auc == auc else "  n/a"
        lines.append(
            f"{r['lambda']:6.1f} | {r['p']:.6f} | {r['ler_osd0']:.4f}  | "
            f"{r['ler_full_1fv']:.4f}  | {gap_s:8s} | {auc_s:9s} | "
            f"{r['n_fail_osd0']:10d} | {r['n_beneficial']:7d} | {r['class']}"
        )
    return "\n".join(lines)


def write_report(pass1, pass2, fig_path, decision) -> str:
    lines = [
        "# H2 noise-amplification viability — bb_36 (ZERO HQC)",
        "",
        "Code: `bb_36` (6×3 torus, same polynomials as the H2 screen), r=12.",
        f"Native model: scalar Stim circuit-level depolarizing $p=\\lambda\\,p_{{\\mathrm{{H2}}}}$, "
        f"$p_{{\\mathrm{{H2}}}}={P_H2}$. Not a full H2 `noise_specs` model.",
        "Decoders frozen: BP@20+OSD-0 vs full one-free-variable search. 0 HQC.",
        "",
        "## First pass (4000 matched shots)",
        "",
        "```",
        fmt_table(pass1),
        "```",
        "",
    ]
    if pass2:
        lines += ["## Second pass (10000 matched shots)", "", "```", fmt_table(pass2), "```", ""]
        for r in pass2:
            if r.get("adaptive"):
                lines.append(f"Adaptive $K=1000$ (or full 1FV if $k_{{\\mathrm{{free}}}}<1000$) at $\\lambda={r['lambda']:g}$, $n={r['n']}$, $k_{{\\mathrm{{free}}}}={r['n_free']}$:")
                for a in r["adaptive"]:
                    lines.append(
                        f"- {100*a['f']:.0f}%  $d_H$ LER={a['ler_dh']:.4f} rec={100*a['rec_dh']:.1f}%  | "
                        f"random LER={a['ler_rnd']:.4f} [{a['ler_rnd_ci'][0]:.4f},{a['ler_rnd_ci'][1]:.4f}]"
                    )
                lines.append("")
    lines += ["## Decision", "", f"**{decision['verdict']}**", ""]
    for k in ("reason", "first_lambda", "ler_regime", "dh_predictive", "gate_folding"):
        if decision.get(k):
            lines.append(f"- {decision[k]}")
    lines += ["", f"Figure: `{fig_path}`", ""]
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "VIABILITY_REPORT.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def decide(pass1: List[dict], pass2: List[dict] | None) -> dict:
    rows = pass2 if pass2 else pass1
    windows = [r for r in rows if r["class"] == "window"]
    nears = [r for r in rows if r["class"] == "near"]
    max_r = max(r["gap_osd0_over_full"] for r in rows)
    max_auc = max((r["auroc_fail"] if r["auroc_fail"] == r["auroc_fail"] else 0.0) for r in rows)
    native = pass1[0]
    if windows:
        first = min(windows, key=lambda r: r["lambda"])
        return {
            "verdict": "GO (simulation only — do not submit hardware)",
            "reason": (
                f"A usable amplified-noise window exists at lambda={first['lambda']:g} "
                f"(p={first['p']:.5f}): R={first['gap_osd0_over_full']:.2f}, "
                f"fail AUROC={first['auroc_fail']:.3f}, OSD-0 LER={first['ler_osd0']:.3f}."
            ),
            "first_lambda": f"Smallest qualifying lambda = {first['lambda']:g}.",
            "ler_regime": "OSD-0 LER is inside the 3–20% screening band.",
            "dh_predictive": f"d_H fail AUROC = {first['auroc_fail']:.3f} (>= 0.8).",
            "gate_folding": (
                "A hardware gate-folding stress test could be discussed separately; "
                "do not submit HQC in this stage."
            ),
        }
    # NO-GO
    if native["ler_osd0"] > LER_HI and max_r < R_MIN:
        ler_msg = (
            f"Native lambda=1 already has OSD-0 LER={native['ler_osd0']:.3f} "
            f"(above the 20% band). Amplifying noise only drives the code deeper "
            f"into saturation; headroom never reaches 2x (max R={max_r:.2f})."
        )
    elif max_r < R_MIN:
        ler_msg = (
            f"OSD-0 and full 1FV degrade together. Max headroom R={max_r:.2f} < 2. "
            f"No scientifically useful decoder gap appears before saturation."
        )
    else:
        ler_msg = (
            f"Headroom briefly approaches {max_r:.2f} but not together with a "
            f"usable non-saturated LER band and AUROC>=0.8."
        )
    return {
        "verdict": "NO-GO",
        "reason": (
            "H2-compatible bb_36 is too small / too weakly structured for the "
            "Adaptive V1 phenomenon, rather than merely sitting at too-low native noise. "
            + ler_msg
        ),
        "first_lambda": "No lambda in {1,1.5,2,3,4,6,8} produces R>=2 in a usable LER regime.",
        "ler_regime": (
            f"Native OSD-0 LER={native['ler_osd0']:.3f}; "
            f"lambda=8 OSD-0 LER={pass1[-1]['ler_osd0']:.3f}."
        ),
        "dh_predictive": (
            f"Max d_H fail AUROC in the sweep is {max_auc:.3f}"
            + ("; routing can look decent even when there is nothing useful to escalate to."
               if max_auc >= 0.75 else ".")
        ),
        "gate_folding": (
            "A hardware gate-folding stress test is not scientifically justified: "
            "amplifying noise will not create a bb144-like fast/strong gap on this code. "
            "Permanently stop the artificial-noise hardware route."
        ),
    }


def run_lambda(lam: float, shots: int, seed: int) -> dict:
    code = bb_36()
    p = float(lam) * P_H2
    row = eval_css(code, p, ROUNDS, shots, seed)
    row["lambda"] = float(lam)
    row["class"] = classify(row)
    return row


def main() -> None:
    import argparse
    from concurrent.futures import ProcessPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    code = bb_36()
    print(
        f"bb_36 n={code.n} k={code.k}  r={ROUNDS}  P_H2={P_H2}  "
        f"shots={SHOTS_PASS1}  seed={SEED}  workers={args.workers}  (local Stim, 0 HQC)",
        flush=True,
    )
    print(
        "Noise: scalar circuit-level depolarizing p = lambda * P_H2 "
        "(same construction as h2_adaptive_screen).",
        flush=True,
    )
    jobs = [(lam, SHOTS_PASS1, SEED + 17 * i) for i, lam in enumerate(LAMBDAS)]
    pass1_map: Dict[float, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        fut = {ex.submit(run_lambda, lam, shots, seed): lam for lam, shots, seed in jobs}
        for f in as_completed(fut):
            row = f.result()
            pass1_map[row["lambda"]] = row
            print(
                f"   lambda={row['lambda']:g}  OSD0={row['ler_osd0']:.4f}[{row['n_fail_osd0']}]  "
                f"1FV={row['ler_full_1fv']:.4f}[{row['n_fail_full']}]  "
                f"R={row['gap_osd0_over_full']:.2f}  "
                f"AUROC={row['auroc_fail']:.3f}  ben={row['n_beneficial']}  "
                f"class={row['class']}  k_free={row['n_free']}",
                flush=True,
            )
    pass1 = [pass1_map[lam] for lam in LAMBDAS]

    promising = [r for r in pass1 if r["class"] in ("window", "near")]
    pass2: List[dict] = []
    if promising:
        print(f"\nSecond pass on {len(promising)} point(s) at n={SHOTS_PASS2}", flush=True)
        rng = np.random.default_rng(20260831)
        for r0 in promising:
            lam = r0["lambda"]
            p = lam * P_H2
            seed = SEED + 1000 + int(round(10 * lam))
            print(f"-- confirm lambda={lam:g}  p={p:.6f}  n={SHOTS_PASS2}", flush=True)
            row, table = decode_tables(code, p, SHOTS_PASS2, seed)
            row["lambda"] = lam
            row["class"] = classify(row)
            row["adaptive"] = adaptive_at(table, rng)
            pass2.append(row)
            print(
                f"   OSD0={row['ler_osd0']:.4f}  1FV={row['ler_full_1fv']:.4f}  "
                f"R={row['gap_osd0_over_full']:.2f}  AUROC={row['auroc_fail']:.3f}  "
                f"class={row['class']}",
                flush=True,
            )

    fig_path = fig_viability(pass1)
    decision = decide(pass1, pass2 or None)
    report = write_report(pass1, pass2, fig_path, decision)
    dump = {
        "noise_model": "stim_circuit_level_single_p",
        "p_h2": P_H2,
        "p_of_lambda": "lambda * P_H2",
        "code": "bb_36",
        "rounds": ROUNDS,
        "pass1": pass1,
        "pass2": pass2,
        "decision": decision,
        "figure": fig_path,
    }
    with open(os.path.join(OUT_DIR, "viability_results.json"), "w") as fh:
        json.dump(_jsonable(dump), fh, indent=2)
    print("\n" + fmt_table(pass1))
    print("\n" + decision["verdict"])
    print(decision["reason"])
    print("wrote", report, fig_path, flush=True)


if __name__ == "__main__":
    main()
