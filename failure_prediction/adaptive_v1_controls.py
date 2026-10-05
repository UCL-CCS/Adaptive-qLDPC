"""Zero-cost Adaptive V1 controls on existing bb144 matched shots.

1. BP@20 converge flag as a failure predictor (AUROC vs d_H).
2. Held-out tau: calibrate d_H threshold on 20 seeds, evaluate on the other 20.

No re-decode. Usage:
    python -m failure_prediction.adaptive_v1_controls
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np

from failure_prediction.adaptive_v1_analysis import P_VALUES, auroc, wilson
from failure_prediction.final_adaptive import (
    ESC_TARGETS,
    OUT_DIR,
    PARTS,
    bootstrap_auroc,
    jsonable,
    mix_ler,
    recovered,
    topk_mask,
)

N_CAL_SEEDS = 20
OUT_JSON = os.path.join(OUT_DIR, "controls_bp_tau.json")
OUT_MD = os.path.join(OUT_DIR, "controls_bp_tau.md")


def load_parts() -> Dict[float, List[dict]]:
    files = sorted(glob.glob(os.path.join(PARTS, "*.npz")))
    files = [f for f in files if "timing" not in os.path.basename(f)]
    by_p: Dict[float, List[dict]] = {}
    for f in files:
        d = np.load(f)
        p = float(np.asarray(d["p"]).ravel()[0])
        rec = {k: np.asarray(d[k]) for k in d.files}
        rec["seed0"] = int(np.asarray(d["seed"]).ravel()[0])
        by_p.setdefault(p, []).append(rec)
    for p in by_p:
        by_p[p].sort(key=lambda r: r["seed0"])
    return by_p


def cat_parts(parts: List[dict]) -> dict:
    keys = [k for k in parts[0] if k not in ("p", "seed0")]
    out = {k: np.concatenate([t[k] for t in parts]) for k in keys}
    out["p"] = float(np.asarray(parts[0]["p"]).ravel()[0])
    out["n"] = int(out["fail_osd0"].size)
    out["n_seeds"] = len(parts)
    return out


def tau_kth(dh: np.ndarray, f: float) -> float:
    """Threshold equal to the k-th largest d_H, k=round(f n). Escalate if d_H >= tau."""
    n = dh.size
    k = int(round(f * n))
    if k <= 0:
        return float("inf")
    if k >= n:
        return float(np.min(dh))
    order = np.argsort(-np.asarray(dh, dtype=float), kind="mergesort")
    return float(dh[order[k - 1]])


def apply_tau(d: dict, tau: float) -> dict:
    esc = np.asarray(d["d_h"], dtype=float) >= tau
    fail0, failk, failf = d["fail_osd0"], d["fail_k1000"], d["fail_full"]
    ler = mix_ler(fail0, failk, esc)
    ler0 = float(fail0.mean())
    lerf = float(failf.mean())
    n = int(fail0.size)
    kfail = int(np.where(esc, failk, fail0).sum())
    lo, hi = wilson(kfail, n)
    return {
        "tau": float(tau) if np.isfinite(tau) else None,
        "f_esc": float(esc.mean()),
        "n_esc": int(esc.sum()),
        "ler": ler,
        "ler_ci": [lo, hi],
        "n_fail": kfail,
        "n": n,
        "ler_osd0": ler0,
        "ler_full": lerf,
        "rec": recovered(ler0, lerf, ler),
    }


def oracle_topk(d: dict, f: float) -> dict:
    esc = topk_mask(d["d_h"], f)
    fail0, failk, failf = d["fail_osd0"], d["fail_k1000"], d["fail_full"]
    ler = mix_ler(fail0, failk, esc)
    ler0 = float(fail0.mean())
    lerf = float(failf.mean())
    n = int(fail0.size)
    kfail = int(np.where(esc, failk, fail0).sum())
    lo, hi = wilson(kfail, n)
    return {
        "f_target": f,
        "f_esc": float(esc.mean()),
        "ler": ler,
        "ler_ci": [lo, hi],
        "n_fail": kfail,
        "n": n,
        "rec": recovered(ler0, lerf, ler),
    }


def bp_flag_block(d: dict, rng) -> dict:
    conv = np.asarray(d["bp_converged"], dtype=float) > 0.5
    fail0 = np.asarray(d["fail_osd0"], dtype=float)
    failk = np.asarray(d["fail_k1000"], dtype=float)
    failf = np.asarray(d["fail_full"], dtype=float)
    dh = np.asarray(d["d_h"], dtype=float)
    n = fail0.size
    n_conv = int(conv.sum())
    n_nc = int((~conv).sum())
    fail_conv = fail0[conv]
    fail_nc = fail0[~conv]
    score_nc = (~conv).astype(float)
    useful = ((fail0 == 1) & (failf == 0)).astype(float)
    auc_fail = bootstrap_auroc(fail0, score_nc, rng)
    auc_ben = bootstrap_auroc(useful, score_nc, rng)
    auc_dh = bootstrap_auroc(fail0, dh, rng)
    auc_dh_nc = (
        bootstrap_auroc(fail0[~conv], dh[~conv], rng)
        if n_nc and fail0[~conv].sum() > 0 and (fail0[~conv] == 0).sum() > 0
        else (float("nan"), float("nan"), float("nan"))
    )
    # Natural BP policy: escalate every non-converged shot.
    nat = apply_tau({"d_h": score_nc, "fail_osd0": fail0, "fail_k1000": failk, "fail_full": failf}, 1.0)
    nat["policy"] = "escalate if BP@20 did not converge"
    return {
        "n": n,
        "n_converged": n_conv,
        "frac_converged": n_conv / n,
        "n_not_converged": n_nc,
        "frac_not_converged": n_nc / n,
        "n_fail_osd0": int(fail0.sum()),
        "n_fail_among_converged": int(fail_conv.sum()) if n_conv else 0,
        "n_fail_among_not_converged": int(fail_nc.sum()) if n_nc else 0,
        "ler_if_converged": float(fail_conv.mean()) if n_conv else None,
        "ler_if_not_converged": float(fail_nc.mean()) if n_nc else None,
        "auroc_bp_nonconv_fail": list(auc_fail),
        "auroc_bp_nonconv_ben": list(auc_ben),
        "auroc_dh_fail": list(auc_dh),
        "auroc_dh_fail_on_nonconv": list(auc_dh_nc),
        "natural_escalate_nonconv": {
            "f_esc": nat["f_esc"],
            "ler_k1000": nat["ler"],
            "ler_ci": nat["ler_ci"],
            "rec": nat["rec"],
            "ler_osd0": float(fail0.mean()),
            "ler_k1000_all": float(failk.mean()),
        },
        "note": (
            "Score is 1 - BP@20.converge (non-convergence = high risk). "
            "The flag is binary and not-converged is the majority class at these p, "
            "so it cannot implement a 10-30% budget."
        ),
    }


def holdout_block(parts: List[dict]) -> dict:
    n_parts = len(parts)
    assert n_parts >= 2 * N_CAL_SEEDS
    folds = [
        ("seeds_first20_cal", parts[:N_CAL_SEEDS], parts[N_CAL_SEEDS:]),
        ("seeds_last20_cal", parts[N_CAL_SEEDS:], parts[:N_CAL_SEEDS]),
    ]
    fold_rows = []
    for name, cal_p, te_p in folds:
        cal, te = cat_parts(cal_p), cat_parts(te_p)
        rows = []
        for f in ESC_TARGETS:
            tau = tau_kth(cal["d_h"], f)
            ho = apply_tau(te, tau)
            ora = oracle_topk(te, f)
            rows.append(
                {
                    "f_target": f,
                    "tau_cal": ho["tau"],
                    "f_esc_test": ho["f_esc"],
                    "ler_holdout": ho["ler"],
                    "ler_holdout_ci": ho["ler_ci"],
                    "rec_holdout": ho["rec"],
                    "ler_oracle_topk": ora["ler"],
                    "ler_oracle_ci": ora["ler_ci"],
                    "rec_oracle": ora["rec"],
                    "delta_ler_holdout_minus_oracle": ho["ler"] - ora["ler"],
                    "n_test": ho["n"],
                    "n_fail_holdout": ho["n_fail"],
                    "ler_osd0_test": ho["ler_osd0"],
                    "ler_full_test": ho["ler_full"],
                }
            )
        fold_rows.append(
            {
                "fold": name,
                "n_cal": cal["n"],
                "n_test": te["n"],
                "cal_seeds": [int(r["seed0"]) for r in cal_p],
                "test_seeds": [int(r["seed0"]) for r in te_p],
                "points": rows,
            }
        )
    # Mean over the two complementary 10k/10k folds.
    mean_pts = []
    for i, f in enumerate(ESC_TARGETS):
        a, b = fold_rows[0]["points"][i], fold_rows[1]["points"][i]
        mean_pts.append(
            {
                "f_target": f,
                "tau_cal_mean": 0.5 * (a["tau_cal"] + b["tau_cal"]),
                "f_esc_test_mean": 0.5 * (a["f_esc_test"] + b["f_esc_test"]),
                "ler_holdout_mean": 0.5 * (a["ler_holdout"] + b["ler_holdout"]),
                "ler_oracle_mean": 0.5 * (a["ler_oracle_topk"] + b["ler_oracle_topk"]),
                "rec_holdout_mean": 0.5 * (a["rec_holdout"] + b["rec_holdout"]),
                "rec_oracle_mean": 0.5 * (a["rec_oracle"] + b["rec_oracle"]),
                "delta_ler_mean": 0.5
                * (a["delta_ler_holdout_minus_oracle"] + b["delta_ler_holdout_minus_oracle"]),
            }
        )
    return {"folds": fold_rows, "mean_over_folds": mean_pts}


def fmt_auc(t: Sequence) -> str:
    x, lo, hi = t
    if x != x:
        return "n/a"
    return f"{x:.3f} [{lo:.3f},{hi:.3f}]"


def write_md(out: dict) -> str:
    lines = [
        "# Adaptive V1 controls: BP-converge AUROC and held-out $\\tau$",
        "",
        "Existing bb144 $n=20000$ matched shots. No new Monte Carlo.",
        "",
        "## 1. BP@20 converge flag as a router",
        "",
        "Score $=1-$ `BP@20.converge` (non-convergence = high OSD-0-failure risk).",
        "",
        "| $p$ | conv. frac. | OSD-0 fails in conv. / not | LER $\\mid$ conv. | LER $\\mid$ not | BP-flag fail AUROC | $d_H$ fail AUROC | $d_H$ AUROC on non-conv. | natural $f_{\\mathrm{esc}}$ (escalate all non-conv.) | natural LER |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in P_VALUES:
        b = out[str(p)]["bp_flag"]
        nfc, nfn = b["n_fail_among_converged"], b["n_fail_among_not_converged"]
        lc = b["ler_if_converged"]
        ln = b["ler_if_not_converged"]
        nat = b["natural_escalate_nonconv"]
        lines.append(
            f"| {p:g} | {100 * b['frac_converged']:.2f}% "
            f"({b['n_converged']}/{b['n']}) | {nfc} / {nfn} | "
            f"{lc:.5f} | {ln:.5f} | {fmt_auc(b['auroc_bp_nonconv_fail'])} | "
            f"{fmt_auc(b['auroc_dh_fail'])} | {fmt_auc(b['auroc_dh_fail_on_nonconv'])} | "
            f"{100 * nat['f_esc']:.2f}% | {nat['ler_k1000']:.5f} |"
        )
    lines += [
        "",
        "The binary flag cannot implement a 10–30% compute budget: not-converged is 91–99% of shots.",
        "Escalating every non-converged shot is essentially always-$K{=}1000$.",
        "$d_H$ remains predictive on the non-converged majority.",
        "",
        "## 2. Held-out $\\tau$ (20 seeds calibrate, 20 evaluate; then swap)",
        "",
        "Calibrate $\\tau$ as the $k$-th largest $d_H$ on the calibration seeds, "
        "$k=\\mathrm{round}(f n_{\\mathrm{cal}})$. Test policy: escalate iff $d_H \\ge \\tau$. "
        "Oracle on the same test seeds is in-sample exact top-$k$.",
        "",
    ]
    for p in P_VALUES:
        h = out[str(p)]["holdout"]
        lines += [
            f"### $p={p:g}$",
            "",
            "| fold | $f$ | $\\tau_{\\mathrm{cal}}$ | $f_{\\mathrm{esc}}$ test | hold-out LER | oracle top-$k$ LER | $\\Delta$ LER | hold-out rec. | oracle rec. |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for fold in h["folds"]:
            for row in fold["points"]:
                lines.append(
                    f"| {fold['fold']} | {100 * row['f_target']:.0f}% | "
                    f"{row['tau_cal']:.0f} | {100 * row['f_esc_test']:.2f}% | "
                    f"{row['ler_holdout']:.5f} [{row['ler_holdout_ci'][0]:.5f},"
                    f"{row['ler_holdout_ci'][1]:.5f}] | "
                    f"{row['ler_oracle_topk']:.5f} | "
                    f"{row['delta_ler_holdout_minus_oracle']:+.5f} | "
                    f"{100 * row['rec_holdout']:.1f}% | {100 * row['rec_oracle']:.1f}% |"
                )
        lines += [
            "",
            "Mean over the two complementary folds:",
            "",
            "| $f$ | mean $\\tau$ | mean $f_{\\mathrm{esc}}$ | mean hold-out LER | mean oracle LER | mean $\\Delta$ | mean rec. hold-out |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in h["mean_over_folds"]:
            lines.append(
                f"| {100 * row['f_target']:.0f}% | {row['tau_cal_mean']:.1f} | "
                f"{100 * row['f_esc_test_mean']:.2f}% | {row['ler_holdout_mean']:.5f} | "
                f"{row['ler_oracle_mean']:.5f} | {row['delta_ler_mean']:+.5f} | "
                f"{100 * row['rec_holdout_mean']:.1f}% |"
            )
        lines.append("")
    lines += [
        "In-sample top-$k$ on the evaluation seeds does not materially beat a $\\tau$ frozen on the complementary seeds.",
        "The paper's main curves remain budget-controlled exact top-$k$; this table is the independent-calibration check.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    rng = np.random.default_rng(20260902)
    by_p = load_parts()
    out = {}
    for p in P_VALUES:
        parts = by_p[p]
        d = cat_parts(parts)
        print(f"p={p:g} n={d['n']} seeds={len(parts)}", flush=True)
        out[str(p)] = {
            "bp_flag": bp_flag_block(d, rng),
            "holdout": holdout_block(parts),
        }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(jsonable(out), fh, indent=2)
    md = write_md(out)
    with open(OUT_MD, "w") as fh:
        fh.write(md)
    print(md)
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
