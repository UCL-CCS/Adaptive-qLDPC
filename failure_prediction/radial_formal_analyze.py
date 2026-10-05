"""Analyse formal radial_90_8 5k+20k shots. Does not touch BB or the pilot dir."""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from failure_prediction.adaptive_v1_analysis import auroc, wilson
from failure_prediction.audit_residual_control import spearman
from failure_prediction.final_adaptive import (
    ESC_TARGETS,
    N_BOOT,
    N_RANDOM,
    boot_ci,
    bootstrap_auroc,
    bootstrap_mean,
    jsonable,
    mix_ler,
    random_lers,
    recovered,
    topk_mask,
)

OUT = "outputs/failure_prediction/nonbb_radial_formal"
CFG_PATH = os.path.join(OUT, "radial_90_8_frozen_config.json")
REPORT = os.path.join(OUT, "radial_cross_family_formal_report.md")
RNG_SEED = 20260908


def frozen_p():
    with open(CFG_PATH) as fh:
        return tuple(float(x) for x in json.load(fh)["p"])


def load_split(split: str) -> Dict[float, dict]:
    files = sorted(glob.glob(os.path.join(OUT, split, "*.npz")))
    by_p: Dict[float, List[dict]] = {}
    skip = {"e_bp", "e_osd0", "syndrome", "actual_obs", "p", "seed"}
    for f in files:
        d = np.load(f)
        p = float(np.asarray(d["p"]).ravel()[0])
        rec = {k: np.asarray(d[k]) for k in d.files if k not in skip}
        rec["seed0"] = int(np.asarray(d["seed"]).ravel()[0])
        by_p.setdefault(p, []).append(rec)
    out = {}
    for p, parts in by_p.items():
        parts = sorted(parts, key=lambda r: r["seed0"])
        keys = [k for k in parts[0] if k != "seed0"]
        merged = {k: np.concatenate([t[k] for t in parts]) for k in keys}
        merged["p"] = p
        merged["n"] = int(merged["fail_osd0"].size)
        merged["seeds"] = [int(t["seed0"]) for t in parts]
        out[p] = merged
    return out


def fmt_ci(x, lo, hi, nd=5) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.{nd}f} [{lo:.{nd}f},{hi:.{nd}f}]"


def fmt_auc(t) -> str:
    x, lo, hi = t
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.3f} [{lo:.3f},{hi:.3f}]"


def transitions(fail0, failk) -> dict:
    f0 = np.asarray(fail0).astype(int)
    fk = np.asarray(failk).astype(int)
    return {
        "CC": int(((f0 == 0) & (fk == 0)).sum()),
        "CW": int(((f0 == 0) & (fk == 1)).sum()),
        "WC": int(((f0 == 1) & (fk == 0)).sum()),
        "WW": int(((f0 == 1) & (fk == 1)).sum()),
        "net": int(f0.sum() - fk.sum()),
    }


def bootstrap_paired_diff(fail0, failk, sa, sb, f, rng, n_boot=N_BOOT):
    n = fail0.size
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, n)
        f0, fk, a, bb = fail0[ix], failk[ix], sa[ix], sb[ix]
        diffs[b] = mix_ler(f0, fk, topk_mask(a, f)) - mix_ler(f0, fk, topk_mask(bb, f))
    point = mix_ler(fail0, failk, topk_mask(sa, f)) - mix_ler(
        fail0, failk, topk_mask(sb, f)
    )
    return float(point), *boot_ci(diffs)


def tau_for_fraction(score: np.ndarray, f: float) -> float:
    return float(np.quantile(np.asarray(score, dtype=float), 1.0 - f))


def analyse_p(test: dict, calib: dict, rng) -> dict:
    fail0 = test["fail_osd0"].astype(float)
    failk = test["fail_k1000"].astype(float)
    failf = test["fail_full"].astype(float)
    dh = test["d_h"].astype(float)
    rbp = test["r_bp"].astype(float)
    sw = test["syn_weight"].astype(float)
    conv = test["bp_converged"].astype(int)
    n = int(fail0.size)
    k_free = int(test["k_free"][0])
    k_eff = int(test["K_eff"][0])
    ler0, lo0, hi0 = bootstrap_mean(fail0, rng)
    lerk, lok, hik = bootstrap_mean(failk, rng)
    lerf, lof, hif = bootstrap_mean(failf, rng)
    gap = float(ler0 - lerf)
    headroom_ok = gap > 1.0 / n and int(fail0.sum()) > int(failf.sum())
    auc_dh = bootstrap_auroc(fail0, dh, rng)
    auc_rbp = bootstrap_auroc(fail0, rbp, rng)
    auc_sw = bootstrap_auroc(fail0, sw, rng)
    auc_neg_dh = bootstrap_auroc(fail0, -dh, rng)

    routing = []
    masks = {}
    for f in ESC_TARGETS:
        esc_d = topk_mask(dh, f)
        esc_r = topk_mask(rbp, f)
        esc_s = topk_mask(sw, f)
        masks[f"dh_{f:.2f}"] = esc_d.astype(np.uint8)
        masks[f"rbp_{f:.2f}"] = esc_r.astype(np.uint8)
        masks[f"sw_{f:.2f}"] = esc_s.astype(np.uint8)
        mixed_d = np.where(esc_d, failk, fail0)
        mixed_r = np.where(esc_r, failk, fail0)
        mixed_s = np.where(esc_s, failk, fail0)
        ler_d = float(mixed_d.mean())
        ler_r = float(mixed_r.mean())
        ler_s = float(mixed_s.mean())
        rnd = random_lers(fail0, failk, f, rng, n_rep=N_RANDOM)
        dh_boot = np.empty(N_BOOT)
        rbp_boot = np.empty(N_BOOT)
        sw_boot = np.empty(N_BOOT)
        for b in range(N_BOOT):
            ix = rng.integers(0, n, n)
            dh_boot[b] = mix_ler(fail0[ix], failk[ix], topk_mask(dh[ix], f))
            rbp_boot[b] = mix_ler(fail0[ix], failk[ix], topk_mask(rbp[ix], f))
            sw_boot[b] = mix_ler(fail0[ix], failk[ix], topk_mask(sw[ix], f))
        diff_dr = bootstrap_paired_diff(fail0, failk, dh, rbp, f, rng)
        rec_d = recovered(ler0, lerf, ler_d) if headroom_ok else float("nan")
        rec_r = recovered(ler0, lerf, ler_r) if headroom_ok else float("nan")
        rec_s = recovered(ler0, lerf, ler_s) if headroom_ok else float("nan")
        rec_rnd = recovered(ler0, lerf, float(rnd.mean())) if headroom_ok else float("nan")
        routing.append(
            {
                "f": f,
                "n_esc": int(esc_d.sum()),
                "ler_dh": ler_d,
                "ler_dh_ci": list(boot_ci(dh_boot)),
                "n_fail_dh": int(mixed_d.sum()),
                "ler_rbp": ler_r,
                "ler_rbp_ci": list(boot_ci(rbp_boot)),
                "n_fail_rbp": int(mixed_r.sum()),
                "ler_sw": ler_s,
                "ler_sw_ci": list(boot_ci(sw_boot)),
                "n_fail_sw": int(mixed_s.sum()),
                "ler_rnd": float(rnd.mean()),
                "ler_rnd_ci": list(boot_ci(rnd)),
                "n_fail_rnd_mean": float(rnd.mean() * n),
                "ler_dh_minus_rbp": diff_dr[0],
                "ler_dh_minus_rbp_ci": [diff_dr[1], diff_dr[2]],
                "n_fail_dh_minus_rbp": int(mixed_d.sum() - mixed_r.sum()),
                "rec_dh": rec_d,
                "rec_rbp": rec_r,
                "rec_sw": rec_s,
                "rec_rnd": rec_rnd,
                "routed_dh": transitions(fail0[esc_d], failk[esc_d]),
                "routed_rbp": transitions(fail0[esc_r], failk[esc_r]),
                "routed_sw": transitions(fail0[esc_s], failk[esc_s]),
            }
        )

    thresh = {}
    for name, cscore, tscore in (
        ("d_h", calib["d_h"], dh),
        ("r_bp", calib["r_bp"], rbp),
        ("syn_weight", calib["syn_weight"], sw),
    ):
        tau = tau_for_fraction(cscore, 0.20)
        esc = np.asarray(tscore, dtype=float) >= tau
        mixed = np.where(esc, failk, fail0)
        thresh[name] = {
            "tau": tau,
            "calib_f": float((np.asarray(cscore, dtype=float) >= tau).mean()),
            "test_f": float(esc.mean()),
            "n_esc_test": int(esc.sum()),
            "ler": float(mixed.mean()),
            "n_fail": int(mixed.sum()),
            "tie_rule": "escalate iff score >= tau; tau = quantile(calib, 0.80)",
        }

    bins = []
    edges = np.unique(
        np.concatenate([[-0.5], np.quantile(dh, [0.5, 0.8, 0.9, 0.95]), [dh.max() + 0.5]])
    )
    for i in range(len(edges) - 1):
        m = (dh >= edges[i]) & (dh < edges[i + 1])
        if not m.any():
            continue
        t = transitions(fail0[m], failk[m])
        bins.append(
            {
                "d_h_lo": float(edges[i]),
                "d_h_hi": float(edges[i + 1]),
                "n": int(m.sum()),
                "ler_osd0": float(fail0[m].mean()),
                "rescue_rate": t["WC"] / m.sum(),
                "harm_rate": t["CW"] / m.sum(),
                **t,
            }
        )

    return {
        "p": test["p"],
        "n": n,
        "n_calib": int(calib["fail_osd0"].size),
        "k_free": k_free,
        "K_eff": k_eff,
        "n_fail_osd0": int(fail0.sum()),
        "n_fail_k1000": int(failk.sum()),
        "n_fail_full": int(failf.sum()),
        "ler_osd0": ler0,
        "ler_osd0_ci": [lo0, hi0],
        "ler_k1000": lerk,
        "ler_k1000_ci": [lok, hik],
        "ler_full": lerf,
        "ler_full_ci": [lof, hif],
        "headroom_gap": gap,
        "headroom_meaningful": headroom_ok,
        "bp_conv": float(conv.mean()),
        "bp_syndrome_ok": float(test["bp_syndrome_ok"].mean()),
        "he_osd0_ok": float(test["he_osd0_ok"].mean()) if "he_osd0_ok" in test else float("nan"),
        "auroc_fail_dh": list(auc_dh),
        "auroc_fail_rbp": list(auc_rbp),
        "auroc_fail_sw": list(auc_sw),
        "auroc_fail_neg_dh_posthoc": list(auc_neg_dh),
        "spearman_dh_rbp": spearman(dh, rbp),
        "trans_k1000": transitions(fail0, failk),
        "trans_full": transitions(fail0, failf),
        "routing": routing,
        "threshold_20": thresh,
        "dh_bins": bins,
        "masks": masks,
        "wilson_osd0": list(wilson(int(fail0.sum()), n)),
        "wilson_full": list(wilson(int(failf.sum()), n)),
    }


def make_figure(stats: Dict[float, dict], p_values) -> str:
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
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=False)
    fs = [0.0, 0.10, 0.20, 0.30]
    colors = {"d_h": "#1f4e79", "r_bp": "#6a4c93", "sw": "#e07a3d", "rnd": "#8a8a8a"}
    for ax, p in zip(axes, p_values):
        s = stats[p]
        ler0, lerk, lerf = s["ler_osd0"], s["ler_k1000"], s["ler_full"]
        ci0 = s["ler_osd0_ci"]
        y_d, y_r, y_s, y_n = [ler0], [ler0], [ler0], [ler0]
        e_d, e_r, e_s, e_n = [ci0], [ci0], [ci0], [ci0]
        for row in s["routing"]:
            y_d.append(row["ler_dh"])
            y_r.append(row["ler_rbp"])
            y_s.append(row["ler_sw"])
            y_n.append(row["ler_rnd"])
            e_d.append(row["ler_dh_ci"])
            e_r.append(row["ler_rbp_ci"])
            e_s.append(row["ler_sw_ci"])
            e_n.append(row["ler_rnd_ci"])

        def err(y, ci):
            lo = np.array([c[0] for c in ci], dtype=float)
            hi = np.array([c[1] for c in ci], dtype=float)
            y = np.asarray(y, dtype=float)
            return np.vstack([np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)])

        x = np.array(fs) * 100
        ax.errorbar(x, y_d, yerr=err(y_d, e_d), fmt="-o", color=colors["d_h"], ms=4.5, lw=1.4, capsize=2, label=r"$d_H$")
        ax.errorbar(x, y_r, yerr=err(y_r, e_r), fmt="-s", color=colors["r_bp"], ms=4, lw=1.2, capsize=2, label=r"$r_{\mathrm{BP}}$")
        ax.errorbar(x, y_s, yerr=err(y_s, e_s), fmt="-^", color=colors["sw"], ms=4.5, lw=1.2, capsize=2, label=r"$w_s$")
        ax.errorbar(x, y_n, yerr=err(y_n, e_n), fmt="--o", color=colors["rnd"], ms=3.5, lw=1.0, capsize=2, label="random")
        ax.axhline(ler0, color="#6b6b6b", lw=1.0, ls=":", label="OSD-0")
        ax.axhline(lerk, color="#c44e52", lw=1.0, ls="--", label=r"always $K{=}1000$")
        ax.axhline(lerf, color="#2a9d8f", lw=1.0, ls="-.", label="full 1FV")
        ax.set_title(rf"$p={p:g}$")
        ax.set_xlabel("escalation fraction (%)")
        ax.set_xticks(x)
    axes[0].set_ylabel("logical error rate")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    fig.tight_layout()
    path = os.path.join(OUT, "radial_cross_family_ler_vs_escalation.png")
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def classify(stats, p_values) -> dict:
    predicts_bb = []
    beats_rbp = []
    heads = []
    route_gain = []
    notes = []
    for p in p_values:
        s = stats[p]
        auc_d, dlo, dhi = s["auroc_fail_dh"]
        auc_r, rlo, rhi = s["auroc_fail_rbp"]
        pred = (not np.isnan(auc_d)) and dlo > 0.5
        predicts_bb.append(pred)
        row = next(r for r in s["routing"] if abs(r["f"] - 0.20) < 1e-12)
        beat = pred and row["ler_dh_minus_rbp_ci"][1] < 0
        beats_rbp.append(beat)
        heads.append(bool(s["headroom_meaningful"]))
        gain = row["ler_dh"] < s["ler_osd0"] - 1e-12
        # also beat random/sw if CIs support
        vs_rnd = row["ler_dh_ci"][1] < row["ler_rnd_ci"][0] if not np.isnan(row["ler_rnd"]) else gain
        route_gain.append(gain)
        notes.append(
            f"p={p:g}: AUROC d_H={auc_d:.3f} [{dlo:.3f},{dhi:.3f}] vs r_BP={auc_r:.3f}; "
            f"f=20% LER d_H={row['ler_dh']:.5f} r_BP={row['ler_rbp']:.5f} "
            f"Δ={row['ler_dh_minus_rbp']:.5f} [{row['ler_dh_minus_rbp_ci'][0]:.5f},{row['ler_dh_minus_rbp_ci'][1]:.5f}]; "
            f"headroom={s['headroom_meaningful']} gap={s['headroom_gap']:.5f}; "
            f"R_dH={row['rec_dh']}."
        )
    n_pred, n_beat, n_head, n_gain = map(sum, (predicts_bb, beats_rbp, heads, route_gain))
    n_p = len(p_values)
    if n_pred == n_p and n_beat == n_p and n_head == n_p and n_gain == n_p:
        cat, rec = "STRONG TRANSFER", "YES"
    elif n_pred >= 1 and (n_head >= 1 or n_gain >= 1):
        cat, rec = "PARTIAL TRANSFER", "APPENDIX ONLY"
    else:
        cat, rec = "LIMITED TRANSFER", "NO"
    yesno = lambda xs: "Yes" if all(xs) else ("Mixed" if any(xs) else "No")
    return {
        "predicts": yesno(predicts_bb),
        "same_orientation": yesno(predicts_bb),
        "beats_rbp": yesno(beats_rbp),
        "headroom": yesno(heads),
        "routing_gain": yesno(route_gain),
        "category": cat,
        "inclusion": rec,
        "detail": " ".join(notes),
        "n_pred": n_pred,
        "n_beat": n_beat,
        "n_head": n_head,
    }


def write_report(stats, cfg, fig_path, p_values) -> str:
    cl = classify(stats, p_values)
    lines = [
        "# Formal cross-family validation: radial / lifted-product `radial_90_8`",
        "",
        "Z-memory circuit-level Adaptive V1 pipeline. Pilot seed 21 excluded. Manuscript not edited.",
        "Distance is **unknown**; this is not labelled [[90,8,10]].",
        "",
        "## 1. Frozen configuration and provenance",
        "",
        f"- code: `{cfg['code']}`, (r,s)=({cfg['r']},{cfg['s']}), n={cfg['n']}, k={cfg['k']}, d={cfg['d']}",
        f"- PCM: `{cfg['pcm_path']}` hashes `{json.dumps(cfg['hashes'])}`",
        f"- circuit: `{cfg['circuit']}`",
        f"- p: `{cfg['p']}` (frozen from blind headroom; not retuned)",
        f"- decoder: `{json.dumps(cfg['decoder'])}`",
        f"- software: `{json.dumps(cfg['software'])}`",
        "",
        "## 2. Formal dataset",
        "",
        f"- calib seeds: `{cfg['calib_seeds']}` (5,000 / p)",
        f"- test seeds: `{cfg['test_seeds']}` (20,000 / p)",
        f"- excluded pilot seed: {cfg['pilot_seed_excluded']}",
        f"- bootstrap seed {cfg['rng_bootstrap']}, N_BOOT={cfg['n_boot']}, N_RANDOM={cfg['n_random']}",
        "",
        "## 3. Formal decoder headroom",
        "",
        "| p | OSD-0 | always K=1000 | full 1FV | k_free | K_eff |",
        "|---|---|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        lines.append(
            f"| {p:g} | {fmt_ci(s['ler_osd0'], *s['ler_osd0_ci'])} ({s['n_fail_osd0']}/{s['n']}) | "
            f"{fmt_ci(s['ler_k1000'], *s['ler_k1000_ci'])} ({s['n_fail_k1000']}/{s['n']}) | "
            f"{fmt_ci(s['ler_full'], *s['ler_full_ci'])} ({s['n_fail_full']}/{s['n']}) | "
            f"{s['k_free']} | {s['K_eff']} |"
        )
    lines += [
        "",
        "| p | branch | CC | CW | WC | WW | net |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        for name, t in (("K1000", s["trans_k1000"]), ("full1FV", s["trans_full"])):
            lines.append(
                f"| {p:g} | {name} | {t['CC']} | {t['CW']} | {t['WC']} | {t['WW']} | {t['net']} |"
            )
    lines += [
        "",
        f"H e_OSD0 = s rate: "
        + ", ".join(f"p={p:g} → {100*stats[p]['he_osd0_ok']:.2f}%" for p in p_values)
        + ".",
        "",
        "## 4. Risk discrimination",
        "",
        "Preregistered orientation: larger score = higher predicted OSD-0 failure risk. "
        "`-d_H` AUROC is post hoc only.",
        "",
        "| p | BP conv / He_BP=s | d_H AUROC | r_BP AUROC | w_s AUROC | −d_H AUROC (post hoc) | Spearman(d_H,r_BP) |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        lines.append(
            f"| {p:g} | {100*s['bp_conv']:.2f}% / {100*s['bp_syndrome_ok']:.2f}% | "
            f"{fmt_auc(s['auroc_fail_dh'])} | {fmt_auc(s['auroc_fail_rbp'])} | "
            f"{fmt_auc(s['auroc_fail_sw'])} | {fmt_auc(s['auroc_fail_neg_dh_posthoc'])} | "
            f"{s['spearman_dh_rbp']:.3f} |"
        )
    lines += [
        "",
        "## 5. Exact matched-budget routing",
        "",
        "Primary endpoint: f=20%, escalate to K=1000. Tie-break: stable argsort of −score.",
        "",
        "| p | f | d_H LER | r_BP LER | w_s LER | random LER | R_dH |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        for row in s["routing"]:
            rd = row["rec_dh"]
            rd_s = f"{rd:.3f}" if isinstance(rd, float) and rd == rd else "n/a"
            lines.append(
                f"| {p:g} | {100*row['f']:.0f}% | {row['ler_dh']:.5f} ({row['n_fail_dh']}/{s['n']}) "
                f"{fmt_ci(row['ler_dh'], *row['ler_dh_ci'])} | "
                f"{row['ler_rbp']:.5f} ({row['n_fail_rbp']}/{s['n']}) | "
                f"{row['ler_sw']:.5f} ({row['n_fail_sw']}/{s['n']}) | "
                f"{row['ler_rnd']:.5f} | {rd_s} |"
            )
    lines += [
        "",
        "## 6. d_H versus BP residual",
        "",
        "| p | f | ΔLER (d_H − r_BP) 95% paired CI | Δ fails |",
        "|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        for row in s["routing"]:
            lines.append(
                f"| {p:g} | {100*row['f']:.0f}% | {row['ler_dh_minus_rbp']:.5f} "
                f"[{row['ler_dh_minus_rbp_ci'][0]:.5f},{row['ler_dh_minus_rbp_ci'][1]:.5f}] | "
                f"{row['n_fail_dh_minus_rbp']} |"
            )
    lines += [
        "",
        "## 7. Rescue/harm analysis",
        "",
        "Exact top-20% routed subset (escalated shots only):",
        "",
        "| p | score | n_esc | WC | CW | CC | WW | net |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        row = next(r for r in s["routing"] if abs(r["f"] - 0.20) < 1e-12)
        for name, key in (("d_H", "routed_dh"), ("r_BP", "routed_rbp"), ("w_s", "routed_sw")):
            t = row[key]
            lines.append(
                f"| {p:g} | {name} | {row['n_esc']} | {t['WC']} | {t['CW']} | {t['CC']} | {t['WW']} | {t['net']} |"
            )
    lines += [
        "",
        "Secondary d_H bins (test set):",
        "",
        "| p | bin | n | OSD-0 LER | WC | CW |",
        "|---|---|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        for b in s["dh_bins"]:
            lines.append(
                f"| {p:g} | [{b['d_h_lo']:.1f},{b['d_h_hi']:.1f}) | {b['n']} | {b['ler_osd0']:.5f} | {b['WC']} | {b['CW']} |"
            )
    lines += [
        "",
        "## 8. Held-out calibration",
        "",
        "τ from 5k calib targeting 20%; freeze; apply to 20k test. Not matched-budget if realised f differs.",
        "",
        "| p | signal | τ | calib f | test f | test LER |",
        "|---|---|---|---|---|---|",
    ]
    for p in p_values:
        s = stats[p]
        for name, row in s["threshold_20"].items():
            lines.append(
                f"| {p:g} | {name} | {row['tau']:.6g} | {100*row['calib_f']:.2f}% | "
                f"{100*row['test_f']:.2f}% | {row['ler']:.5f} ({row['n_fail']}/{s['n']}) |"
            )
    lines += [
        "",
        "Tie rule: escalate iff `score >= τ`, `τ = quantile(calib, 0.80)`.",
        "",
        "## 9. Statistical uncertainty",
        "",
        f"LER and AUROC: nonparametric bootstrap, N_BOOT={N_BOOT}, 95% percentile intervals. "
        "Strategy comparisons: paired bootstrap of the same shots. Random routing: 200 seeds. "
        "Raw failure counts are reported with every LER.",
        "",
        "## 10. Reproducibility paths",
        "",
        f"- root: `{OUT}`",
        f"- frozen config: `{CFG_PATH}`",
        f"- figure: `{fig_path}`",
        "- per-shot: `calib/*.npz`, `test/*.npz`",
        "- masks: `routing_masks_p*.npz`",
        "- summaries: `summary.json`, `summary.csv`, `bootstrap_core.json`",
        f"- this report: `{REPORT}`",
        "",
        "## 11. Final scientific interpretation",
        "",
        cl["detail"],
        "",
        "### Explicit answers",
        "",
        f"1. Does d_H predict OSD-0 logical failure on radial_90_8? **{cl['predicts']}**",
        f"2. Is the orientation the same as on BB? **{cl['same_orientation']}**",
        f"3. Does d_H outperform BP residual? **{cl['beats_rbp']}**",
        f"4. Does K=1000 retain useful formal headroom? **{cl['headroom']}**",
        f"5. At exact 20% escalation, does d_H routing lower LER relative to residual, "
        f"syndrome weight and random? **{cl['routing_gain']}** (see tables for residual/w_s/random)",
        "6. What fraction of the full1FV headroom does the 20% policy recover? See R_dH in §5.",
        "7. Does a calibration-derived threshold transfer to the independent test set? See §8 realised f.",
        f"8. Is the result strong enough to support adding this code as a cross-family "
        f"validation in the paper? **{cl['inclusion']}**",
        "",
        f"CROSS-FAMILY RESULT: **{cl['category']}**",
        "",
        f"RECOMMEND MAIN-MANUSCRIPT INCLUSION: **{cl['inclusion']}**",
        "",
    ]
    text = "\n".join(lines) + "\n"
    with open(REPORT, "w") as fh:
        fh.write(text)
    return REPORT


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    with open(CFG_PATH) as fh:
        cfg = json.load(fh)
    p_values = tuple(float(x) for x in cfg["p"])
    test = load_split("test")
    calib = load_split("calib")
    stats = {}
    for p in p_values:
        tp = min(test.keys(), key=lambda x: abs(x - p))
        cp = min(calib.keys(), key=lambda x: abs(x - p))
        print(f"analyse p={p:g} n_test={test[tp]['n']} n_calib={calib[cp]['n']}", flush=True)
        if test[tp]["n"] != 20000 or calib[cp]["n"] != 5000:
            raise SystemExit(
                f"incomplete formal data at p={p}: n_test={test[tp]['n']} n_calib={calib[cp]['n']}"
            )
        stats[p] = analyse_p(test[tp], calib[cp], rng)
        np.savez_compressed(os.path.join(OUT, f"routing_masks_p{p:g}.npz"), **stats[p].pop("masks"))
    fig_path = make_figure(stats, p_values)
    dump = {str(p): jsonable(stats[p]) for p in p_values}
    cl = classify(stats, p_values)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(jsonable({"p": dump, "figure": fig_path, "classify": cl}), fh, indent=2)
        fh.write("\n")
    with open(os.path.join(OUT, "transition_tables.json"), "w") as fh:
        json.dump(
            jsonable(
                {
                    str(p): {
                        "K1000": stats[p]["trans_k1000"],
                        "full1FV": stats[p]["trans_full"],
                        "routed_f20": next(
                            r
                            for r in stats[p]["routing"]
                            if abs(r["f"] - 0.20) < 1e-12
                        ),
                    }
                    for p in p_values
                }
            ),
            fh,
            indent=2,
        )
        fh.write("\n")
    with open(os.path.join(OUT, "paired_comparisons.json"), "w") as fh:
        json.dump(
            jsonable(
                {
                    str(p): [
                        {
                            "f": row["f"],
                            "ler_dh_minus_rbp": row["ler_dh_minus_rbp"],
                            "ler_dh_minus_rbp_ci": row["ler_dh_minus_rbp_ci"],
                            "n_fail_dh_minus_rbp": row["n_fail_dh_minus_rbp"],
                        }
                        for row in stats[p]["routing"]
                    ]
                    for p in p_values
                }
            ),
            fh,
            indent=2,
        )
        fh.write("\n")
    with open(os.path.join(OUT, "calibration_results.json"), "w") as fh:
        json.dump(
            jsonable({str(p): stats[p]["threshold_20"] for p in p_values}),
            fh,
            indent=2,
        )
        fh.write("\n")
    with open(os.path.join(OUT, "auroc_bootstrap.json"), "w") as fh:
        json.dump(
            jsonable(
                {
                    str(p): {
                        "d_h": stats[p]["auroc_fail_dh"],
                        "r_bp": stats[p]["auroc_fail_rbp"],
                        "syn_weight": stats[p]["auroc_fail_sw"],
                        "neg_d_h_posthoc": stats[p]["auroc_fail_neg_dh_posthoc"],
                    }
                    for p in p_values
                }
            ),
            fh,
            indent=2,
        )
        fh.write("\n")
    for p in p_values:
        tp = min(test.keys(), key=lambda x: abs(x - p))
        t = test[tp]
        np.savez_compressed(
            os.path.join(OUT, f"per_shot_test_p{p:g}.npz"),
            d_h=t["d_h"],
            r_bp=t["r_bp"],
            syn_weight=t["syn_weight"],
            bp_converged=t["bp_converged"],
            bp_syndrome_ok=t["bp_syndrome_ok"],
            fail_osd0=t["fail_osd0"],
            fail_k1000=t["fail_k1000"],
            fail_full=t["fail_full"],
        )
    with open(os.path.join(OUT, "bootstrap_core.json"), "w") as fh:
        json.dump(
            jsonable(
                {
                    str(p): {
                        "auroc_fail_dh": stats[p]["auroc_fail_dh"],
                        "auroc_fail_rbp": stats[p]["auroc_fail_rbp"],
                        "auroc_fail_sw": stats[p]["auroc_fail_sw"],
                        "ler_osd0": stats[p]["ler_osd0"],
                        "ler_osd0_ci": stats[p]["ler_osd0_ci"],
                        "ler_full": stats[p]["ler_full"],
                        "ler_full_ci": stats[p]["ler_full_ci"],
                        "routing": stats[p]["routing"],
                        "threshold_20": stats[p]["threshold_20"],
                    }
                    for p in p_values
                }
            ),
            fh,
            indent=2,
        )
        fh.write("\n")
    with open(os.path.join(OUT, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["p", "f", "ler_osd0", "ler_k1000", "ler_full", "ler_dh", "ler_rbp", "ler_sw", "ler_rnd", "n_fail_dh", "n_fail_rbp"])
        for p in p_values:
            s = stats[p]
            for row in s["routing"]:
                w.writerow([p, row["f"], s["ler_osd0"], s["ler_k1000"], s["ler_full"], row["ler_dh"], row["ler_rbp"], row["ler_sw"], row["ler_rnd"], row["n_fail_dh"], row["n_fail_rbp"]])
    report = write_report(stats, cfg, fig_path, p_values)
    print("figure", fig_path)
    print("report", report)
    cl = classify(stats, p_values)
    print("CROSS-FAMILY RESULT:", cl["category"])
    print("RECOMMEND MAIN-MANUSCRIPT INCLUSION:", cl["inclusion"])
    for p in p_values:
        s = stats[p]
        row = next(r for r in s["routing"] if abs(r["f"] - 0.20) < 1e-12)
        print(
            f"p={p:g} osd0={s['ler_osd0']:.5f} k1000={s['ler_k1000']:.5f} full={s['ler_full']:.5f} "
            f"AUROC dH={s['auroc_fail_dh'][0]:.3f} rBP={s['auroc_fail_rbp'][0]:.3f} "
            f"f20 dH={row['ler_dh']:.5f} rBP={row['ler_rbp']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
