"""Main-text figure for radial_90_8 cross-family matched-budget routing.

Reads frozen formal outputs only. Does not re-decode, re-sample, or edit
the manuscript / BB Adaptive V1 results / headroom-pilot directory.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

from failure_prediction.final_adaptive import C_DH, C_FULL, C_OSD0, C_RND, C_SW, _style

SRC = "outputs/failure_prediction/nonbb_radial_formal/summary.json"
OUT_DIR = "outputs/paper_figures/final_adaptive"
STEM = "fig_radial_cross_family"
C_RBP = "#6a4c93"
C_K1000 = "#c44e52"
N_SHOTS = 20000
P_VALUES = (0.008, 0.009)


def _ci(lo, hi, y):
    return np.vstack([max(0.0, y - lo), max(0.0, hi - y)])


def recovered(ler0, ler_full, ler) -> float:
    gap = ler0 - ler_full
    if gap <= 0:
        return float("nan")
    return (ler0 - ler) / gap


def load_stats() -> dict:
    with open(SRC) as fh:
        blob = json.load(fh)
    return {float(k): v for k, v in blob["p"].items()}


def tidy_rows(stats: dict) -> list:
    rows = []
    methods = (
        ("d_h", "ler_dh", "ler_dh_ci", "n_fail_dh", "rec_dh"),
        ("r_bp", "ler_rbp", "ler_rbp_ci", "n_fail_rbp", "rec_rbp"),
        ("syn_weight", "ler_sw", "ler_sw_ci", "n_fail_sw", "rec_sw"),
        ("random", "ler_rnd", "ler_rnd_ci", "n_fail_rnd_mean", "rec_rnd"),
    )
    for p in P_VALUES:
        s = stats[p]
        ler0, lerk, lerf = s["ler_osd0"], s["ler_k1000"], s["ler_full"]
        by_f = {round(r["f"], 2): r for r in s["routing"]}
        for method, _, _, _, _ in methods:
            rows.append(
                {
                    "p": p,
                    "f_esc": 0.0,
                    "method": method,
                    "ler": ler0,
                    "ler_lo": s["ler_osd0_ci"][0],
                    "ler_hi": s["ler_osd0_ci"][1],
                    "n_fail": s["n_fail_osd0"],
                    "n_shots": N_SHOTS,
                    "n_esc": 0,
                    "R": 0.0,
                    "note": "f=0 is OSD-0 for every router",
                }
            )
        for f in (0.10, 0.20, 0.30):
            r = by_f[f]
            for method, yk, cik, nk, rk in methods:
                rows.append(
                    {
                        "p": p,
                        "f_esc": f,
                        "method": method,
                        "ler": r[yk],
                        "ler_lo": r[cik][0],
                        "ler_hi": r[cik][1],
                        "n_fail": r[nk],
                        "n_shots": N_SHOTS,
                        "n_esc": r["n_esc"],
                        "R": r[rk],
                        "note": "exact matched-budget top-k to K=1000",
                    }
                )
        for method, _, _, _, _ in methods:
            rows.append(
                {
                    "p": p,
                    "f_esc": 1.0,
                    "method": method,
                    "ler": lerk,
                    "ler_lo": s["ler_k1000_ci"][0],
                    "ler_hi": s["ler_k1000_ci"][1],
                    "n_fail": s["n_fail_k1000"],
                    "n_shots": N_SHOTS,
                    "n_esc": N_SHOTS,
                    "R": recovered(ler0, lerf, lerk),
                    "note": "f=1 is always K=1000 for every router",
                }
            )
        for method, ler, ci, nfail, note in (
            ("osd0_ref", ler0, s["ler_osd0_ci"], s["n_fail_osd0"], "horizontal OSD-0 reference"),
            ("k1000_ref", lerk, s["ler_k1000_ci"], s["n_fail_k1000"], "horizontal always-K=1000 reference"),
            ("full1fv_ref", lerf, s["ler_full_ci"], s["n_fail_full"], "horizontal full-1FV reference; 1 shot from K=1000"),
        ):
            rows.append(
                {
                    "p": p,
                    "f_esc": "",
                    "method": method,
                    "ler": ler,
                    "ler_lo": ci[0],
                    "ler_hi": ci[1],
                    "n_fail": nfail,
                    "n_shots": N_SHOTS,
                    "n_esc": "",
                    "R": "",
                    "note": note,
                }
            )
    return rows


def write_csv(rows: list) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, STEM + ".csv")
    fields = [
        "p",
        "f_esc",
        "method",
        "ler",
        "ler_lo",
        "ler_hi",
        "n_fail",
        "n_shots",
        "n_esc",
        "R",
        "note",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    return path


def make_figure(stats: dict) -> tuple[str, str]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    _style()
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "legend.fontsize": 7.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 3.05), sharey=False)
    xs_mid = np.array([0.0, 0.10, 0.20, 0.30, 1.0])
    series = (
        ("d_h", "ler_dh", "ler_dh_ci", C_DH, "o", "-", 1.85, 5.2, r"$d_H$"),
        ("r_bp", "ler_rbp", "ler_rbp_ci", C_RBP, "s", "-", 1.35, 4.2, r"$r_{\mathrm{BP}}$"),
        ("syn_weight", "ler_sw", "ler_sw_ci", C_SW, "^", "-", 1.25, 4.6, r"$w_s$"),
        ("random", "ler_rnd", "ler_rnd_ci", C_RND, "o", "--", 1.05, 3.6, "random"),
    )
    for ax, p, lab in zip(axes, P_VALUES, ("a", "b")):
        s = stats[p]
        ler0, lerk, lerf = s["ler_osd0"], s["ler_k1000"], s["ler_full"]
        by_f = {round(r["f"], 2): r for r in s["routing"]}
        ax.axvline(0.20, color="#cfcfcf", lw=0.7, zorder=0)
        ax.axhline(ler0, color=C_OSD0, ls=":", lw=1.05, zorder=1)
        ax.axhline(lerf, color=C_FULL, ls="-.", lw=1.15, alpha=0.70, zorder=1)
        ax.axhline(lerk, color=C_K1000, ls="--", lw=1.0, zorder=2)
        for _name, yk, cik, color, mk, ls, lw, ms, _lab in series:
            ys = [ler0]
            los = [s["ler_osd0_ci"][0]]
            his = [s["ler_osd0_ci"][1]]
            for f in (0.10, 0.20, 0.30):
                r = by_f[f]
                ys.append(r[yk])
                los.append(r[cik][0])
                his.append(r[cik][1])
            ys.append(lerk)
            los.append(s["ler_k1000_ci"][0])
            his.append(s["ler_k1000_ci"][1])
            ys = np.asarray(ys, dtype=float)
            los = np.asarray(los, dtype=float)
            his = np.asarray(his, dtype=float)
            ax.plot(xs_mid, ys, color=color, ls=ls, lw=lw, zorder=3)
            if _name == "random":
                ax.fill_between(
                    xs_mid[1:4], los[1:4], his[1:4], color=C_RND, alpha=0.18, linewidth=0, zorder=2
                )
                ax.errorbar(
                    xs_mid[1:4],
                    ys[1:4],
                    yerr=np.vstack([ys[1:4] - los[1:4], his[1:4] - ys[1:4]]),
                    fmt="o",
                    color=C_RND,
                    ms=ms,
                    ls="none",
                    capsize=1.6,
                    elinewidth=0.7,
                    zorder=4,
                )
            else:
                ax.plot(
                    xs_mid[1:4],
                    ys[1:4],
                    marker=mk,
                    color=color,
                    ms=ms,
                    ls="none",
                    zorder=5,
                    markerfacecolor=color,
                    markeredgecolor=color,
                    markeredgewidth=0.6,
                )
                if _name == "d_h":
                    ax.plot(
                        [0.20],
                        [by_f[0.20]["ler_dh"]],
                        marker="o",
                        color=C_DH,
                        ms=7.0,
                        zorder=6,
                        markerfacecolor=C_DH,
                        markeredgecolor="white",
                        markeredgewidth=0.7,
                    )
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(0.0, ler0 * 1.16)
        ax.set_xticks(xs_mid)
        ax.set_xticklabels(["0", "0.1", "0.2", "0.3", "1"])
        ax.set_xlabel(r"escalation fraction  $f_{\mathrm{esc}}$")
        ax.set_title(rf"$p={p:g}$", pad=3)
        ax.text(
            -0.14,
            1.06,
            rf"({lab})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
    axes[0].set_ylabel("logical error rate")
    handles = [
        Line2D([0], [0], color=C_DH, lw=1.85, marker="o", ms=4.5, label=r"$d_H$"),
        Line2D([0], [0], color=C_RBP, lw=1.35, marker="s", ms=4.0, label=r"$r_{\mathrm{BP}}$"),
        Line2D([0], [0], color=C_SW, lw=1.25, marker="^", ms=4.5, label=r"$w_s$"),
        Line2D([0], [0], color=C_RND, lw=1.05, ls="--", marker="o", ms=3.5, label="random"),
        Line2D([0], [0], color=C_OSD0, lw=1.05, ls=":", label="OSD-0"),
        Line2D([0], [0], color=C_K1000, lw=1.0, ls="--", label=r"always $K{=}1000$"),
        Line2D([0], [0], color=C_FULL, lw=1.15, ls="-.", alpha=0.85, label="full 1FV"),
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=7,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
        columnspacing=1.05,
        handletextpad=0.4,
        handlelength=2.15,
    )
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.90), w_pad=1.6)
    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, STEM + ".png")
    pdf = os.path.join(OUT_DIR, STEM + ".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def write_note(png: str, pdf: str, csv_path: str) -> str:
    path = os.path.join(OUT_DIR, STEM + "_note.md")
    text = """# Radial cross-family main-text figure

Source: frozen formal experiment `outputs/failure_prediction/nonbb_radial_formal/`.
No shots were regenerated. No decoder, budget, or routing policy was changed.
Manuscript not edited.

## Files

- PNG: `outputs/paper_figures/final_adaptive/fig_radial_cross_family.png`
- PDF: `outputs/paper_figures/final_adaptive/fig_radial_cross_family.pdf`
- tidy plotting CSV: `outputs/paper_figures/final_adaptive/fig_radial_cross_family.csv`

## What the figure shows

Primary comparison: **exact matched-budget routing** (stable `argsort` of $-$score; escalate exactly `round(f n)` shots to $K=1000$). The primary endpoint is **$f_{\\mathrm{esc}}=20\\%$** (vertical guide). Code: radial / lifted-product `radial_90_8`, $n=90$, $k=8$, **distance unknown**. $K=1000$ is a real truncation ($k_{\\mathrm{free}}=3289$). Full 1FV differs from always $K=1000$ by **one shot** at each $p$ (418 vs 419 failures at $p=0.008$; 931 vs 932 at $p=0.009$); the two reference lines therefore overlap on the plot and must not be read as a visual gap.

Preregistered scores, larger = higher predicted OSD-0 risk: $d_H$ AUROC $0.951$ / $0.940$; residual AUROC $0.832$ / $0.809$. Exact top-20% LER: $p=0.008$: $d_H$ $0.02275$, residual $0.03290$, syndrome $0.03860$, random $0.04525$; $p=0.009$: $d_H$ $0.05390$, residual $0.07275$, syndrome $0.07975$, random $0.09054$. Recovered headroom $R(d_H)$ at 20% $=0.939$ and $0.866$. Random bands are 95% intervals over 200 seeds; other CIs are in the CSV.

## Caption draft

Cross-family matched-budget routing on the radial / lifted-product code `radial_90_8` ($n=90$, $k=8$; distance unknown). Each panel is an independent 20,000-shot test set at circuit-level Z-memory noise $p=0.008$ (a) and $p=0.009$ (b). Shots are ranked by a score computed from the OSD-0 fast path only, then the top $f_{\\mathrm{esc}}$ are escalated to truncated one-free-variable search with $K=1000$; $f=0$ is OSD-0 and $f=1$ is always $K=1000$. $K=1000$ is a genuine truncation ($k_{\\mathrm{free}}=3289$). Decoder disagreement $d_H$ outperforms BP residual $r_{\\mathrm{BP}}$, syndrome weight $w_s$, and random routing at the primary 20% budget, recovering $93.9\\%$ and $86.6\\%$ of the OSD-0 to full-1FV gap. Full 1FV (dash-dotted) differs from always $K=1000$ (dashed) by one shot at each $p$ and is visually coincident. Random bands are 95% intervals over 200 seeds. AUROC for OSD-0 failure: $d_H$ $0.951$ / $0.940$, residual $0.832$ / $0.809$.

## Is this enough for a main-text figure?

Yes. The two-panel LER-versus-budget plot is the same claim geometry as the BB Adaptive V1 routing figure, on a structurally distinct code, with the extra residual control the cross-family question requires. It is compact enough for a two-column layout. Suggested insertion: immediately after the BB matched-budget / recovered-headroom discussion, as a short cross-family validation paragraph plus this figure — not as a replacement for the BB panels. Keep the Hamming-HGP negative control out of the main text. Do not add this figure to the manuscript in this step.
"""
    with open(path, "w") as fh:
        fh.write(text)
    return path


def main() -> None:
    stats = load_stats()
    csv_path = write_csv(tidy_rows(stats))
    png, pdf = make_figure(stats)
    note = write_note(png, pdf, csv_path)
    print("png", png)
    print("pdf", pdf)
    print("csv", csv_path)
    print("note", note)


if __name__ == "__main__":
    main()
