"""Publication PNG figures for Adaptive Decoder V1.

Writes only final PNG files under outputs/paper_figures/adaptive_v1/.
No SVG/PDF, no intermediate dashboards.
"""

from __future__ import annotations

import os

import numpy as np

from failure_prediction.adaptive_v1_analysis import (
    ESC_TARGETS,
    P_VALUES,
    apply_tau,
    auroc,
    curve_for,
    load_dir,
    tau_for_escalation,
)

OUT_DIR = "outputs/paper_figures/adaptive_v1"
DATA_DIR = "outputs/failure_prediction/adaptive_v1"

# Same-node serial pass (wolpy01, p=0.007, n=200). Adaptive cost is a linear
# mix of these measured times, not a candidate-count estimate.
SERIAL_MS = {
    "osd0": 22.413830715231597,
    "k500": 248.74730209695153,
    "k1000": 258.8906915554027,
    "full": 450.519605204463,
}

# decoder_survey OSD-0 / OSD-CS LER (bb72 n=50k; reused as a headroom control).
BB72_GAP = {
    0.006: (0.043100, 0.036440),
    0.008: (0.133700, 0.109120),
}

C_OSD0 = "#6b6b6b"
C_K500 = "#9bb7d4"
C_K1000 = "#1f4e79"
C_FULL = "#2a9d8f"
C_FAIL = "#c44e52"
C_OK = "#8a8a8a"


def _style():
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save(fig, name: str) -> str:
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def adapt_serial_ms(f_esc: float, branch: str = "k1000") -> float:
    t0 = SERIAL_MS["osd0"]
    return t0 + float(f_esc) * (SERIAL_MS[branch] - t0)


def recovery(ler0: float, ler_full: float, ler: float) -> float:
    gap = ler0 - ler_full
    if gap <= 0:
        return float("nan")
    return (ler0 - ler) / gap


def fig1_schematic() -> str:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

    fig, ax = plt.subplots(figsize=(6.6, 8.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#222", lw=1.2, fs=10, weight="normal"):
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
        )
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight, color="#111")
        return p

    def arrow(x1, y1, x2, y2, color="#333"):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color=color,
                shrinkA=0,
                shrinkB=0,
            )
        )

    # Always-executed band: BP, OSD-0, and the free d_H statistic.
    ax.add_patch(
        Rectangle((1.15, 6.05), 7.7, 7.05, facecolor="#eef3f8", edgecolor="none", zorder=0)
    )
    ax.text(1.35, 12.85, "always executed", fontsize=8.5, color="#3d5a73", fontstyle="italic")

    box(5, 12.35, 3.4, 0.85, "syndrome  $s$", "#ffffff", fs=11)
    arrow(5, 11.92, 5, 11.52)
    box(5, 11.1, 3.4, 0.85, "BP@20", "#d9e6f2", fs=11, weight="bold")
    arrow(5, 10.67, 5, 10.27)
    box(5, 9.85, 3.4, 0.85, "OSD-0", "#d9e6f2", fs=11, weight="bold")

    arrow(5, 9.42, 2.7, 8.55)
    arrow(5, 9.42, 7.3, 8.55)
    box(2.7, 8.15, 3.0, 0.75, r"$\hat e_{\mathrm{BP}}$", "#ffffff", fs=10)
    box(7.3, 8.15, 3.0, 0.75, r"$\hat e_{\mathrm{OSD0}}$", "#ffffff", fs=10)

    arrow(2.7, 7.77, 5, 6.95)
    arrow(7.3, 7.77, 5, 6.95)
    box(
        5,
        6.55,
        5.6,
        0.9,
        r"$d_H(\hat e_{\mathrm{BP}},\hat e_{\mathrm{OSD0}})$" + "\ncheap online router",
        "#fff4d6",
        fs=10,
        weight="bold",
    )

    arrow(5, 6.1, 5, 5.55)
    box(5, 5.2, 3.6, 0.7, r"threshold  $\tau$", "#ffffff", fs=11)

    arrow(5, 4.85, 2.55, 4.15)
    arrow(5, 4.85, 7.45, 4.15)

    ax.text(2.55, 4.28, "low-risk", ha="center", fontsize=8.5, color="#3a6b3a")
    ax.text(7.45, 4.28, "high-risk  (minority)", ha="center", fontsize=8.5, color="#7a3030")

    box(2.55, 3.45, 3.5, 1.05, "return OSD-0", "#e5f4e5", fs=10.5, weight="bold")
    box(
        7.45,
        3.45,
        3.9,
        1.05,
        "truncated 1FV search\n$K=1000$  (primary)",
        "#f8e0e0",
        fs=10,
        weight="bold",
    )
    arrow(7.45, 2.92, 7.45, 2.35)
    box(7.45, 1.95, 3.5, 0.8, "correction", "#f8e0e0", fs=10.5)

    ax.text(
        5,
        0.55,
        "Adaptive Decoder V1  ·  only high-$d_H$ shots pay for extra OSD search",
        ha="center",
        fontsize=9,
        color="#333",
    )
    return _save(fig, "fig1_adaptive_v1_schematic.png")


def fig2_tradeoff(by_p: dict) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55), sharey=False)
    for ax, p in zip(axes, P_VALUES):
        d = by_p[p]
        ler0 = float(d["fail_osd0"].mean())
        ler_full = float(d["fail_full"].mean())
        c500 = curve_for(d, "fail_k500")
        c1000 = curve_for(d, "fail_k1000")
        ax.axhline(ler0, color=C_OSD0, ls=":", lw=1.0, label="OSD-0")
        ax.axhline(ler_full, color=C_FULL, ls="--", lw=1.0, label="full 1FV")
        ax.plot(
            [r["f_esc"] for r in c500],
            [r["ler"] for r in c500],
            color=C_K500,
            lw=1.5,
            label="Adaptive $K{=}500$",
        )
        ax.plot(
            [r["f_esc"] for r in c1000],
            [r["ler"] for r in c1000],
            color=C_K1000,
            lw=2.3,
            label="Adaptive $K{=}1000$",
        )
        ax.scatter([0.0], [ler0], marker="s", color=C_OSD0, s=28, zorder=4)
        ax.scatter([1.0], [ler_full], marker="D", color=C_FULL, s=28, zorder=4)
        for f in ESC_TARGETS:
            tau = tau_for_escalation(d, f)
            r = apply_tau(d, tau, "fail_k1000")
            rec = recovery(ler0, ler_full, r["ler"])
            ax.errorbar(
                r["f_esc"],
                r["ler"],
                yerr=[[r["ler"] - r["ci_lo"]], [r["ci_hi"] - r["ler"]]],
                fmt="o",
                color=C_K1000,
                ms=6.5,
                zorder=5,
                capsize=2,
                elinewidth=0.8,
            )
            if f in (0.20, 0.30):
                ax.annotate(
                    f"{100 * rec:.0f}%",
                    (r["f_esc"], r["ler"]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    color=C_K1000,
                )
        for f, ls in ((0.10, ":"), (0.20, "--"), (0.30, ":")):
            ax.axvline(f, color="#cccccc", lw=0.7, ls=ls, zorder=0)
        ax.set_xlim(-0.03, 1.05)
        ymax = max(ler0 * 1.18, 1.05 * max(ler0, 1e-4))
        ax.set_ylim(0.0, ymax)
        ax.set_title(rf"$p={p:g}$")
        ax.set_xlabel(r"escalation fraction  $f_{\mathrm{esc}}$")
        ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    axes[0].set_ylabel("logical error rate")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig2_ler_vs_escalation.png")


def fig3_pareto(by_p: dict) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.55), sharey=False)
    for ax, p in zip(axes, P_VALUES):
        d = by_p[p]
        ler0 = float(d["fail_osd0"].mean())
        ler_full = float(d["fail_full"].mean())
        ler_k500 = float(d["fail_k500"].mean())
        ler_k1000 = float(d["fail_k1000"].mean())
        c1000 = curve_for(d, "fail_k1000")
        xs = [adapt_serial_ms(r["f_esc"], "k1000") for r in c1000]
        ys = [r["ler"] for r in c1000]
        ax.plot(xs, ys, color=C_K1000, lw=2.0, label="Adaptive $K{=}1000$")
        ax.scatter(
            [SERIAL_MS["osd0"]], [ler0], marker="s", color=C_OSD0, s=42, zorder=5, label="OSD-0"
        )
        ax.scatter(
            [SERIAL_MS["k500"]],
            [ler_k500],
            marker="^",
            color=C_K500,
            s=42,
            zorder=5,
            label="always $K{=}500$",
        )
        ax.scatter(
            [SERIAL_MS["k1000"]],
            [ler_k1000],
            marker="o",
            color=C_K1000,
            s=42,
            zorder=5,
            label="always $K{=}1000$",
        )
        ax.scatter(
            [SERIAL_MS["full"]],
            [ler_full],
            marker="D",
            color=C_FULL,
            s=42,
            zorder=5,
            label="full 1FV",
        )
        for f in ESC_TARGETS:
            tau = tau_for_escalation(d, f)
            r = apply_tau(d, tau, "fail_k1000")
            t = adapt_serial_ms(r["f_esc"], "k1000")
            ax.scatter([t], [r["ler"]], marker="o", color=C_K1000, s=28, zorder=6, edgecolors="white", linewidths=0.6)
            ax.annotate(
                f"{100 * f:.0f}%",
                (t, r["ler"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=C_K1000,
            )
        ax.set_title(rf"$p={p:g}$")
        ax.set_xlabel("serial time / shot  (ms)")
        ax.set_xlim(0, 500)
        ax.set_ylim(0.0, ler0 * 1.18)
    axes[0].set_ylabel("logical error rate")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig3_ler_vs_runtime.png")


def fig4_router(by_p: dict) -> str:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), sharey=True)
    aucs = {
        0.006: (0.9581, 0.9646),
        0.007: (0.9582, 0.9572),
        0.008: (0.9367, 0.9309),
    }
    for ax, p in zip(axes, P_VALUES):
        d = by_p[p]
        dh = d["d_h"]
        fail = d["fail_osd0"].astype(bool)
        xmax = float(np.quantile(dh[fail], 0.98)) if fail.any() else float(dh.max())
        bins = np.linspace(0, max(xmax, 8), 36)
        ax.hist(
            dh[~fail],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color=C_OK,
            label="OSD-0 correct",
        )
        ax.hist(
            dh[fail],
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.45,
            color=C_FAIL,
            label="OSD-0 fail",
        )
        ax.hist(dh[~fail], bins=bins, density=True, histtype="step", color=C_OK, lw=1.1)
        ax.hist(dh[fail], bins=bins, density=True, histtype="step", color=C_FAIL, lw=1.3)
        a_fail, a_ben = aucs[p]
        # Recompute so the panel is not a hardcoded-only sticker.
        a_fail_m = auroc(d["fail_osd0"], dh)
        a_ben_m = auroc((d["fail_osd0"] == 1) & (d["fail_full"] == 0), dh)
        ax.set_title(rf"$p={p:g}$")
        ax.set_xlabel(r"$d_H(\hat e_{\mathrm{BP}},\hat e_{\mathrm{OSD0}})$")
        ax.text(
            0.97,
            0.95,
            f"fail AUROC {a_fail_m:.3f}\nbeneficial {a_ben_m:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", lw=0.6),
        )
        _ = (a_fail, a_ben)
    axes[0].set_ylabel("density")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, "fig4_dh_router.png")


def fig5_headroom_control(by_p: dict) -> str:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ps_72 = [0.006, 0.008]
    gap_72 = [BB72_GAP[p][0] / BB72_GAP[p][1] for p in ps_72]
    ps_144 = list(P_VALUES)
    gap_144 = [
        float(by_p[p]["fail_osd0"].mean() / by_p[p]["fail_full"].mean()) for p in ps_144
    ]
    ax.plot(
        ps_72,
        gap_72,
        "s--",
        color="#a0a0a0",
        lw=1.4,
        ms=7,
        label=r"bb72  OSD-0 / OSD-CS  (survey)",
    )
    ax.plot(
        ps_144,
        gap_144,
        "o-",
        color=C_K1000,
        lw=2.0,
        ms=8,
        label=r"bb144  OSD-0 / full 1FV  (Adaptive V1)",
    )
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":")
    ax.set_xlabel(r"physical error rate  $p$")
    ax.set_ylabel("fast / strong  LER ratio")
    ax.set_xticks([0.006, 0.007, 0.008])
    ax.set_ylim(0.8, 5.6)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Fast vs strong LER headroom (control, not a bb72 Adaptive V1 run)")
    fig.tight_layout()
    return _save(fig, "fig5_bb72_headroom_control.png")


def main() -> None:
    _style()
    import matplotlib.pyplot as plt

    by_p = load_dir(os.path.join(DATA_DIR, "parts"))
    paths = [
        fig1_schematic(),
        fig2_tradeoff(by_p),
        fig3_pareto(by_p),
        fig4_router(by_p),
        fig5_headroom_control(by_p),
    ]
    plt.close("all")
    print("Wrote:")
    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
