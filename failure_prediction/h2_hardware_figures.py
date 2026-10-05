"""Standalone H2 full-CSS hardware figure in the Adaptive V1 paper style.

Does not submit hardware. Reads decoded_shots.npz + hardware_analysis.json.
"""

from __future__ import annotations

import json
import os

import numpy as np

from failure_prediction.adaptive_v1_analysis import wilson
from failure_prediction.final_adaptive import (
    C_DH,
    C_FAIL,
    C_FULL,
    C_OK,
    C_OSD0,
    C_RND,
    OUT_DIR,
    _save,
    _style,
)

HW_DIR = "outputs/failure_prediction/h2_full_css_hardware"


def _roc(y: np.ndarray, s: np.ndarray):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    p = max(int(ys.sum()), 1)
    n = max(int(ys.size - ys.sum()), 1)
    tps = np.cumsum(ys)
    fps = np.cumsum(1 - ys)
    fpr = np.concatenate([[0.0], fps / n, [1.0]])
    tpr = np.concatenate([[0.0], tps / p, [1.0]])
    return fpr, tpr


def _panel_a(ax) -> None:
    from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.4)
    ax.axis("off")

    ax.add_patch(Rectangle((0.25, 0.25), 9.5, 9.7, facecolor="#eef3f8", edgecolor="none", zorder=0))
    ax.text(0.45, 9.35, r"bb$_{12,2}$  ·  22 qubits  ·  12 rounds", fontsize=8, color="#3d5a73", fontstyle="italic")

    def qubit_row(y, n, fc, ec, label, x0=0.7, gap=0.62):
        for i in range(n):
            ax.add_patch(Circle((x0 + i * gap, y), 0.26, facecolor=fc, edgecolor=ec, lw=0.95, zorder=2))
        ax.text(x0 + n * gap + 0.15, y, label, ha="left", va="center", fontsize=8.5, color="#222")

    qubit_row(8.2, 12, "#fff", C_DH, "12 data", x0=0.55, gap=0.52)
    qubit_row(7.05, 5, "#d9e6f2", C_DH, "5 X-check ancillas", x0=0.55, gap=0.62)
    qubit_row(5.9, 5, "#dceee8", C_FULL, "5 Z-check ancillas", x0=0.55, gap=0.62)

    def box(x, y, w, h, text, fc, fs=9.5, weight="normal"):
        ax.add_patch(
            FancyBboxPatch(
                (x - w / 2, y - h / 2),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                facecolor=fc,
                edgecolor="#222",
                linewidth=1.1,
                zorder=2,
            )
        )
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight=weight, zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                linewidth=1.1, color="#333", shrinkA=1, shrinkB=1, zorder=2,
            )
        )

    arrow(5.0, 5.45, 5.0, 4.9)
    box(5.0, 4.35, 7.6, 0.9, r"$|00\rangle_L$  via stabilizer prep + Pauli frame", "#fff", 8.8)
    arrow(5.0, 3.88, 5.0, 3.38)
    box(5.0, 2.85, 7.6, 0.95, r"12 rounds: extract $H_X$ and $H_Z$", "#d9e6f2", 9.5, "bold")
    arrow(5.0, 2.35, 5.0, 1.85)
    box(5.0, 1.25, 7.6, 0.9, "final data $Z$ readout  →  logical $Z$", "#fff4d6", 9.2, "bold")


def _panel_b(ax, dh, fail, auc, ci) -> None:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fail = np.asarray(fail, dtype=bool)
    xmax = float(np.quantile(dh[fail], 0.90)) if fail.any() else 8.0
    xmax = max(12.0, min(xmax, 16.0))
    bins = np.arange(0, xmax + 2) - 0.5
    ax.hist(dh[~fail], bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=C_OK, label="OSD-0 correct")
    ax.hist(dh[fail], bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=C_FAIL, label="OSD-0 fail")
    ax.hist(dh[~fail], bins=bins, density=True, histtype="step", color=C_OK, lw=1.1)
    ax.hist(dh[fail], bins=bins, density=True, histtype="step", color=C_FAIL, lw=1.3)
    ax.set_xlim(-0.6, xmax + 0.6)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"$d_H(\hat e_{\mathrm{BP}},\hat e_{\mathrm{OSD0}})$")
    ax.set_ylabel("density")
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fpr, tpr = _roc(fail.astype(int), dh)
    ins = inset_axes(ax, width="38%", height="40%", loc="upper right", borderpad=0.7)
    ins.plot(fpr, tpr, color=C_DH, lw=1.7)
    ins.plot([0, 1], [0, 1], color=C_RND, ls=":", lw=0.9)
    ins.set_xlim(0, 1)
    ins.set_ylim(0, 1)
    ins.set_xticks([0, 1])
    ins.set_yticks([0, 1])
    ins.tick_params(labelsize=6.5)
    ins.set_xlabel("FPR", fontsize=7, labelpad=0)
    ins.set_ylabel("TPR", fontsize=7, labelpad=0)
    for s in ins.spines.values():
        s.set_linewidth(0.7)
    ins.set_title(rf"AUROC ${auc:.3f}$" + "\n" + rf"$[{ci[0]:.3f},{ci[1]:.3f}]$", fontsize=7, color=C_DH, pad=2)


def _panel_c(ax, hw: dict) -> None:
    stim_n, stim_k = 4000, 1062
    stim_ler = stim_k / stim_n
    stim_ler_ci = wilson(stim_k, stim_n)
    stim_auc = float(hw["stim_preflight"]["auroc_fail"])
    hw_ler = float(hw["ler_osd0"])
    hw_ler_ci = hw["ler_osd0_wilson95"]
    hw_auc = float(hw["auroc_fail"])
    hw_auc_ci = hw["auroc_fail_ci95"]

    # two clusters: LER and AUROC; Stim = grey square, H2 = navy circle
    x_ler = np.array([0.0, 0.55])
    x_auc = np.array([2.05, 2.60])
    ax.errorbar(
        [x_ler[0]], [stim_ler],
        yerr=[[stim_ler - stim_ler_ci[0]], [stim_ler_ci[1] - stim_ler]],
        fmt="s", color=C_OSD0, ms=8, capsize=3, elinewidth=1.0, zorder=5, label="Stim preflight",
    )
    ax.errorbar(
        [x_ler[1]], [hw_ler],
        yerr=[[hw_ler - hw_ler_ci[0]], [hw_ler_ci[1] - hw_ler]],
        fmt="o", color=C_DH, ms=8.5, capsize=3, elinewidth=1.0, zorder=5, label="H2-1 hardware",
    )
    ax.plot([x_auc[0]], [stim_auc], "s", color=C_OSD0, ms=8, zorder=5)
    ax.errorbar(
        [x_auc[1]], [hw_auc],
        yerr=[[hw_auc - hw_auc_ci[0]], [hw_auc_ci[1] - hw_auc]],
        fmt="o", color=C_DH, ms=8.5, capsize=3, elinewidth=1.0, zorder=5,
    )
    ax.set_xlim(-0.45, 3.15)
    ax.set_ylim(0.0, 1.08)
    ax.set_xticks([0.275, 2.325])
    ax.set_xticklabels(["OSD-0 LER", r"$d_H$ AUROC"])
    ax.set_ylabel("value")
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main() -> None:
    _style()
    import matplotlib.pyplot as plt

    with open(os.path.join(HW_DIR, "hardware_analysis.json")) as fh:
        hw = json.load(fh)
    shots = np.load(os.path.join(HW_DIR, "decoded_shots.npz"))
    dh = shots["d_h"]
    fail = shots["fail_osd0"].astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.95), gridspec_kw={"width_ratios": [1.12, 1.15, 1.0]})
    _panel_a(axes[0])
    _panel_b(axes[1], dh, fail, hw["auroc_fail"], hw["auroc_fail_ci95"])
    _panel_c(axes[2], hw)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.96, bottom=0.16, wspace=0.32)
    path = _save(fig, "fig6_h2_fullcss.png")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
