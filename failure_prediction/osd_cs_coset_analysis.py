"""Compare min-S OSD-CS selection with coset-aware aggregation of the same family.

Usage:
    python -m failure_prediction.osd_cs_coset_analysis \
        --data outputs/failure_prediction/osd_cs_coset/coset.npz
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import numpy as np

CC, WC, CW, WW = 0, 1, 2, 3
CLASS = ["CC", "WC", "CW", "WW"]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def fmt(k, n) -> str:
    lo, hi = wilson(int(k), int(n))
    return f"{int(k)}/{int(n)} = {k / max(n, 1):.4f}  [{lo:.4f},{hi:.4f}]"


def report(d: dict) -> str:
    y = d["outcome"].astype(int)
    mins = d["mins_correct"].astype(int)
    agg = d["agg_correct"].astype(int)
    present = d["true_present"].astype(int)
    lines: List[str] = [
        "Coset-aware aggregation of the OSD surrogate",
        "  same candidate family: OSD-0 + k one-free-variable perturbations + 1 double",
        "  min-S: argmin_e S(e)",
        "  aggregate: L* = argmax_L sum_{e in C_L} exp[-S(e)]",
        "  S is the OSD channel-prior surrogate, not a negative log probability.",
        "  A one-free-variable perturbation re-solves He=s on the pivots; it is",
        "  not a single-qubit error.",
        "",
        f"  shots={y.size}  (replay of trajectories.npz, no new Monte Carlo)",
        f"  aggregate equals min-S on {int(d['agg_equals_mins'].sum())}/{y.size}",
        "",
        "1) Per-class correctness: min-S vs aggregate",
        f"  {'class':5s} {'n':>5s} {'min-S ok':>22s} {'aggregate ok':>22s} "
        f"{'true in family':>22s}",
    ]
    for c, name in enumerate(CLASS):
        m = y == c
        n = int(m.sum())
        if n == 0:
            continue
        lines.append(
            f"  {name:5s} {n:5d} {fmt(mins[m].sum(), n):>22s} "
            f"{fmt(agg[m].sum(), n):>22s} {fmt(present[m].sum(), n):>22s}"
        )

    # The three questions.
    ww = y == WW
    scoring_fail = ww & (present == 1) & (mins == 0)
    search_fail = ww & (present == 0)
    n_sf = int(scoring_fail.sum())
    rescued_sf = int(agg[scoring_fail].sum()) if n_sf else 0
    cw = y == CW
    wc = y == WC
    cc = y == CC

    lines += [
        "",
        "2) The three questions this experiment was for",
        f"  WW with correct class already in the family (scoring failure): {n_sf}",
        f"    of which aggregate rescues: {fmt(rescued_sf, n_sf) if n_sf else 'n/a'}",
        f"  WW with correct class absent (search failure): {int(search_fail.sum())}",
        f"    aggregate cannot help these by construction; still wrong: "
        f"{fmt(int((1 - agg[search_fail]).sum()), int(search_fail.sum()))}"
        if int(search_fail.sum())
        else "",
        f"  CW (min-S self-harm): {int(cw.sum())}",
        f"    aggregate still wrong: {fmt(int((1 - agg[cw]).sum()), int(cw.sum()))}",
        f"    aggregate repairs (back to OSD-0's correct coset): "
        f"{fmt(int(agg[cw].sum()), int(cw.sum()))}",
        f"  WC harmed (min-S correct → aggregate wrong): "
        f"{fmt(int(((mins == 1) & (agg == 0) & wc).sum()), int(wc.sum()))}",
        f"  high-d CC harmed: "
        f"{fmt(int(((mins == 1) & (agg == 0) & cc).sum()), int(cc.sum()))}",
    ]

    if n_sf:
        gap = d["delta_gap"][scoring_fail]
        neff = d["n_eff_true"][scoring_fail]
        n_true = d["n_true"][scoring_fail]
        log_ratio = d["logW_true"][scoring_fail] - d["logW_mins"][scoring_fail]
        lines += [
            "",
            "3) Why aggregate can or cannot beat min-S on scoring-failure WW",
            "  The wrong class is represented by one very good physical candidate.",
            "  log W(L) is dominated by max ΔS in that class, so a gap of G in ΔS",
            "  needs roughly exp(G) equal-scoring correct copies to tie.",
            f"  median ΔS gap (winner minus best true) = {np.median(gap):.3f}",
            f"  median n_true = {np.median(n_true):.1f}   median n_eff = {np.median(neff):.2f}",
            f"  median log W(true) - log W(min-S class) = {np.median(log_ratio):.3f}",
            f"  fraction with logW(true) > logW(min-S class) = "
            f"{fmt(int((log_ratio > 0).sum()), n_sf)}",
        ]

    # Transitions.
    both = (mins == 1) & (agg == 1)
    only_mins = (mins == 1) & (agg == 0)
    only_agg = (mins == 0) & (agg == 1)
    neither = (mins == 0) & (agg == 0)
    lines += [
        "",
        "4) Shot-level transitions (all analysed shots)",
        f"  both correct     {fmt(int(both.sum()), y.size)}",
        f"  min-S only       {fmt(int(only_mins.sum()), y.size)}",
        f"  aggregate only   {fmt(int(only_agg.sum()), y.size)}",
        f"  both wrong       {fmt(int(neither.sum()), y.size)}",
    ]
    return "\n".join(lines)


def figures(d: dict, out_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    y = d["outcome"].astype(int)
    scoring_fail = (
        (y == WW) & (d["true_present"] > 0.5) & (d["mins_correct"] < 0.5)
    )
    paths = []
    if not np.any(scoring_fail):
        return paths

    gap = d["delta_gap"][scoring_fail]
    log_ratio = d["logW_true"][scoring_fail] - d["logW_mins"][scoring_fail]
    rescued = d["agg_correct"][scoring_fail] > 0.5

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].hist(gap, bins=30, color="#7a7a7a", edgecolor="k", linewidth=0.4)
    axes[0].set_xlabel("ΔS(min-S winner) − ΔS(best true candidate)")
    axes[0].set_ylabel("WW scoring-failure shots")
    axes[0].set_title("Surrogate gap on scoring-failure WW")

    axes[1].scatter(
        gap[~rescued], log_ratio[~rescued], s=12, c="#7a7a7a", label="still wrong", alpha=0.8
    )
    if np.any(rescued):
        axes[1].scatter(
            gap[rescued], log_ratio[rescued], s=18, c="#d1495b", label="aggregate rescues", zorder=3
        )
    axes[1].axhline(0, ls="--", c="k", lw=1)
    axes[1].set_xlabel("ΔS gap")
    axes[1].set_ylabel("log W(true) − log W(min-S class)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Coset mass vs single-candidate gap")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_coset_gap.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="outputs/failure_prediction/osd_cs_coset/coset.npz")
    p.add_argument("--out-dir", default="outputs/failure_prediction/osd_cs_coset")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = np.load(args.data)
    d = {k: raw[k] for k in raw.files}
    text = report(d)
    print(text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "coset_report.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.out_dir, "coset_summary.json"), "w") as fh:
        json.dump(
            {
                "n": int(d["outcome"].size),
                "mins_ok": int(d["mins_correct"].sum()),
                "agg_ok": int(d["agg_correct"].sum()),
            },
            fh,
            indent=2,
        )
    figs = figures(d, args.out_dir)
    if figs:
        print("Figures: " + ", ".join(figs))


if __name__ == "__main__":
    main()
