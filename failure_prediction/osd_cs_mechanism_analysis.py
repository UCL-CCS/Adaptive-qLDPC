"""Analyse OSD-CS rescue trajectories: what actually saves WC, and why WW is not saved.

Reads the replay tables written by `osd_cs_mechanism`. Score S is the package
objective sum_{i: x_i=1} log(1/p_i), not a probability and not Hamming weight.
Delta S = S(OSD-0) - S(winner) is a prior-weighted cost improvement.

Usage:
    python -m failure_prediction.osd_cs_mechanism_analysis \
        --data outputs/failure_prediction/osd_cs_mechanism/trajectories.npz
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np

CC, WC, CW, WW = 0, 1, 2, 3
CLASS = ["CC", "WC", "CW", "WW"]
WINNER = ["osd0", "single", "double"]
K_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def records_of(d, i: int):
    a, b = int(d["rec_offsets"][i]), int(d["rec_offsets"][i + 1])
    return (
        d["rec_idx"][a:b],
        d["rec_score"][a:b],
        d["rec_correct"][a:b],
    )


def truncated_correct(d, i: int, k_singles: int) -> int:
    """Logical correctness of the incumbent after the first K singles (no double)."""

    idx, _, cor = records_of(d, i)
    m = idx < k_singles
    if not np.any(m):
        return int(d["osd0_correct"][i])
    return int(cor[m][-1])


def truncated_with_double(d, i: int) -> int:
    """Incumbent after all singles plus the order-2 double: the full sweep."""

    return int(d["cs_correct"][i])


def load(path: str) -> dict:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def fmt_frac(k, n) -> str:
    lo, hi = wilson(int(k), int(n))
    return f"{k}/{n} = {k / max(n, 1):.4f}  [{lo:.4f},{hi:.4f}]"


def report(d: dict) -> str:
    y = d["outcome"].astype(int)
    lines: List[str] = [
        "OSD-CS rescue mechanism",
        "  score S(x) = sum_{i: x_i=1} log(1/p_i)  (channel priors, not BP, not Hamming)",
        "  Delta S = S(OSD-0) - S(winner); a candidate wins only on a strict increase",
        "  CS family = k singles in ascending-LPR free-column order + 1 double on (0,1)",
        "",
    ]
    agree = d.get("label_agreement")
    if agree is not None:
        lines.append(
            f"  replayed labels agree with the 20k dataset on "
            f"{int(agree[0])}/{int(agree[1])} selected shots"
        )
    if "h_delta_zero" in d:
        lines.append(
            f"  H (e0 XOR e_CS) = 0 on {int(d['h_delta_zero'].sum())}/"
            f"{d['h_delta_zero'].size} shots"
        )
    lines.append("")

    lines.append("1) Candidate-search behaviour by class")
    lines.append(
        f"  {'class':5s} {'n':>5s} {'osd0':>6s} {'single':>7s} {'double':>7s} "
        f"{'med ΔS':>8s} {'mean ΔS':>8s} {'coset Δ':>8s}"
    )
    for c, name in enumerate(CLASS):
        m = y == c
        n = int(m.sum())
        if n == 0:
            continue
        wt = d["winner_type"][m].astype(int)
        ds = d["winner_score"][m]
        coset = d["winner_logical_nontrivial"][m]
        lines.append(
            f"  {name:5s} {n:5d} {int((wt == 0).sum()):6d} {int((wt == 1).sum()):7d} "
            f"{int((wt == 2).sum()):7d} {np.median(ds):8.3f} {np.mean(ds):8.3f} "
            f"{int(coset.sum()):8d}"
        )

    wc = y == WC
    ww = y == WW
    cw = y == CW
    n_wc, n_ww, n_cw = int(wc.sum()), int(ww.sum()), int(cw.sum())

    lines += ["", "2A) What rescues WC?"]
    wt = d["winner_type"][wc].astype(int)
    n_single = int((wt == 1).sum())
    n_double = int((wt == 2).sum())
    n_osd0 = int((wt == 0).sum())
    lines.append(f"  single-flip winner : {fmt_frac(n_single, n_wc)}")
    lines.append(f"  double-flip winner : {fmt_frac(n_double, n_wc)}")
    lines.append(f"  OSD-0 still wins   : {fmt_frac(n_osd0, n_wc)}  (should be 0 for WC)")

    ranks = d["winner_rank"][wc & (d["winner_type"] == 1)].astype(int)
    lines.append("  single-winner free-index rank (0 = least-LPR free column):")
    if ranks.size:
        lines.append(
            f"    median={np.median(ranks):.0f}  p90={np.quantile(ranks, 0.9):.0f}  "
            f"p95={np.quantile(ranks, 0.95):.0f}  p99={np.quantile(ranks, 0.99):.0f}  "
            f"max={ranks.max()}"
        )
        for k in (1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
            hit = int((ranks < k).sum())
            lines.append(f"    final winner rank < {k:5d} : {fmt_frac(hit, n_single)}")
    n_correct = d["n_correct_candidates"][wc]
    lines.append(
        f"  correct singles already in the family: min={int(n_correct.min())}  "
        f"median={np.median(n_correct):.0f}  max={int(n_correct.max())}"
    )
    lines.append(
        "  so WC rescue is not a unique needle: the right logical class is typically "
        "represented many times, and CS then picks the best-scoring of those."
    )

    lines += ["", "2B) Why are WW not rescued?"]
    n_in_family = int((d["n_correct_plus_double"][ww] > 0).sum()) if "n_correct_plus_double" in d else int(
        (d["n_correct_candidates"][ww] > 0).sum()
    )
    n_singles_ok = int((d["n_correct_candidates"][ww] > 0).sum())
    n_double_ok = int((d["double_correct"][ww] > 0).sum())
    lines.append(
        f"  correct logical class is in the CS family : {fmt_frac(n_in_family, n_ww)}"
    )
    lines.append(f"    via a single flip : {fmt_frac(n_singles_ok, n_ww)}")
    lines.append(f"    via the double    : {fmt_frac(n_double_ok, n_ww)}")
    missing = n_ww - n_in_family
    lines.append(
        f"  correct class never appears in the k+1 candidates : {fmt_frac(missing, n_ww)}"
    )
    # Scoring failure: correct candidate exists but a wrong one scores better.
    exists = d["n_correct_candidates"][ww] > 0
    if np.any(exists):
        best_ok = d["best_correct_score"][ww][exists]
        win = d["winner_score"][ww][exists]
        ranked_wrong = int(np.sum(best_ok + 1e-15 < win))
        lines.append(
            f"  among WW with a correct single: scoring picks a *wrong* higher-ΔS "
            f"candidate on {fmt_frac(ranked_wrong, int(exists.sum()))}"
        )
        lines.append(
            f"    median ΔS(best correct)={np.median(best_ok):.3f}  "
            f"median ΔS(winner)={np.median(win):.3f}"
        )
    lines.append(
        f"  median ΔS  WC={np.median(d['winner_score'][wc]):.3f}  "
        f"WW={np.median(d['winner_score'][ww]):.3f}  "
        f"CW={np.median(d['winner_score'][cw]):.3f}" if n_cw else
        f"  median ΔS  WC={np.median(d['winner_score'][wc]):.3f}  "
        f"WW={np.median(d['winner_score'][ww]):.3f}"
    )

    lines += ["", "2C) CW self-harm (N is small; do not overgeneralise)"]
    if n_cw:
        wt_cw = d["winner_type"][cw].astype(int)
        lines.append(
            f"  n={n_cw}  winner: osd0={int((wt_cw == 0).sum())} "
            f"single={int((wt_cw == 1).sum())} double={int((wt_cw == 2).sum())}"
        )
        lines.append(
            f"  all {n_cw} have ΔS>0 and a nontrivial logical action: "
            "surrogate-score improvement ≠ logical improvement"
            if np.all(d["winner_score"][cw] > 0)
            and np.all(d["winner_logical_nontrivial"][cw] > 0)
            else f"  ΔS>0 on {int((d['winner_score'][cw] > 0).sum())}/{n_cw}, "
            f"logical-coset change on {int(d['winner_logical_nontrivial'][cw].sum())}/{n_cw}"
        )
        ranks_cw = d["winner_rank"][cw & (d["winner_type"] == 1)].astype(int)
        if ranks_cw.size:
            lines.append(
                f"  single-winner ranks: median={np.median(ranks_cw):.0f}  "
                f"max={ranks_cw.max()}"
            )

    lines += ["", "3) Logical-coset transitions"]
    lines.append(
        "  δ = e0 XOR e_CS lives in ker(H) (same syndrome). It is a nontrivial "
        "logical iff the two decoders disagree on a Stim observable."
    )
    for c, name in [(WC, "WC"), (WW, "WW"), (CW, "CW")]:
        m = y == c
        n = int(m.sum())
        if n == 0:
            continue
        changed = int(d["winner_logical_nontrivial"][m].sum())
        ham = d["delta_hamming"][m] if "delta_hamming" in d else None
        extra = (
            f"  median |δ|={np.median(ham):.0f}" if ham is not None else ""
        )
        lines.append(
            f"  {name}: coset-changing winner {fmt_frac(changed, n)}{extra}"
        )

    # Truncated search.
    lines += [
        "",
        "4) Truncated combination sweep",
        "  K = number of single-flip candidates evaluated (double is last, shown separately)",
        f"  {'K':>8s} {'WC rescued':>22s} {'frac of 915':>12s} "
        f"{'high-d CC harm':>16s}",
    ]
    cc = y == CC
    n_cc = int(cc.sum())
    curve = []
    for k in K_GRID:
        rescued = int(sum(truncated_correct(d, i, k) for i in np.flatnonzero(wc)))
        harmed = (
            int(sum(1 - truncated_correct(d, i, k) for i in np.flatnonzero(cc)))
            if n_cc
            else 0
        )
        lo, hi = wilson(rescued, n_wc)
        lines.append(
            f"  {k:8d}  WC {rescued:4d}/{n_wc} = {rescued / max(n_wc, 1):.4f} "
            f"[{lo:.4f},{hi:.4f}]   high-d CC harm {fmt_frac(harmed, max(n_cc, 1))}"
        )
        curve.append(
            {
                "k": k,
                "wc_rescued": rescued,
                "wc_n": n_wc,
                "cc_harmed": harmed,
                "cc_n": n_cc,
            }
        )
    # Full sweep (all singles + double): cs_correct
    rescued_full = int(d["cs_correct"][wc].sum())
    harmed_full = int((1 - d["cs_correct"][cc]).sum()) if n_cc else 0
    lines.append(
        f"  {'all+dbl':>8s}  WC {rescued_full:4d}/{n_wc} = {rescued_full / max(n_wc, 1):.4f} "
        f"  high-d CC harm {fmt_frac(harmed_full, max(n_cc, 1))}"
    )
    curve.append(
        {
            "k": "all+double",
            "wc_rescued": rescued_full,
            "wc_n": n_wc,
            "cc_harmed": harmed_full,
            "cc_n": n_cc,
        }
    )

    # Exact first K at which a correct incumbent appears (operational truncation).
    first_k = []
    for i in np.flatnonzero(wc):
        ridx, _, rcor = records_of(d, i)
        hit = None
        for rr, cc in zip(ridx, rcor):
            if cc:
                hit = int(rr) + 1
                break
        first_k.append(hit)
    ok = np.array([k for k in first_k if k is not None], dtype=int)
    ok.sort()
    lines += [
        "",
        "  Operational K: first time the running incumbent is logically correct.",
        f"  defined on {ok.size}/{n_wc} WC shots  median={np.median(ok):.0f}",
    ]
    for frac in (0.80, 0.90, 0.95):
        need = int(np.ceil(frac * n_wc))
        k_hit = int(ok[need - 1]) if need <= ok.size else None
        n_hit = int(np.sum(ok <= k_hit)) if k_hit is not None else 0
        lines.append(
            f"    {100 * frac:.0f}% of WC ({need}/{n_wc}) first reached at K = {k_hit} "
            f"(rescued {n_hit})"
        )
    lines.append(
        "  This is much larger than 20: rescue lives deep in the free-column list, "
        "not in the first handful of CS candidates. K ≈ 500 is still only ~5% of the "
        f"full k+1 ≈ {int(d['n_candidates'].mean())} search."
    )
    return "\n".join(lines), curve


def figures(d: dict, curve: List[dict], out_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    y = d["outcome"].astype(int)
    paths = []

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8))
    colors = {"WC": "#d1495b", "WW": "#7a7a7a", "CW": "#edae49"}
    for name, c in (("WC", WC), ("WW", WW), ("CW", CW)):
        m = y == c
        if not m.any():
            continue
        vals = np.sort(d["winner_score"][m])
        axes[0].step(
            vals, np.arange(1, vals.size + 1) / vals.size,
            where="post", color=colors[name], label=f"{name} n={m.sum()}", lw=2,
        )
    axes[0].set_xlabel("ΔS = S(OSD-0) − S(winner)")
    axes[0].set_ylabel("ECDF")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Score improvement by class")

    wc = y == WC
    ranks = d["winner_rank"][wc & (d["winner_type"] == 1)].astype(int)
    if ranks.size:
        ks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
        cdf = np.array([(ranks < k).mean() for k in ks])
        axes[1].plot(ks, cdf, "o-", color="#d1495b")
        axes[1].set_xscale("log")
        axes[1].set_ylim(0, 1.02)
        axes[1].set_xlabel("winning free-index rank < K")
        axes[1].set_ylabel("fraction of single-flip WC")
        axes[1].set_title("Where in the free-column list the rescue lives")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig1_winner_type_deltaS.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ks = [row["k"] for row in curve if isinstance(row["k"], int)]
    rescued = [row["wc_rescued"] / max(row["wc_n"], 1) for row in curve if isinstance(row["k"], int)]
    ax.plot(ks, rescued, "o-", color="#d1495b", label="WC rescued (truncated singles)")
    full = curve[-1]["wc_rescued"] / max(curve[-1]["wc_n"], 1)
    ax.axvline(504, ls=":", c="0.4", label="80% WC @ K=504")
    ax.axvline(714, ls=":", c="0.55", label="90% WC @ K=714")
    ax.axvline(1042, ls=":", c="0.7", label="95% WC @ K=1042")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("K = number of single-flip candidates evaluated")
    ax.set_ylabel("fraction of WC rescued")
    ax.legend(fontsize=8)
    ax.set_title("Truncated OSD-CS: rescue versus search budget")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig2_truncated_search.png")
    fig.savefig(p, dpi=160)
    plt.close(fig)
    paths.append(p)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data",
        default="outputs/failure_prediction/osd_cs_mechanism/trajectories.npz",
    )
    p.add_argument(
        "--out-dir", default="outputs/failure_prediction/osd_cs_mechanism"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    d = load(args.data)
    text, curve = report(d)
    print(text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "mechanism_report.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.out_dir, "mechanism_summary.json"), "w") as fh:
        json.dump({"curve": curve}, fh, indent=2, default=str)
    figs = figures(d, curve, args.out_dir)
    print("\nFigures: " + ", ".join(figs))


if __name__ == "__main__":
    main()
