"""Analyse Adaptive Decoder V1: LER vs escalation and measured cost.

Usage:
    python -m failure_prediction.adaptive_v1_analysis \\
        --dir outputs/failure_prediction/adaptive_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np

P_VALUES = (0.006, 0.007, 0.008)
CALIBRATE_P = 0.007
ESC_TARGETS = (0.10, 0.20, 0.30)


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def load_dir(path: str) -> Dict[float, dict]:
    files = sorted(glob.glob(os.path.join(path, "**", "*.npz"), recursive=True))
    files = [f for f in files if "timing" not in os.path.basename(f)]
    by_p: Dict[float, List[dict]] = {}
    for f in files:
        d = np.load(f)
        p = float(np.asarray(d["p"]).ravel()[0])
        by_p.setdefault(p, []).append({k: d[k] for k in d.files})
    out = {}
    for p, parts in by_p.items():
        keys = [k for k in parts[0] if k not in ("p", "seed")]
        merged = {k: np.concatenate([t[k] for t in parts]) for k in keys}
        merged["p"] = p
        merged["n"] = int(merged["fail_osd0"].size)
        out[p] = merged
    return out


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    # Mann-Whitney
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1)
    # ties: average
    i = 0
    sr = s[order]
    while i < s.size:
        j = i
        while j + 1 < s.size and sr[j + 1] == sr[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    u = ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0
    return float(u / (pos.sum() * neg.sum()))


def curve_for(d: dict, fail_key: str) -> List[dict]:
    dh = d["d_h"]
    fail0 = d["fail_osd0"]
    failk = d[fail_key]
    n = dh.size
    # Escalation if d_H >= τ. Sweep τ through unique values plus +inf.
    taus = np.concatenate([[-1.0], np.unique(dh), [dh.max() + 1.0]])
    rows = []
    for tau in taus:
        esc = dh >= tau
        f_esc = float(esc.mean())
        fail = np.where(esc, failk, fail0)
        kfail = int(fail.sum())
        lo, hi = wilson(kfail, n)
        tkey = "t_k500_ms" if fail_key == "fail_k500" else "t_k1000_ms"
        t_ms = float(np.mean(d["t_osd0_ms"] + esc * d[tkey]))
        rows.append(
            {
                "tau": float(tau),
                "f_esc": f_esc,
                "ler": kfail / n,
                "n_fail": kfail,
                "ci_lo": lo,
                "ci_hi": hi,
                "ms": t_ms,
            }
        )
    return rows


def tau_for_escalation(d: dict, f_target: float) -> float:
    """Smallest τ such that P(d_H ≥ τ) is as close as possible to f_target, from above.

    Using a high quantile of d_H: τ = quantile(d_H, 1 - f_target).
    """

    dh = d["d_h"]
    return float(np.quantile(dh, 1.0 - f_target))


def apply_tau(d: dict, tau: float, fail_key: str) -> dict:
    esc = d["d_h"] >= tau
    fail = np.where(esc, d[fail_key], d["fail_osd0"])
    n = fail.size
    k = int(fail.sum())
    lo, hi = wilson(k, n)
    tkey = "t_k500_ms" if fail_key == "fail_k500" else "t_k1000_ms"
    return {
        "tau": float(tau),
        "f_esc": float(esc.mean()),
        "ler": k / n,
        "n_fail": k,
        "n": n,
        "ci_lo": lo,
        "ci_hi": hi,
        "ms": float(np.mean(d["t_osd0_ms"] + esc * d[tkey])),
    }


def fmt_ler(row: dict) -> str:
    return (
        f"{row['ler']:.5f} [{row['ci_lo']:.5f},{row['ci_hi']:.5f}]  "
        f"n_fail={row['n_fail']}/{row.get('n', '?')}  "
        f"f_esc={row['f_esc']:.3f}  {row['ms']:.2f} ms"
    )


def report(by_p: Dict[float, dict], timing: dict | None) -> str:
    lines: List[str] = [
        "Adaptive Decoder V1",
        "  fast: BP@20 + OSD-0",
        "  router: d_H(e_BP, e_OSD0)  (online, no extra decode)",
        "  strong: truncated one-free-variable CS (no order-2 double)",
        "  K=500 primary, K=1000 sensitivity, full one-free CS accuracy reference",
        "",
    ]
    for p in P_VALUES:
        if p not in by_p:
            lines.append(f"  p={p:g}: MISSING")
            continue
        d = by_p[p]
        n = d["n"]
        lines += [
            f"p={p:g}  n={n}",
            f"  BP converged {100 * d['bp_converged'].mean():.2f}%",
            f"  AUROC d_H vs OSD-0 fail: {auroc(d['fail_osd0'], d['d_h']):.4f}",
            f"  AUROC d_H vs beneficial (OSD-0 fail & full CS ok): "
            f"{auroc((d['fail_osd0'] == 1) & (d['fail_full'] == 0), d['d_h']):.4f}",
            f"  LER OSD-0     {d['fail_osd0'].mean():.5f}  "
            f"mean t={d['t_osd0_ms'].mean():.2f} ms",
            f"  LER K=500 all {d['fail_k500'].mean():.5f}  "
            f"mean t_branch={d['t_k500_ms'].mean():.2f} ms",
            f"  LER K=1000 all {d['fail_k1000'].mean():.5f}  "
            f"mean t_branch={d['t_k1000_ms'].mean():.2f} ms",
            f"  LER full 1FV  {d['fail_full'].mean():.5f}  "
            f"mean t_branch={d['t_full_ms'].mean():.2f} ms",
            "",
        ]

    if CALIBRATE_P not in by_p:
        lines.append("Cannot calibrate τ: p=0.007 records missing.")
        return "\n".join(lines)

    cal = by_p[CALIBRATE_P]
    tau20 = tau_for_escalation(cal, 0.20)
    lines += [
        f"1) Representative τ calibrated at p={CALIBRATE_P:g} for ~20% escalation",
        f"   τ* = {tau20:.1f}  (quantile of d_H at 0.80)",
        "",
    ]
    for p in P_VALUES:
        if p not in by_p:
            continue
        r500 = apply_tau(by_p[p], tau20, "fail_k500")
        r1000 = apply_tau(by_p[p], tau20, "fail_k1000")
        lines.append(f"  p={p:g}  K=500   {fmt_ler(r500)}")
        lines.append(f"  p={p:g}  K=1000  {fmt_ler(r1000)}")

    lines += ["", "2) Accuracy recovered at 10% / 20% / 30% escalation (τ set per p)"]
    for p in P_VALUES:
        if p not in by_p:
            continue
        d = by_p[p]
        n = d["n"]
        ler0 = d["fail_osd0"].mean()
        ler_full = d["fail_full"].mean()
        lines.append(f"  p={p:g}  OSD-0 LER={ler0:.5f}  full 1FV LER={ler_full:.5f}")
        for f in ESC_TARGETS:
            tau = tau_for_escalation(d, f)
            r = apply_tau(d, tau, "fail_k500")
            gap = ler0 - ler_full
            rec = (ler0 - r["ler"]) / gap if gap > 0 else float("nan")
            lines.append(
                f"    f_esc≈{f:.2f} (τ={tau:.1f}, actual {r['f_esc']:.3f})  "
                f"K=500 LER={r['ler']:.5f}  recovered {100 * rec:.1f}% of OSD-0→full gap  "
                f"{r['ms']:.1f} ms"
            )
            r2 = apply_tau(d, tau, "fail_k1000")
            rec2 = (ler0 - r2["ler"]) / gap if gap > 0 else float("nan")
            lines.append(
                f"                         K=1000 LER={r2['ler']:.5f}  "
                f"recovered {100 * rec2:.1f}%  {r2['ms']:.1f} ms"
            )

    if timing:
        lines += ["", "3) Serial wall-clock on the same node (not inferred from candidate counts)"]
        for name, row in timing.items():
            if not isinstance(row, dict) or "mean_ms" not in row:
                continue
            lines.append(
                f"  {name:24s}  n={row.get('n', '?')}  "
                f"mean={row.get('mean_ms', float('nan')):.2f} ms  "
                f"median={row.get('median_ms', float('nan')):.2f} ms"
            )

    lines += ["", "4) K=500 vs K=1000 at the calibrated τ*"]
    if CALIBRATE_P in by_p:
        tau20 = tau_for_escalation(by_p[CALIBRATE_P], 0.20)
        for p in P_VALUES:
            if p not in by_p:
                continue
            a = apply_tau(by_p[p], tau20, "fail_k500")
            b = apply_tau(by_p[p], tau20, "fail_k1000")
            lines.append(
                f"  p={p:g}  LER K500={a['ler']:.5f}  K1000={b['ler']:.5f}  "
                f"Δ={b['ler'] - a['ler']:+.5f}  "
                f"ms K500={a['ms']:.1f} K1000={b['ms']:.1f}"
            )

    lines += ["", "5) Checks that would contradict the adaptive reading"]
    for p in P_VALUES:
        if p not in by_p:
            continue
        d = by_p[p]
        auc0 = auroc(d["fail_osd0"], d["d_h"])
        if auc0 < 0.7:
            lines.append(f"  CONTRADICTION p={p:g}: d_H vs OSD-0-fail AUROC={auc0:.3f} < 0.7")
        r = apply_tau(d, tau_for_escalation(d, 0.20), "fail_k500")
        if r["ler"] > d["fail_osd0"].mean() + 1e-12:
            lines.append(
                f"  CONTRADICTION p={p:g}: 20% escalation K=500 LER "
                f"{r['ler']:.5f} worse than OSD-0 {d['fail_osd0'].mean():.5f}"
            )
        if d["fail_k500"].mean() > d["fail_osd0"].mean() + 1e-12:
            lines.append(
                f"  CONTRADICTION p={p:g}: always-on K=500 LER "
                f"{d['fail_k500'].mean():.5f} worse than OSD-0"
            )
    if not any("CONTRADICTION" in x for x in lines):
        lines.append("  none at p in {0.006, 0.007, 0.008}")

    lines += [
        "",
        "6) Notes",
        "  Escalation cost includes Gaussian elimination plus the K one-free-variable",
        "  scores. Candidate-count reduction is not a wall-clock claim.",
        "  d_H is an empirical disagreement statistic, not a claimed universal hardness.",
        "  Full one-free-variable CS is the strong-branch accuracy reference;",
        "  package OSD-CS (order 2) is timed on a serial subsample only.",
        "  STOP after this report. No cross-code, no QPU, no ML router.",
    ]
    return "\n".join(lines)


def figures(by_p: Dict[float, dict], out_dir: str) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    colors = {0.006: "#4c78a8", 0.007: "#f58518", 0.008: "#d1495b"}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    for p, d in sorted(by_p.items()):
        c500 = curve_for(d, "fail_k500")
        xs = [r["f_esc"] for r in c500]
        ys = [r["ler"] for r in c500]
        ax.plot(xs, ys, "-", color=colors.get(p, "k"), label=f"p={p:g} K=500", lw=2)
        c1000 = curve_for(d, "fail_k1000")
        ax.plot(
            [r["f_esc"] for r in c1000],
            [r["ler"] for r in c1000],
            "--",
            color=colors.get(p, "k"),
            label=f"p={p:g} K=1000",
            lw=1.4,
        )
        ax.scatter([0.0], [d["fail_osd0"].mean()], marker="s", color=colors.get(p, "k"), zorder=3)
        ax.scatter([1.0], [d["fail_full"].mean()], marker="o", color=colors.get(p, "k"), zorder=3)
    ax.set_xlabel("escalation fraction f_esc")
    ax.set_ylabel("LER")
    ax.set_title("Adaptive V1: LER vs escalation")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    for p, d in sorted(by_p.items()):
        c500 = curve_for(d, "fail_k500")
        ax.plot(
            [r["ms"] for r in c500],
            [r["ler"] for r in c500],
            "-",
            color=colors.get(p, "k"),
            label=f"p={p:g} K=500",
            lw=2,
        )
        ax.scatter(
            [d["t_osd0_ms"].mean()],
            [d["fail_osd0"].mean()],
            marker="s",
            color=colors.get(p, "k"),
            zorder=3,
        )
        ax.scatter(
            [(d["t_osd0_ms"] + d["t_full_ms"]).mean()],
            [d["fail_full"].mean()],
            marker="o",
            color=colors.get(p, "k"),
            zorder=3,
        )
    ax.set_xlabel("mean wall-clock ms / shot (OSD-0 + gated truncated CS)")
    ax.set_ylabel("LER")
    ax.set_title("Adaptive V1: LER vs measured cost")
    ax.legend(fontsize=7)
    fig.tight_layout()
    pth = os.path.join(out_dir, "fig1_ler_tradeoff.png")
    fig.savefig(pth, dpi=160)
    plt.close(fig)
    paths.append(pth)

    if CALIBRATE_P in by_p:
        tau20 = tau_for_escalation(by_p[CALIBRATE_P], 0.20)
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ps, ler500, ler0, lerf, fesc = [], [], [], [], []
        for p in P_VALUES:
            if p not in by_p:
                continue
            r = apply_tau(by_p[p], tau20, "fail_k500")
            ps.append(p)
            ler500.append(r["ler"])
            ler0.append(by_p[p]["fail_osd0"].mean())
            lerf.append(by_p[p]["fail_full"].mean())
            fesc.append(r["f_esc"])
        ax.plot(ps, ler0, "s--", label="OSD-0", color="#7a7a7a")
        ax.plot(ps, ler500, "o-", label=f"adaptive K=500, τ*={tau20:.0f}", color="#d1495b")
        ax.plot(ps, lerf, "^--", label="full one-free CS", color="#4c78a8")
        ax.set_xlabel("p")
        ax.set_ylabel("LER")
        ax.set_title("Same τ* (calibrated at p=0.007) across operating points")
        ax.legend(fontsize=8)
        fig.tight_layout()
        pth = os.path.join(out_dir, "fig2_tau_transfer.png")
        fig.savefig(pth, dpi=160)
        plt.close(fig)
        paths.append(pth)
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="outputs/failure_prediction/adaptive_v1")
    p.add_argument("--timing", default=None)
    p.add_argument("--out-dir", default="outputs/failure_prediction/adaptive_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    by_p = load_dir(os.path.join(args.dir, "parts") if os.path.isdir(os.path.join(args.dir, "parts")) else args.dir)
    timing = None
    tpath = args.timing or os.path.join(args.dir, "serial_timing.json")
    if os.path.isfile(tpath):
        with open(tpath) as fh:
            timing = json.load(fh)
    text = report(by_p, timing)
    print(text)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "adaptive_v1_report.txt"), "w") as fh:
        fh.write(text + "\n")
    figs = figures(by_p, args.out_dir)
    print("Figures: " + ", ".join(figs))


if __name__ == "__main__":
    main()
