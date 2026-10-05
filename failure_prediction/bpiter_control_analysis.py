"""Analyse the BP-effort control: does BP max_iter change the OSD-0/OSD-CS gap?

The control holds code and noise fixed (bb_144_12_12, p=0.008, r=24) and varies
only BP max_iter, running the OSD-0 and OSD-CS arms at matched effort so the
separation is measurable at each level. Two mechanisms predict different things:

  candidate-space / code-size   the gap is set by how much room OSD-CS has to
                                search beyond the OSD-0 pivot, so it is flat in
                                max_iter;
  poor BP convergence           the gap is BP repairing its own transients, so
                                it shrinks as max_iter grows and BP converges.

All max_iter levels are run on the same seed and circuit, hence on identical
shots. Per-shot outcomes are not persisted, so the cross-level comparisons here
are unpaired and therefore conservative: an effect that fails to show up
unpaired on identical shots is not a large effect.

Usage:
    python -m failure_prediction.bpiter_control_analysis \
        --dir outputs/failure_prediction/decoder_survey/bpiter_control
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from math import erfc, sqrt
from typing import Any, Dict, List, Optional

import numpy as np


def _wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _two_prop_z(k1: int, n1: int, k2: int, n2: int) -> Dict[str, float]:
    """Unpaired two-proportion test, used only across max_iter levels."""

    if n1 == 0 or n2 == 0:
        return {"z": 0.0, "p": 1.0}
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return {"z": 0.0, "p": 1.0}
    z = (p1 - p2) / se
    return {"z": float(z), "p": float(erfc(abs(z) / sqrt(2.0)))}


def _log_ratio_ci(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple:
    """Delta-method CI for the ratio of two independent proportions."""

    if k1 == 0 or k2 == 0:
        return float("nan"), float("nan"), float("nan")
    ratio = (k1 / n1) / (k2 / n2)
    se_log = sqrt(1.0 / k1 - 1.0 / n1 + 1.0 / k2 - 1.0 / n2)
    return (
        float(ratio),
        float(ratio * np.exp(-z * se_log)),
        float(ratio * np.exp(z * se_log)),
    )


def _chi2_sf(x: float, df: int) -> float:
    """Survival function for chi-square, via the regularised upper gamma."""

    try:
        from math import gamma, lgamma

        # Series for the regularised lower incomplete gamma P(a, x).
        a, xx = df / 2.0, x / 2.0
        if xx <= 0:
            return 1.0
        if xx < a + 1.0:
            term = 1.0 / a
            total = term
            for n in range(1, 300):
                term *= xx / (a + n)
                total += term
                if abs(term) < abs(total) * 1e-14:
                    break
            p_low = total * np.exp(-xx + a * np.log(xx) - lgamma(a))
            return float(max(0.0, min(1.0, 1.0 - p_low)))
        # Continued fraction for Q(a, x) in the other regime.
        tiny = 1e-300
        b, c, d, h = xx + 1.0 - a, 1.0 / tiny, 1.0 / (xx + 1.0 - a), 1.0 / (xx + 1.0 - a)
        for i in range(1, 300):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < tiny:
                d = tiny
            c = b + an / c
            if abs(c) < tiny:
                c = tiny
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-14:
                break
        q = np.exp(-xx + a * np.log(xx) - lgamma(a)) * h
        return float(max(0.0, min(1.0, q)))
    except Exception:
        return float("nan")


def _homogeneity(counts: List[int], totals: List[int]) -> Dict[str, float]:
    """Chi-square test that several binomial proportions share a common value."""

    k = sum(counts)
    n = sum(totals)
    if n == 0 or k == 0 or k == n or len(counts) < 2:
        return {"chi2": 0.0, "df": max(0, len(counts) - 1), "p": 1.0}
    pool = k / n
    chi2 = 0.0
    for ki, ni in zip(counts, totals):
        exp_fail = ni * pool
        exp_ok = ni * (1 - pool)
        if exp_fail > 0:
            chi2 += (ki - exp_fail) ** 2 / exp_fail
        if exp_ok > 0:
            chi2 += ((ni - ki) - exp_ok) ** 2 / exp_ok
    df = len(counts) - 1
    return {"chi2": float(chi2), "df": int(df), "p": _chi2_sf(chi2, df)}


def _load(directory: str) -> List[Dict[str, Any]]:
    """Collect one survey JSON per max_iter level."""

    files = sorted(glob.glob(os.path.join(directory, "**", "*.json"), recursive=True))
    levels: List[Dict[str, Any]] = []
    for path in files:
        with open(path) as fh:
            data = json.load(fh)
        names = [r["decoder"] for r in data.get("decoders", [])]
        iters = set()
        for name in names:
            m = re.match(r"bp(\d+)", name)
            if m:
                iters.add(int(m.group(1)))
        if len(iters) != 1:
            continue
        max_iter = iters.pop()
        by = {r["decoder"]: r for r in data["decoders"]}
        osd0 = by.get(f"bp{max_iter}_osd0")
        osdcs = by.get(f"bp{max_iter}_osdcs2")
        if osd0 is None or osdcs is None:
            continue
        pair: Optional[Dict[str, Any]] = None
        for row in data.get("pairs", []):
            if row["fast"] == osd0["decoder"] and row["strong"] == osdcs["decoder"]:
                pair = row
                break
        levels.append(
            {
                "max_iter": max_iter,
                "path": path,
                "meta": data,
                "osd0": osd0,
                "osdcs": osdcs,
                "pair": pair,
            }
        )
    levels.sort(key=lambda r: r["max_iter"])
    return levels


def report(levels: List[Dict[str, Any]]) -> str:
    if not levels:
        return "No usable control outputs found."

    meta = levels[0]["meta"]
    out: List[str] = [
        "BP-effort control: is the OSD-0 / OSD-CS separation driven by BP effort?",
        f"  code={meta['code']}  p={meta['p']:g}  rounds={meta['rounds']}  "
        f"shots={meta['shots']}  seed={meta['seed']}",
        "  Code and noise fixed; only BP max_iter varies. All levels share the seed",
        "  and circuit, so the shot set is identical across levels.",
        "",
        "1) BP convergence and the two arms at matched effort",
        f"  {'max_iter':>8s} {'BP conv':>8s} {'LER osd0':>10s} {'95% CI':>18s} "
        f"{'LER osdcs2':>11s} {'95% CI':>18s} {'ms osd0':>9s} {'ms osdcs2':>10s}",
    ]

    for lv in levels:
        n = lv["meta"]["shots"]
        k0, kc = lv["osd0"]["n_fail"], lv["osdcs"]["n_fail"]
        lo0, hi0 = _wilson(k0, n)
        loc, hic = _wilson(kc, n)
        out.append(
            f"  {lv['max_iter']:8d} {lv['osd0']['converged_rate']:8.4f} "
            f"{lv['osd0']['ler']:10.5f} [{lo0:7.5f},{hi0:7.5f}] "
            f"{lv['osdcs']['ler']:11.5f} [{loc:7.5f},{hic:7.5f}] "
            f"{lv['osd0']['mean_ms']:9.2f} {lv['osdcs']['mean_ms']:10.2f}"
        )
    out.append(
        "  Caveat: the levels ran as separate array tasks on different nodes, so the"
    )
    out.append(
        "  ms/shot columns carry node-speed differences and are not comparable across"
    )
    out.append("  rows. Only the LER columns support cross-level comparison.")

    out += [
        "",
        "2) The separation at each effort level",
        f"  {'max_iter':>8s} {'ratio':>7s} {'95% CI':>16s} {'abs gap':>9s} "
        f"{'fOnly':>6s} {'sOnly':>6s} {'McNemar p':>10s}",
    ]
    for lv in levels:
        n = lv["meta"]["shots"]
        k0, kc = lv["osd0"]["n_fail"], lv["osdcs"]["n_fail"]
        ratio, rlo, rhi = _log_ratio_ci(k0, n, kc, n)
        pair = lv["pair"] or {}
        mcn = pair.get("mcnemar") or {}
        out.append(
            f"  {lv['max_iter']:8d} {ratio:7.2f} [{rlo:6.2f},{rhi:6.2f}] "
            f"{lv['osd0']['ler'] - lv['osdcs']['ler']:9.5f} "
            f"{pair.get('n_fast_only_wrong', -1):6d} "
            f"{pair.get('n_strong_only_wrong', -1):6d} "
            f"{mcn.get('p_value_two_sided', float('nan')):10.3g}"
        )

    out += ["", "3) Does BP effort move anything? (unpaired across levels, conservative)"]
    n_list = [lv["meta"]["shots"] for lv in levels]
    h0 = _homogeneity([lv["osd0"]["n_fail"] for lv in levels], n_list)
    hc = _homogeneity([lv["osdcs"]["n_fail"] for lv in levels], n_list)
    out.append(
        f"  OSD-0   LER homogeneous across max_iter: chi2={h0['chi2']:.2f} "
        f"df={h0['df']} p={h0['p']:.3g}"
    )
    out.append(
        f"  OSD-CS  LER homogeneous across max_iter: chi2={hc['chi2']:.2f} "
        f"df={hc['df']} p={hc['p']:.3g}"
    )

    lo, hi = levels[0], levels[-1]
    n_lo, n_hi = lo["meta"]["shots"], hi["meta"]["shots"]
    t0 = _two_prop_z(lo["osd0"]["n_fail"], n_lo, hi["osd0"]["n_fail"], n_hi)
    tc = _two_prop_z(lo["osdcs"]["n_fail"], n_lo, hi["osdcs"]["n_fail"], n_hi)
    out.append(
        f"  OSD-0   max_iter {lo['max_iter']} vs {hi['max_iter']}: "
        f"z={t0['z']:+.2f} p={t0['p']:.3g}"
    )
    out.append(
        f"  OSD-CS  max_iter {lo['max_iter']} vs {hi['max_iter']}: "
        f"z={tc['z']:+.2f} p={tc['p']:.3g}"
    )

    r_lo, rlo_lo, rhi_lo = _log_ratio_ci(
        lo["osd0"]["n_fail"], n_lo, lo["osdcs"]["n_fail"], n_lo
    )
    r_hi, rlo_hi, rhi_hi = _log_ratio_ci(
        hi["osd0"]["n_fail"], n_hi, hi["osdcs"]["n_fail"], n_hi
    )
    overlap = not (rhi_lo < rlo_hi or rhi_hi < rlo_lo)
    out += [
        "",
        "4) Verdict",
        f"  separation ratio at max_iter={lo['max_iter']}: {r_lo:.2f} "
        f"[{rlo_lo:.2f},{rhi_lo:.2f}]",
        f"  separation ratio at max_iter={hi['max_iter']}: {r_hi:.2f} "
        f"[{rlo_hi:.2f},{rhi_hi:.2f}]",
    ]
    conv_rates = [lv["osd0"]["converged_rate"] for lv in levels]
    if max(conv_rates) < 0.05:
        out.append(
            f"  BP converges on at most {100 * max(conv_rates):.2f}% of shots at every "
            "effort level, so OSD is carrying the decode throughout."
        )
    if overlap and hc["p"] > 0.05:
        out.append(
            "  The separation ratio CIs overlap and OSD-CS LER is homogeneous in "
            "max_iter: the gap is insensitive to BP effort, consistent with the "
            "OSD candidate-space mechanism rather than poor BP convergence."
        )
    else:
        out.append(
            "  The separation shifts with max_iter: BP effort is implicated and the "
            "candidate-space reading alone does not account for the gap."
        )
    if h0["p"] < 0.05:
        out.append(
            "  Note: OSD-0 LER alone does move with max_iter, so BP effort is not "
            "inert; it shifts both arms rather than the separation between them."
        )
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        default="outputs/failure_prediction/decoder_survey/bpiter_control",
    )
    parser.add_argument("--out", default=None, help="Write the report here as well.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels = _load(args.dir)
    text = report(levels)
    print(text)
    out_path = args.out or os.path.join(args.dir, "control_report.txt")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write(text + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
