"""Mechanistic analysis of the decoder-gap survey.

Reads the survey JSON files and asks, without running new simulations:

  1. Do OSD-0 and OSD-CS share a logical-error exponent (same effective
     distance) and differ only in prefactor, or does OSD-CS extend the
     effective distance?
  2. How does the OSD-0 -> OSD-CS gap track BP convergence, and how does it
     track code / detector-error-model size?
  3. Is the discordance symmetric, i.e. does the expensive decoder ever lose to
     the cheap one on individual shots?
  4. Does the gain saturate in OSD order?
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import numpy as np


def load_points(survey_dir: str) -> List[Dict[str, Any]]:
    points = []
    for path in sorted(glob.glob(os.path.join(survey_dir, "*.json"))):
        with open(path) as f:
            points.append(json.load(f))
    return points


def _by_name(point: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {row["decoder"]: row for row in point["decoders"]}


def _pair(point: Dict[str, Any], fast: str, strong: str) -> Dict[str, Any] | None:
    for row in point.get("pairs", []):
        if row["fast"] == fast and row["strong"] == strong:
            return row
    return None


def local_exponents(ps: np.ndarray, lers: np.ndarray) -> List[float]:
    """d log LER / d log p between consecutive noise points."""

    out = []
    for i in range(len(ps) - 1):
        if lers[i] <= 0 or lers[i + 1] <= 0:
            out.append(float("nan"))
            continue
        out.append(float(np.log(lers[i + 1] / lers[i]) / np.log(ps[i + 1] / ps[i])))
    return out


def analyse(survey_dir: str) -> str:
    points = load_points(survey_dir)
    lines: List[str] = ["Decoder-gap mechanism analysis", "=" * 78, ""]

    # ---------- 1. exponent vs prefactor ----------
    lines.append("1. Logical-error exponent: does OSD-CS extend the effective distance?")
    lines.append("")
    for code in ("bb_72_12_6", "bb_144_12_12"):
        rows = sorted(
            [pt for pt in points if pt["code"] == code],
            key=lambda pt: pt["p"],
        )
        strong_name = "bp50_osdcs2"
        fast_name = "bp20_osd0"
        ps, fast_ler, strong_ler = [], [], []
        for pt in rows:
            names = _by_name(pt)
            if fast_name not in names or strong_name not in names:
                continue
            if names[fast_name]["n_fail"] == 0 or names[strong_name]["n_fail"] == 0:
                continue
            ps.append(pt["p"])
            fast_ler.append(names[fast_name]["ler"])
            strong_ler.append(names[strong_name]["ler"])
        if len(ps) < 2:
            lines.append(f"  {code}: fewer than two points with non-zero failures on both")
            lines.append("")
            continue
        ps_a = np.asarray(ps)
        fe = local_exponents(ps_a, np.asarray(fast_ler))
        se = local_exponents(ps_a, np.asarray(strong_ler))
        lines.append(f"  {code}")
        lines.append(
            f"    {'p range':>16s} {'exp(OSD-0)':>11s} {'exp(OSD-CS)':>12s} {'difference':>11s}"
        )
        for i, (a, b) in enumerate(zip(fe, se)):
            lines.append(
                f"    {ps[i]:.4f}->{ps[i+1]:.4f} {a:11.2f} {b:12.2f} {b - a:11.2f}"
            )
        lines.append(
            f"    mean exponent: OSD-0 {np.nanmean(fe):.2f}, OSD-CS {np.nanmean(se):.2f}"
        )
        ratios = np.asarray(fast_ler) / np.asarray(strong_ler)
        lines.append(
            f"    LER ratio OSD-0/OSD-CS across p: "
            + ", ".join(f"{p:.4f}:{r:.2f}x" for p, r in zip(ps, ratios))
        )
        lines.append("")

    # ---------- 2. gap vs BP convergence and model size ----------
    lines.append("2. Gap versus BP convergence and detector-error-model size")
    lines.append("")
    lines.append(
        f"  {'code':14s} {'p':>7s} {'dets':>6s} {'mechs':>7s} {'BP@20conv':>10s} "
        f"{'LERfast':>10s} {'LERstrong':>10s} {'gap':>7s} {'cost x':>7s}"
    )
    table = []
    for pt in sorted(points, key=lambda q: (q["code"], q["p"])):
        names = _by_name(pt)
        fast = names.get("bp20_osd0") or names.get("bp10_osd0")
        strong = names.get("bp50_osdcs2") or names.get("bp50_osdcs8")
        bp20 = names.get("bp20")
        if not fast or not strong:
            continue
        gap = fast["ler"] / strong["ler"] if strong["ler"] > 0 else float("inf")
        row = {
            "code": pt["code"],
            "p": pt["p"],
            "dets": pt["n_detectors"],
            "mechs": pt["n_error_mechanisms"],
            "conv": bp20["converged_rate"] if bp20 else float("nan"),
            "ler_fast": fast["ler"],
            "ler_strong": strong["ler"],
            "gap": gap,
            "cost_ratio": strong["mean_ms"] / fast["mean_ms"],
        }
        table.append(row)
        gap_s = "inf" if not np.isfinite(gap) else f"{gap:.2f}x"
        lines.append(
            f"  {row['code']:14s} {row['p']:7.4f} {row['dets']:6d} {row['mechs']:7d} "
            f"{row['conv']:10.4f} {row['ler_fast']:10.6f} {row['ler_strong']:10.6f} "
            f"{gap_s:>7s} {row['cost_ratio']:7.1f}"
        )
    lines.append("")
    finite = [r for r in table if np.isfinite(r["gap"])]
    for code in ("bb_72_12_6", "bb_144_12_12"):
        sub = [r for r in finite if r["code"] == code]
        if len(sub) >= 2:
            lines.append(
                f"  {code}: BP@20 convergence {min(r['conv'] for r in sub):.3f}-"
                f"{max(r['conv'] for r in sub):.3f}, gap "
                f"{min(r['gap'] for r in sub):.2f}x-{max(r['gap'] for r in sub):.2f}x, "
                f"{sub[0]['mechs']} error mechanisms"
            )
    lines.append("")

    # ---------- 3. discordance asymmetry ----------
    lines.append("3. Discordance: does the expensive decoder ever lose on a shot?")
    lines.append("")
    lines.append(
        f"  {'code':14s} {'p':>7s} {'fast wrong only':>16s} {'strong wrong only':>18s} "
        f"{'strong-loss frac':>17s} {'McNemar p':>11s}"
    )
    for pt in sorted(points, key=lambda q: (q["code"], q["p"])):
        for fast in ("bp20_osd0", "bp20_lsd", "bp10_osd0"):
            for strong in ("bp50_osdcs2", "bp50_osdcs8"):
                row = _pair(pt, fast, strong)
                if not row or "n_fast_only_wrong" not in row:
                    continue
                f_only = row["n_fast_only_wrong"]
                s_only = row["n_strong_only_wrong"]
                total = f_only + s_only
                frac = s_only / total if total else float("nan")
                mcn = (row.get("mcnemar") or {}).get("p_value_two_sided", float("nan"))
                lines.append(
                    f"  {pt['code']:14s} {pt['p']:7.4f} {f_only:16d} {s_only:18d} "
                    f"{frac:17.3f} {mcn:11.3g}   [{fast} vs {strong}]"
                )
                break
            else:
                continue
            break
    lines.append("")

    # ---------- 4. saturation in OSD order ----------
    lines.append("4. Does the gain saturate in OSD order?")
    lines.append("")
    lines.append(
        f"  {'code':14s} {'p':>7s} {'osdcs2 fails':>13s} {'osdcs8 fails':>13s} "
        f"{'osdcs2 ms':>10s} {'osdcs8 ms':>10s}"
    )
    for pt in sorted(points, key=lambda q: (q["code"], q["p"])):
        names = _by_name(pt)
        if "bp50_osdcs2" not in names or "bp50_osdcs8" not in names:
            continue
        a, b = names["bp50_osdcs2"], names["bp50_osdcs8"]
        lines.append(
            f"  {pt['code']:14s} {pt['p']:7.4f} {a['n_fail']:13d} {b['n_fail']:13d} "
            f"{a['mean_ms']:10.1f} {b['mean_ms']:10.1f}"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey-dir",
        default="outputs/failure_prediction/decoder_survey",
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    text = analyse(args.survey_dir)
    print(text)
    out = args.out or os.path.join(args.survey_dir, "mechanism_analysis.txt")
    with open(out, "w") as f:
        f.write(text)
