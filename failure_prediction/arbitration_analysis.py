"""Mechanism analysis of decoder complementarity: what distinguishes WC shots?

Reads the paired shards written by `failure_prediction.arbitration` and asks
whether the fast-path structural quantities separate the four decoder-outcome
classes, in particular:

    WC vs CC   can beneficial escalation be spotted among the easy majority?
    WC vs CW   can beneficial escalation be told apart from harmful escalation?
    WC vs WW   can it be told apart from futile escalation?

Separability is reported as the rank statistic AUC, which for a single feature
is the Mann-Whitney U statistic rescaled, together with Cliff's delta
(= 2*AUC - 1) as the effect size. p-values are Benjamini-Hochberg adjusted over
every feature-by-contrast test performed.

The logistic-regression and shallow-tree probes at the end are diagnostics for
whether the combined fast-path information carries signal at all. They are not a
proposed router.

Usage:
    python -m failure_prediction.arbitration_analysis \
        --shards outputs/failure_prediction/arbitration_mechanism/shards \
        --out-dir outputs/failure_prediction/arbitration_mechanism
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from math import erfc, sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

from failure_prediction.arbitration import CLASS_NAMES
from failure_prediction.metrics import precision_recall_auc, roc_auc

CC, WC, CW, WW = 0, 1, 2, 3

# Grouping is what makes the mechanism question answerable. "Severity" quantities
# only say how corrupted a shot is; the boundary quantities are the ones the
# candidate-space hypothesis actually predicts. Asking whether boundary adds
# anything on top of severity is the test, not the raw ranking.
GROUPS: Dict[str, List[str]] = {
    "context": ["syndrome_weight", "bp_converged", "bp_iter"],
    "llr_dist": [
        "llr_min", "llr_p01", "llr_p05", "llr_median", "llr_mean", "llr_std",
        "frac_llr_negative", "llr_entropy",
        "n_lowconf_0p5", "n_lowconf_1", "n_lowconf_2",
    ],
    "osd0_cand": [
        "osd0_weight_prior", "osd0_hamming", "osd0_mean_logp",
        "hamming_bp_osd0", "n_bp_negative",
    ],
    "boundary": [
        "boundary_llr", "boundary_pos", "n_dep_rejected", "frac_dep_rejected",
        "gap_after_boundary", "dep_rejected_min_llr",
        "llr_free_0", "llr_free_1", "llr_free_2", "free01_span",
        "boundary_density_0p5", "boundary_density_1", "boundary_density_2",
    ],
}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def load_shards(pattern: str) -> Dict[str, np.ndarray]:
    paths = sorted(glob.glob(os.path.join(pattern, "*.npz")))
    if not paths:
        raise SystemExit(f"No shards under {pattern}")
    feats, outs, meta = [], [], None
    names: Optional[List[str]] = None
    for path in paths:
        d = np.load(path, allow_pickle=True)
        if names is None:
            names = [str(x) for x in d["feature_names"]]
            meta = {
                "code": str(d["code"]),
                "p": float(d["p"]),
                "rounds": int(d["rounds"]),
                "max_iter": int(d["max_iter"]),
                "rank": int(d["rank"]),
                "n_cols": int(d["n_cols"]),
                "fast_decoder": str(d["fast_decoder"]),
                "strong_decoder": str(d["strong_decoder"]),
            }
        elif [str(x) for x in d["feature_names"]] != names:
            raise SystemExit(f"Feature-name mismatch in {path}")
        feats.append(d["features"])
        outs.append(d["outcome"])
    return {
        "features": np.concatenate(feats, axis=0),
        "outcome": np.concatenate(outs, axis=0),
        "names": names,
        "meta": meta,
        "n_shards": len(paths),
    }


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """AUC of a-over-b and a normal-approximation two-sided p-value, tie-corrected."""

    na, nb = a.size, b.size
    if na == 0 or nb == 0:
        return float("nan"), float("nan")
    joint = np.concatenate([a, b])
    order = np.argsort(joint, kind="mergesort")
    ranks = np.empty(joint.size, dtype=np.float64)
    sorted_vals = joint[order]
    i = 0
    while i < joint.size:
        j = i
        while j + 1 < joint.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_a = ranks[:na].sum()
    u_a = r_a - na * (na + 1) / 2.0
    auc = u_a / (na * nb)

    _, counts = np.unique(joint, return_counts=True)
    tie_term = np.sum(counts**3 - counts)
    n = na + nb
    var = na * nb / 12.0 * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        return float(auc), float("nan")
    z = (u_a - na * nb / 2.0) / sqrt(var)
    return float(auc), float(erfc(abs(z) / sqrt(2.0)))


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    if not ok.any():
        return out
    sub = p[ok]
    m = sub.size
    order = np.argsort(sub)
    ranked = sub[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.clip(adj, 0, 1)
    out[ok] = res
    return out


def contrast_table(
    features: np.ndarray, outcome: np.ndarray, names: List[str], pos: int, neg: int
) -> List[Dict[str, object]]:
    rows = []
    for idx, name in enumerate(names):
        col = features[:, idx]
        a = col[(outcome == pos) & ~np.isnan(col)]
        b = col[(outcome == neg) & ~np.isnan(col)]
        auc, pval = _mannwhitney(a, b)
        rows.append(
            {
                "feature": name,
                "n_pos": int(a.size),
                "n_neg": int(b.size),
                "median_pos": float(np.median(a)) if a.size else float("nan"),
                "median_neg": float(np.median(b)) if b.size else float("nan"),
                "auc": auc,
                "cliffs_delta": 2 * auc - 1 if not np.isnan(auc) else float("nan"),
                "p_raw": pval,
            }
        )
    return rows


def decile_curve(
    values: np.ndarray, is_pos: np.ndarray, n_bins: int = 10
) -> List[Dict[str, float]]:
    ok = ~np.isnan(values)
    v, y = values[ok], is_pos[ok]
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    out = []
    for i in range(n_bins):
        if edges[i + 1] <= edges[i]:
            continue
        m = (v >= edges[i]) & (v < edges[i + 1])
        n = int(m.sum())
        if n == 0:
            continue
        k = int(y[m].sum())
        lo, hi = wilson(k, n)
        out.append(
            {
                "bin": i,
                "lo_edge": float(edges[i]),
                "hi_edge": float(edges[i + 1]),
                "n": n,
                "k": k,
                "rate": k / n,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return out


def spearman_matrix(features: np.ndarray, names: List[str], idxs: List[int]) -> np.ndarray:
    sub = features[:, idxs]
    ok = ~np.isnan(sub).any(axis=1)
    sub = sub[ok]
    ranks = np.apply_along_axis(
        lambda c: np.argsort(np.argsort(c, kind="mergesort"), kind="mergesort"), 0, sub
    ).astype(float)
    ranks -= ranks.mean(axis=0, keepdims=True)
    denom = np.sqrt((ranks**2).sum(axis=0))
    denom[denom == 0] = 1.0
    return (ranks.T @ ranks) / np.outer(denom, denom)


def redundancy_groups(
    features: np.ndarray, names: List[str], threshold: float = 0.999
) -> List[List[str]]:
    """Cluster quantities that are monotone transforms of one another.

    `boundary_pos`, `n_dep_rejected` and `frac_dep_rejected` differ only by the
    constant rank(H), so they carry one piece of information between them and
    would otherwise be counted three times in any ranking.
    """

    idxs = list(range(len(names)))
    corr = spearman_matrix(features, names, idxs)
    seen, clusters = set(), []
    for i in range(len(names)):
        if i in seen:
            continue
        members = [j for j in range(len(names)) if abs(corr[i, j]) >= threshold]
        for j in members:
            seen.add(j)
        if len(members) > 1:
            clusters.append([names[j] for j in members])
    return clusters


def collapse_ranking(
    rows: List[Dict[str, object]], clusters: List[List[str]], top: int
) -> List[Dict[str, object]]:
    """Keep the best representative of each redundant cluster in a ranking."""

    rep_of = {}
    for cluster in clusters:
        for member in cluster:
            rep_of[member] = cluster[0]
    ordered = sorted(
        rows,
        key=lambda r: -abs(r["auc"] - 0.5) if not np.isnan(r["auc"]) else 0,
    )
    out, used = [], set()
    for row in ordered:
        key = rep_of.get(row["feature"], row["feature"])
        if key in used:
            continue
        used.add(key)
        out.append(row)
        if len(out) >= top:
            break
    return out


def nested_group_probes(
    features: np.ndarray, names: List[str], y: np.ndarray, seed: int = 0
) -> List[Dict[str, object]]:
    """Does boundary geometry add anything once severity is already known?

    The candidate-space hypothesis predicts it should. If the severity-only model
    already achieves the same AUROC, then WC shots are simply the noisier ones and
    the boundary reading is not supported at instance level.
    """

    ladder = [
        ("context", ["context"]),
        ("context+llr", ["context", "llr_dist"]),
        ("context+llr+osd0 (no boundary)", ["context", "llr_dist", "osd0_cand"]),
        ("all (adds boundary)", ["context", "llr_dist", "osd0_cand", "boundary"]),
        ("boundary only", ["boundary"]),
    ]
    results = []
    for label, groups in ladder:
        cols = [n for g in groups for n in GROUPS[g] if n in names]
        idxs = [names.index(n) for n in cols]
        res = run_probe(features[:, idxs], cols, y, seed=seed)
        res["label"] = label
        res["n_features"] = len(cols)
        results.append(res)
    return results


def stratified_curve(
    features: np.ndarray,
    names: List[str],
    outcome: np.ndarray,
    feature: str,
    stratifier: str,
    n_strata: int = 3,
    n_bins: int = 4,
) -> List[Dict[str, object]]:
    """P(WC | feature) inside strata of a severity proxy.

    A boundary quantity that only tracks severity will go flat once the
    stratifier is held fixed.
    """

    v = features[:, names.index(feature)]
    s = features[:, names.index(stratifier)]
    is_wc = (outcome == WC).astype(float)
    ok = ~np.isnan(v) & ~np.isnan(s)
    v, s, y = v[ok], s[ok], is_wc[ok]
    edges = np.quantile(s, np.linspace(0, 1, n_strata + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    out = []
    for k in range(n_strata):
        m = (s >= edges[k]) & (s < edges[k + 1])
        if m.sum() < 50:
            continue
        curve = decile_curve(v[m], y[m], n_bins=n_bins)
        out.append(
            {
                "stratum": k,
                "stratifier": stratifier,
                "lo": float(edges[k]),
                "hi": float(edges[k + 1]),
                "n": int(m.sum()),
                "curve": curve,
            }
        )
    return out


def routing_sweep(
    score: np.ndarray, outcome: np.ndarray, fractions: Tuple[float, ...]
) -> List[Dict[str, float]]:
    """What a threshold on one statistic would buy, before building any router.

    Escalated shots take the strong decoder's outcome, the rest keep the fast
    one, so the achievable LER is bounded below by the CW population: escalating
    everything still pays for the strong decoder's own failures.
    """

    n = outcome.size
    fast_wrong = np.isin(outcome, [WC, WW])
    strong_wrong = np.isin(outcome, [CW, WW])
    is_wc = outcome == WC
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-s, kind="mergesort")
    rows = []
    for frac in fractions:
        k = int(round(frac * n))
        sel = np.zeros(n, dtype=bool)
        sel[order[:k]] = True
        ler = float((fast_wrong & ~sel).sum() + (strong_wrong & sel).sum()) / n
        rows.append(
            {
                "escalated_fraction": k / n,
                "ler": ler,
                "wc_recall": float((sel & is_wc).sum() / max(is_wc.sum(), 1)),
                "cw_escalated": int((sel & (outcome == CW)).sum()),
                "precision_wc": float((sel & is_wc).sum() / max(k, 1)),
            }
        )
    return rows


def run_probe(
    features: np.ndarray, names: List[str], y: np.ndarray, seed: int = 0
) -> Dict[str, object]:
    """Cross-validated logistic regression and a depth-3 tree, as a signal check."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ModuleNotFoundError:
        return {"error": "sklearn unavailable"}

    ok = ~np.isnan(features).any(axis=1)
    X, yy = features[ok], y[ok]
    if yy.sum() < 20 or (1 - yy).sum() < 20:
        return {"error": f"insufficient class counts: pos={int(yy.sum())}"}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(yy.size)
    for tr, te in skf.split(X, yy):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5),
        )
        model.fit(X[tr], yy[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]

    full = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    ).fit(X, yy)
    coefs = full[-1].coef_.ravel()
    top = np.argsort(-np.abs(coefs))[:10]

    tree = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=max(20, int(0.01 * yy.size)),
        class_weight="balanced", random_state=seed,
    ).fit(X, yy)
    oof_tree = np.zeros(yy.size)
    for tr, te in skf.split(X, yy):
        t = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=max(20, int(0.01 * yy.size)),
            class_weight="balanced", random_state=seed,
        ).fit(X[tr], yy[tr])
        oof_tree[te] = t.predict_proba(X[te])[:, 1]

    return {
        "n": int(yy.size),
        "n_pos": int(yy.sum()),
        "base_rate": float(yy.mean()),
        "lr_auroc": float(roc_auc(yy, oof)),
        "lr_auprc": float(precision_recall_auc(yy, oof)),
        "tree_auroc": float(roc_auc(yy, oof_tree)),
        "tree_auprc": float(precision_recall_auc(yy, oof_tree)),
        "lr_top_coefficients": [
            {"feature": names[i], "coef": float(coefs[i])} for i in top
        ],
        "tree_rules": export_text(
            tree, feature_names=list(names), max_depth=3, decimals=3
        ),
    }


def make_figures(
    features: np.ndarray,
    outcome: np.ndarray,
    names: List[str],
    top_feats: List[str],
    out_dir: str,
    meta: Dict[str, object],
) -> List[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    counts = np.bincount(outcome, minlength=4)
    total = counts.sum()

    # Figure 1: outcome composition.
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = ["#9bb7d4", "#d1495b", "#edae49", "#7a7a7a"]
    axes[0].bar(CLASS_NAMES, counts, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_ylim(top=counts.max() * 6)
    axes[0].set_ylabel("shots (log)")
    for i, c in enumerate(counts):
        axes[0].text(i, c, f"{c}\n{100 * c / total:.2f}%", ha="center", va="bottom", fontsize=8)
    axes[0].set_title(f"Outcome composition, N={total}")

    disc = [counts[WC], counts[CW]]
    lo_wc, hi_wc = wilson(int(counts[WC]), int(total))
    lo_cw, hi_cw = wilson(int(counts[CW]), int(total))
    axes[1].bar(["WC\n(beneficial)", "CW\n(harmful)"], disc, color=[colors[WC], colors[CW]])
    axes[1].errorbar(
        [0, 1], disc,
        yerr=[[disc[0] - lo_wc * total, disc[1] - lo_cw * total],
              [hi_wc * total - disc[0], hi_cw * total - disc[1]]],
        fmt="none", ecolor="k", capsize=4,
    )
    axes[1].set_ylabel("shots")
    axes[1].set_title("Discordant populations (95% CI)")
    fig.suptitle(
        f"{meta['code']}  p={meta['p']:g}  r={meta['rounds']}  "
        f"fast={meta['fast_decoder']}  strong={meta['strong_decoder']}",
        fontsize=9,
    )
    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_outcome_composition.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    # Figure 2: ECDF of the most separating quantities, by class.
    show = top_feats[:6]
    ncol = 3
    nrow = int(np.ceil(len(show) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False)
    for ax, feat in zip(axes.ravel(), show):
        idx = names.index(feat)
        for cls in (CC, WC, CW, WW):
            v = features[outcome == cls, idx]
            v = v[~np.isnan(v)]
            if v.size == 0:
                continue
            vs = np.sort(v)
            ax.step(
                vs, np.arange(1, vs.size + 1) / vs.size,
                where="post", color=colors[cls],
                label=f"{CLASS_NAMES[cls]} (n={vs.size})",
                lw=2.0 if cls in (WC, CW) else 1.2,
                alpha=0.9 if cls in (WC, CW) else 0.7,
            )
        ax.set_xlabel(feat, fontsize=8)
        ax.set_ylabel("ECDF", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    for ax in axes.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle("Fast-path structural quantities by decoder-outcome class", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_structure_by_class.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def make_figure3(
    features: np.ndarray,
    outcome: np.ndarray,
    names: List[str],
    feats: List[str],
    out_dir: str,
) -> Optional[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not feats:
        return None
    is_wc = (outcome == WC).astype(float)
    fig, axes = plt.subplots(1, len(feats), figsize=(4.6 * len(feats), 3.6), squeeze=False)
    for ax, feat in zip(axes.ravel(), feats):
        curve = decile_curve(features[:, names.index(feat)], is_wc)
        centres = [0.5 * (c["lo_edge"] + c["hi_edge"]) for c in curve]
        rates = [c["rate"] for c in curve]
        lo = [c["rate"] - c["ci_lo"] for c in curve]
        hi = [c["ci_hi"] - c["rate"] for c in curve]
        ax.errorbar(centres, rates, yerr=[lo, hi], fmt="o-", capsize=3, color="#d1495b")
        ax.axhline(is_wc.mean(), ls="--", c="k", lw=1, label="base rate")
        ax.set_xlabel(feat)
        ax.set_ylabel("P(beneficial escalation | decile)")
        ax.legend(fontsize=8)
    fig.suptitle("Probability of beneficial escalation vs fast-path statistic", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_escalation_probability.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", default="outputs/failure_prediction/arbitration_mechanism/shards"
    )
    parser.add_argument("--out-dir", default="outputs/failure_prediction/arbitration_mechanism")
    parser.add_argument("--top", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    data = load_shards(args.shards)
    features, outcome, names, meta = (
        data["features"], data["outcome"], data["names"], data["meta"]
    )
    total = outcome.size
    counts = np.bincount(outcome, minlength=4)

    lines: List[str] = [
        "Decoder arbitration mechanism analysis",
        f"  {meta['code']}  p={meta['p']:g}  rounds={meta['rounds']}  "
        f"BP max_iter={meta['max_iter']}",
        f"  fast={meta['fast_decoder']}  strong={meta['strong_decoder']}  "
        f"(both read from one OSD-CS decode)",
        f"  H rank={meta['rank']} of {meta['n_cols']} columns, so OSD-CS-2 sweeps "
        f"{meta['n_cols'] - meta['rank'] + 1} candidates per shot",
        f"  shots={total} from {data['n_shards']} independent shards",
        "",
        "1) Outcome composition",
        f"  {'class':6s} {'count':>7s} {'fraction':>10s} {'95% CI':>20s}",
    ]
    for cls in range(4):
        lo, hi = wilson(int(counts[cls]), int(total))
        lines.append(
            f"  {CLASS_NAMES[cls]:6s} {counts[cls]:7d} {counts[cls] / total:10.5f} "
            f"[{lo:.5f},{hi:.5f}]"
        )
    ler_fast = (counts[WC] + counts[WW]) / total
    ler_strong = (counts[CW] + counts[WW]) / total
    lines += [
        f"  fast LER   = {ler_fast:.5f}   strong LER = {ler_strong:.5f}",
        f"  beneficial escalation WC = {counts[WC]}, harmful escalation CW = {counts[CW]},"
        f" ratio = {counts[WC] / max(counts[CW], 1):.1f}",
        f"  oracle routing would escalate {counts[WC] / total:.4f} of shots to reach "
        f"LER {(counts[CW] + counts[WW]) / total:.5f}"
        if counts[WC]
        else "",
    ]

    n_conv = int(np.count_nonzero(features[:, names.index("bp_converged")] > 0.5))
    lines += [
        "",
        f"  BP converged on {n_conv} shots ({100 * n_conv / total:.2f}%); OSD never runs "
        "there, so boundary quantities are undefined and those shots drop out of the",
        "  structural contrasts below.",
    ]

    contrasts = [("WC", WC, "CC", CC), ("WC", WC, "CW", CW), ("WC", WC, "WW", WW)]
    all_rows: Dict[str, List[Dict[str, object]]] = {}
    pooled_p: List[float] = []
    for pos_name, pos, neg_name, neg in contrasts:
        rows = contrast_table(features, outcome, names, pos, neg)
        all_rows[f"{pos_name}_vs_{neg_name}"] = rows
        pooled_p.extend([r["p_raw"] for r in rows])
    adj = benjamini_hochberg(np.array(pooled_p, dtype=float))
    cursor = 0
    for key in all_rows:
        for row in all_rows[key]:
            row["p_bh"] = float(adj[cursor])
            cursor += 1

    lines += [
        "",
        "2) Separability of each fast-path quantity (AUC = P(pos ranks above neg))",
        f"   p-values Benjamini-Hochberg adjusted over all "
        f"{len(pooled_p)} feature-by-contrast tests",
    ]
    clusters = redundancy_groups(features, names)
    if clusters:
        lines += ["", "  Redundant quantities collapsed to one representative each:"]
        for cluster in clusters:
            lines.append(f"    {cluster[0]}  <=>  {', '.join(cluster[1:])}")

    for key in all_rows:
        rows = collapse_ranking(all_rows[key], clusters, args.top)
        lines += [
            "",
            f"  {key}   (n_pos={rows[0]['n_pos']}, n_neg={rows[0]['n_neg']})",
            f"    {'feature':24s} {'AUC':>7s} {'delta':>7s} {'med_pos':>11s} "
            f"{'med_neg':>11s} {'p_BH':>10s}",
        ]
        for row in rows:
            lines.append(
                f"    {row['feature']:24s} {row['auc']:7.4f} {row['cliffs_delta']:+7.3f} "
                f"{row['median_pos']:11.4g} {row['median_neg']:11.4g} {row['p_bh']:10.3g}"
            )

    top_feats = [r["feature"] for r in collapse_ranking(all_rows["WC_vs_CC"], clusters, args.top)]
    # Figure 2 has to speak to the hypothesis, not just to severity, so make sure
    # the leading boundary quantities appear even if severity dominates the ranking.
    boundary_ranked = collapse_ranking(
        [r for r in all_rows["WC_vs_CW"] if r["feature"] in GROUPS["boundary"]],
        clusters,
        2,
    )
    fig2_feats = top_feats[:4] + [
        r["feature"] for r in boundary_ranked if r["feature"] not in top_feats[:4]
    ]

    idxs = [names.index(f) for f in top_feats]
    corr = spearman_matrix(features, names, idxs)
    lines += ["", "3) Spearman correlation among the top quantities (redundancy check)", ""]
    lines.append("    " + " " * 24 + " ".join(f"{f[:9]:>10s}" for f in top_feats))
    for i, f in enumerate(top_feats):
        lines.append(
            f"    {f:24s} " + " ".join(f"{corr[i, j]:10.3f}" for j in range(len(top_feats)))
        )

    lines += ["", "4) Beneficial-escalation rate by decile of the leading quantities"]
    is_wc = (outcome == WC).astype(float)
    for feat in top_feats[:3]:
        curve = decile_curve(features[:, names.index(feat)], is_wc)
        lines += [
            "",
            f"  {feat}",
            f"    {'range':>26s} {'n':>7s} {'WC':>6s} {'rate':>8s} {'95% CI':>18s}",
        ]
        for c in curve:
            lines.append(
                f"    [{c['lo_edge']:11.4g},{c['hi_edge']:11.4g}) {c['n']:7d} {c['k']:6d} "
                f"{c['rate']:8.4f} [{c['ci_lo']:.4f},{c['ci_hi']:.4f}]"
            )

    lines += ["", "5) Combined fast-path signal (diagnostic probe, not a proposed router)"]
    probes = {
        "WC_vs_rest": run_probe(features, names, (outcome == WC).astype(int)),
        "WC_vs_CC": None,
        "WC_vs_CW": None,
    }
    mask = np.isin(outcome, [WC, CC])
    probes["WC_vs_CC"] = run_probe(
        features[mask], names, (outcome[mask] == WC).astype(int)
    )
    mask = np.isin(outcome, [WC, CW])
    probes["WC_vs_CW"] = run_probe(
        features[mask], names, (outcome[mask] == WC).astype(int)
    )
    for key, res in probes.items():
        lines.append("")
        if not res or "error" in res:
            lines.append(f"  {key}: {res.get('error') if res else 'not run'}")
            continue
        lines.append(
            f"  {key}: n={res['n']} pos={res['n_pos']} base={res['base_rate']:.4f}  "
            f"LR AUROC={res['lr_auroc']:.4f} AUPRC={res['lr_auprc']:.4f}  "
            f"tree AUROC={res['tree_auroc']:.4f} AUPRC={res['tree_auprc']:.4f}"
        )
        if key == "WC_vs_rest":
            lines.append("    strongest standardized logistic coefficients:")
            for item in res["lr_top_coefficients"][:8]:
                lines.append(f"      {item['feature']:24s} {item['coef']:+8.4f}")
            lines.append("    depth-3 tree:")
            lines += ["      " + ln for ln in res["tree_rules"].splitlines()]

    lines += [
        "",
        "6) Does boundary geometry add anything beyond severity?",
        "   If the 'no boundary' and 'all' rows agree, WC shots are just the noisier",
        "   ones and the candidate-space reading gains no instance-level support.",
    ]
    nested: Dict[str, object] = {}
    for label, mask in (
        ("WC_vs_rest", np.ones(total, dtype=bool)),
        ("WC_vs_CW", np.isin(outcome, [WC, CW])),
    ):
        res_list = nested_group_probes(
            features[mask], names, (outcome[mask] == WC).astype(int)
        )
        nested[label] = res_list
        lines += [
            "",
            f"  {label}",
            f"    {'model':34s} {'#feat':>6s} {'AUROC':>8s} {'AUPRC':>8s}",
        ]
        for res in res_list:
            if "error" in res:
                lines.append(f"    {res['label']:34s} {res['n_features']:6d}  {res['error']}")
                continue
            lines.append(
                f"    {res['label']:34s} {res['n_features']:6d} "
                f"{res['lr_auroc']:8.4f} {res['lr_auprc']:8.4f}"
            )

    lines += [
        "",
        "7) Boundary quantity inside strata of severity",
        "   A boundary reading that merely tracks severity flattens out here.",
    ]
    strat_out: Dict[str, object] = {}
    # syndrome_weight is only a weak severity proxy, so a gradient surviving it
    # proves little. hamming_bp_osd0 is the strongest severity quantity here, and
    # it is the honest thing to condition on.
    for stratifier in ("syndrome_weight", "hamming_bp_osd0"):
        for feat in [r["feature"] for r in boundary_ranked][:2]:
            strata = stratified_curve(features, names, outcome, feat, stratifier)
            strat_out[f"{feat}|{stratifier}"] = strata
            lines.append("")
            lines.append(f"  {feat} within terciles of {stratifier}")
            for st in strata:
                lines.append(
                    f"    {stratifier} in [{st['lo']:.0f},{st['hi']:.0f})  n={st['n']}"
                )
                for c in st["curve"]:
                    lines.append(
                        f"      [{c['lo_edge']:11.4g},{c['hi_edge']:11.4g}) n={c['n']:6d} "
                        f"WC={c['k']:4d} rate={c['rate']:.4f} "
                        f"[{c['ci_lo']:.4f},{c['ci_hi']:.4f}]"
                    )

    lines += [
        "",
        "8) What a threshold on a single statistic would buy",
        "   Escalated shots take the strong outcome, the rest keep the fast one.",
        f"   fast-only LER = {(np.isin(outcome, [WC, WW])).mean():.5f}, "
        f"strong-only LER = {(np.isin(outcome, [CW, WW])).mean():.5f}",
    ]
    fractions = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
    sweep_out: Dict[str, object] = {}
    for feat in [top_feats[0], "osd0_weight_prior"]:
        if feat not in names:
            continue
        rows = routing_sweep(features[:, names.index(feat)], outcome, fractions)
        sweep_out[feat] = rows
        lines += [
            "",
            f"  escalate on high {feat}",
            f"    {'esc frac':>9s} {'LER':>9s} {'WC recall':>10s} "
            f"{'WC prec':>9s} {'CW sent':>8s}",
        ]
        for row in rows:
            lines.append(
                f"    {row['escalated_fraction']:9.3f} {row['ler']:9.5f} "
                f"{row['wc_recall']:10.4f} {row['precision_wc']:9.4f} "
                f"{row['cw_escalated']:8d}"
            )

    lines += [
        "",
        "9) Separability inside the top severity tercile",
        "   Almost every WC shot lives here, so this asks whether anything still",
        "   discriminates once the dominant severity axis is held fixed.",
    ]
    sev = features[:, names.index(top_feats[0])]
    cut = np.nanquantile(sev, 2.0 / 3.0)
    band = sev >= cut
    band_rows = contrast_table(features[band], outcome[band], names, WC, CC)
    band_adj = benjamini_hochberg(np.array([r["p_raw"] for r in band_rows], dtype=float))
    for row, padj in zip(band_rows, band_adj):
        row["p_bh"] = float(padj)
    lines += [
        f"   stratum: {top_feats[0]} >= {cut:.4g},  n={int(band.sum())}, "
        f"WC={int((outcome[band] == WC).sum())}, CC={int((outcome[band] == CC).sum())}",
        "",
        f"    {'feature':24s} {'AUC':>7s} {'delta':>7s} {'med_pos':>11s} "
        f"{'med_neg':>11s} {'p_BH':>10s}",
    ]
    for row in collapse_ranking(band_rows, clusters, args.top):
        lines.append(
            f"    {row['feature']:24s} {row['auc']:7.4f} {row['cliffs_delta']:+7.3f} "
            f"{row['median_pos']:11.4g} {row['median_neg']:11.4g} {row['p_bh']:10.3g}"
        )
    band_probe = run_probe(
        features[band][np.isin(outcome[band], [WC, CC])],
        names,
        (outcome[band][np.isin(outcome[band], [WC, CC])] == WC).astype(int),
    )
    if "error" not in band_probe:
        lines.append(
            f"    combined probe within stratum: LR AUROC={band_probe['lr_auroc']:.4f} "
            f"AUPRC={band_probe['lr_auprc']:.4f} (base rate {band_probe['base_rate']:.4f})"
        )

    figs = make_figures(features, outcome, names, fig2_feats, args.out_dir, meta)
    fig3 = make_figure3(features, outcome, names, top_feats[:2], args.out_dir)
    if fig3:
        figs.append(fig3)

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(args.out_dir, "mechanism_report.txt"), "w") as fh:
        fh.write(text + "\n")
    with open(os.path.join(args.out_dir, "mechanism_summary.json"), "w") as fh:
        json.dump(
            {
                "meta": meta,
                "counts": {CLASS_NAMES[c]: int(counts[c]) for c in range(4)},
                "total": int(total),
                "contrasts": {k: v for k, v in all_rows.items()},
                "probes": probes,
                "nested_group_probes": nested,
                "stratified_curves": strat_out,
                "routing_sweep": sweep_out,
                "top_severity_stratum": {
                    "stratifier": top_feats[0],
                    "cut": float(cut),
                    "rows": band_rows,
                    "probe": band_probe,
                },
                "redundant_clusters": clusters,
                "top_features": top_feats,
            },
            fh,
            indent=2,
            default=float,
        )
    print("\nFigures: " + ", ".join(figs))


if __name__ == "__main__":
    main()
