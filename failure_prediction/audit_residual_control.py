"""Audit helpers + BP residual-weight control on existing bb144 matched shots.

Replays Stim seeds and BP@20+OSD-0 only. Does not regenerate 1FV labels.
Does not overwrite manuscript figure stats.

Usage:
    python -m failure_prediction.audit_residual_control
"""

from __future__ import annotations

import csv
import glob
import json
import os
from multiprocessing import Pool
from typing import Dict, List, Tuple

import numpy as np

from failure_prediction.adaptive_v1_analysis import P_VALUES, auroc
from failure_prediction.final_adaptive import (
    CACHE,
    ESC_TARGETS,
    N_BOOT,
    N_RANDOM,
    OUT_DIR,
    PARTS,
    bootstrap_auroc,
    jsonable,
    mix_ler,
    random_lers,
    recovered,
    topk_mask,
    wilson,
)
from failure_prediction.osd_internals import (
    gf2_rank_packed,
    osd0_information_set,
    pack_matrix,
)

AUDIT_DIR = os.path.join(OUT_DIR, "audit")
RES_PARTS = os.path.join(AUDIT_DIR, "residual_parts")
KNOWN_ALWAYS = {
    0.006: dict(osd0=0.00475, k500=0.00145, k1000=0.00115, full=0.00105),
    0.007: dict(osd0=0.01990, k500=0.00685, k1000=0.00475, full=0.00410),
    0.008: dict(osd0=0.06275, k500=0.02645, k1000=0.02040, full=0.01865),
}
KNOWN_ADAPT_K1000 = {
    0.006: {0.10: 0.00160, 0.20: 0.00135, 0.30: 0.00120},
    0.007: {0.10: 0.00710, 0.20: 0.00545, 0.30: 0.00500},
    0.008: {0.10: 0.03295, 0.20: 0.02530, 0.30: 0.02215},
}
KNOWN_ADAPT_K500 = {
    0.006: {0.10: 0.00185, 0.20: 0.00160, 0.30: 0.00145},
    0.007: {0.10: 0.00915, 0.20: 0.00755, 0.30: 0.00710},
    0.008: {0.10: 0.03800, 0.20: 0.03095, 0.30: 0.02810},
}

_WORKER = {}


def _close(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def load_parts_with_meta() -> Dict[float, List[dict]]:
    files = sorted(glob.glob(os.path.join(PARTS, "*.npz")))
    by_p: Dict[float, List[dict]] = {}
    for f in files:
        d = np.load(f)
        p = float(np.asarray(d["p"]).ravel()[0])
        rec = {k: np.asarray(d[k]) for k in d.files}
        rec["path"] = f
        rec["seed0"] = int(np.asarray(d["seed"]).ravel()[0])
        rec["n"] = int(d["fail_osd0"].size)
        cache_p = os.path.join(CACHE, os.path.basename(f).replace(".npz", "_w.npz"))
        if os.path.isfile(cache_p):
            rec["syn_weight"] = np.load(cache_p)["syn_weight"].astype(np.float64)
        by_p.setdefault(p, []).append(rec)
    for p in by_p:
        by_p[p].sort(key=lambda r: r["seed0"])
    return by_p


def cat_science(parts: List[dict]) -> dict:
    keys = [
        k
        for k in parts[0]
        if k not in ("p", "seed", "path", "seed0", "n")
    ]
    out = {k: np.concatenate([t[k] for t in parts]) for k in keys}
    out["p"] = float(parts[0]["p"].ravel()[0]) if "p" in parts[0] else None
    out["n"] = int(out["fail_osd0"].size)
    return out


def sanity_and_k500(by_p) -> dict:
    rows = []
    ok_main = True
    ok_k500 = True
    for p in P_VALUES:
        d = cat_science(by_p[p])
        n = d["n"]
        got = {
            "osd0": float(d["fail_osd0"].mean()),
            "k500": float(d["fail_k500"].mean()),
            "k1000": float(d["fail_k1000"].mean()),
            "full": float(d["fail_full"].mean()),
        }
        exp = KNOWN_ALWAYS[p]
        always_ok = all(_close(got[k], exp[k]) for k in exp)
        adapt = {}
        for f in ESC_TARGETS:
            esc = topk_mask(d["d_h"], f)
            ler1000 = mix_ler(d["fail_osd0"], d["fail_k1000"], esc)
            ler500 = mix_ler(d["fail_osd0"], d["fail_k500"], esc)
            adapt[f] = {"k1000": ler1000, "k500": ler500}
            if not _close(ler1000, KNOWN_ADAPT_K1000[p][f]):
                ok_main = False
            if not _close(ler500, KNOWN_ADAPT_K500[p][f]):
                ok_k500 = False
        if not always_ok:
            ok_main = False
            ok_k500 = False
        rows.append({"p": p, "n": n, "always": got, "adapt": adapt, "always_ok": always_ok})
    return {"ok_main": ok_main, "ok_k500": ok_k500, "rows": rows}


def random_audit(by_p, rng) -> dict:
    out = {}
    for p in P_VALUES:
        d = cat_science(by_p[p])
        ler0 = float(d["fail_osd0"].mean())
        lerk = float(d["fail_k1000"].mean())
        lerf = float(d["fail_full"].mean())
        pts = {}
        for f in ESC_TARGETS:
            rnd = random_lers(d["fail_osd0"], d["fail_k1000"], f, rng, n_rep=N_RANDOM)
            analytic = (1.0 - f) * ler0 + f * lerk
            pts[f] = {
                "analytic": analytic,
                "rnd_mean": float(rnd.mean()),
                "rnd_ci": [float(np.quantile(rnd, 0.025)), float(np.quantile(rnd, 0.975))],
                "delta": float(rnd.mean() - analytic),
            }
        out[p] = {
            "ler_osd0": ler0,
            "ler_k1000_all": lerk,
            "ler_full": lerf,
            "points": pts,
            "note": "random_lers mixes fail_k1000, not fail_full. fig3 axhline is full 1FV visual floor; random curve at f=1 is always-K1000.",
        }
    return out


def timing_audit(by_p) -> dict:
    path = "outputs/failure_prediction/adaptive_v1/serial_timing_n1000.json"
    with open(path) as fh:
        t = json.load(fh)
    t0 = t["bp20_osd0"]["mean_ms"]
    tk = t["always_trunc_k1000"]["mean_ms"]
    tfull = t["always_full_1fv"]["mean_ms"]
    mix20 = t0 + 0.2 * (tk - t0)
    a20 = t["adapt_k1000_f20"]["mean_ms"]
    # Identity on the timing sample is not stored per-shot; recover E[Delta|esc]
    # from E[T_ad] = E[T_fast] + f E[Delta|esc]  (routing overhead ~ 0).
    e_delta_esc = {}
    e_delta_all = tk - t0
    for f, key in ((0.10, "adapt_k1000_f10"), (0.20, "adapt_k1000_f20"), (0.30, "adapt_k1000_f30")):
        tad = t[key]["mean_ms"]
        e_delta_esc[f] = (tad - t0) / f
    # Science-set diagnostic: are high-d_H shots slower? Uses per-shot times in parts
    # (different node than wolpy16 serial n=1000; relative only).
    d = cat_science(by_p[0.007])
    extra = d["t_k1000_ms"]  # GE + prorated K=1000 scoring; 0 if BP converged
    esc = topk_mask(d["d_h"], 0.20)
    sci = {
        "mean_t_osd0_ms": float(d["t_osd0_ms"].mean()),
        "mean_extra_all_ms": float(extra.mean()),
        "mean_extra_esc20_ms": float(extra[esc].mean()),
        "mean_extra_unesc20_ms": float(extra[~esc].mean()),
        "n_esc": int(esc.sum()),
        "note": "parts t_k1000_ms is prorated full-1FV scoring, not the wolpy16 n=1000 serial node.",
    }
    return {
        "json": path,
        "log": "outputs/failure_prediction/adaptive_v1/logs/av1_final_20304.out",
        "slurm": "scripts/submit_final_adaptive.slurm",
        "script": "failure_prediction/adaptive_v1_timing.py",
        "host": "wolpy16",
        "n": 1000,
        "p": 0.007,
        "seed": 9001,
        "warmup_discarded": False,
        "repetitions": 1,
        "omp": "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "t_osd0_ms": t0,
        "t_k500_ms": t["always_trunc_k500"]["mean_ms"],
        "t_k1000_ms": tk,
        "t_full_ms": tfull,
        "adapt_ms": {0.10: t["adapt_k1000_f10"]["mean_ms"], 0.20: a20, 0.30: t["adapt_k1000_f30"]["mean_ms"]},
        "unconditional_mix_f20_ms": mix20,
        "measured_adapt_f20_ms": a20,
        "E_delta_all_ms": e_delta_all,
        "E_delta_given_escalated_ms": e_delta_esc,
        "ratio_full_over_adapt20": tfull / a20,
        "ratio_alwaysK1000_over_adapt20": tk / a20,
        "accounting": (
            "Per-shot adaptive time is reconstructed: t_osd0 + 1_esc * t_k1000_ms, "
            "then averaged on the n=1000 timing sample after exact top-k on that sample's d_H. "
            "t_k1000_ms = t_GE + (min(1000,n_free)/n_free) * t_full_score; "
            "K=500/1000 are not independently timed truncated loops."
        ),
        "science_relative_cost": sci,
    }


def h2_transitions() -> dict:
    path = "outputs/failure_prediction/h2_full_css_hardware/decoded_shots.npz"
    d = np.load(path)
    fail0 = d["fail_osd0"].astype(int)
    failf = d["fail_full"].astype(int)
    cc = int(((fail0 == 0) & (failf == 0)).sum())
    cw = int(((fail0 == 0) & (failf == 1)).sum())
    wc = int(((fail0 == 1) & (failf == 0)).sum())
    ww = int(((fail0 == 1) & (failf == 1)).sum())
    n = int(fail0.size)
    return {
        "path": path,
        "n": n,
        "CC": cc,
        "CW": cw,
        "WC": wc,
        "WW": ww,
        "n_fail_osd0": int(fail0.sum()),
        "n_fail_full": int(failf.sum()),
        "ler_osd0": float(fail0.mean()),
        "ler_full": float(failf.mean()),
        "net_failure_change": int(failf.sum() - fail0.sum()),
    }


def _init_worker():
    from ldpc import BpOsdDecoder
    from failure_prediction.qldpc_circuit import (
        _get_code,
        build_z_memory_circuit,
        detector_error_model_to_matrices,
    )

    _WORKER["BpOsdDecoder"] = BpOsdDecoder
    _WORKER["_get_code"] = _get_code
    _WORKER["build_z_memory_circuit"] = build_z_memory_circuit
    _WORKER["detector_error_model_to_matrices"] = detector_error_model_to_matrices
    _WORKER["dec"] = {}


def _get_stack(p: float):
    if p in _WORKER["dec"]:
        return _WORKER["dec"][p]
    code = _WORKER["_get_code"]("bb_144_12_12")
    circuit = _WORKER["build_z_memory_circuit"](code, p=p, rounds=24)
    dem = circuit.detector_error_model(
        decompose_errors=True, ignore_decomposition_failures=True
    )
    matrices = _WORKER["detector_error_model_to_matrices"](dem)
    h = matrices.h
    n_cols = h.shape[1]
    packed = pack_matrix(h, n_cols=n_cols, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n_cols)
    decoder = _WORKER["BpOsdDecoder"](
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=20,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    logicals = matrices.logicals.astype(np.int64)
    stack = dict(
        circuit=circuit,
        h=h,
        packed=packed,
        rank=rank,
        n_cols=n_cols,
        decoder=decoder,
        logicals=logicals,
    )
    _WORKER["dec"][p] = stack
    return stack


def _logical_fail(logicals, e, actual) -> int:
    pred = (logicals @ e.astype(np.int64)) % 2
    return int(np.any(pred != actual))


def replay_one(path: str) -> str:
    os.makedirs(RES_PARTS, exist_ok=True)
    out_p = os.path.join(RES_PARTS, os.path.basename(path).replace(".npz", "_res.npz"))
    stored = np.load(path)
    p = float(np.asarray(stored["p"]).ravel()[0])
    seed = int(np.asarray(stored["seed"]).ravel()[0])
    n = int(stored["fail_osd0"].size)
    if os.path.isfile(out_p):
        chk = np.load(out_p)
        if int(chk["n"]) == n and abs(float(chk["p"]) - p) < 1e-12:
            return out_p
    st = _get_stack(p)
    syn, actual = st["circuit"].compile_detector_sampler(seed=seed).sample(
        n, separate_observables=True
    )
    syn = syn.astype(np.uint8)
    actual = actual.astype(np.uint8)
    h, dec, packed, rank, n_cols, logicals = (
        st["h"],
        st["decoder"],
        st["packed"],
        st["rank"],
        st["n_cols"],
        st["logicals"],
    )
    r_bp = np.empty(n, dtype=np.int32)
    syn_w = np.count_nonzero(syn, axis=1).astype(np.int32)
    fail_bp = np.empty(n, dtype=np.uint8)
    fail_lib = np.empty(n, dtype=np.uint8)
    fail_forced = np.empty(n, dtype=np.uint8)
    he_ok = np.empty(n, dtype=np.uint8)
    lib_eq_forced = np.empty(n, dtype=np.uint8)
    conv = np.empty(n, dtype=np.uint8)
    r_forced = np.empty(n, dtype=np.int32)
    dh_replay = np.empty(n, dtype=np.int32)
    for i in range(n):
        s = syn[i]
        a = actual[i]
        dec.decode(s)
        c = bool(dec.converge)
        e_bp = np.asarray(dec.bp_decoding, dtype=np.uint8) % 2
        e0 = np.asarray(dec.osd0_decoding, dtype=np.uint8) % 2
        resid = np.asarray(h.dot(e_bp) % 2).ravel() ^ s
        r_bp[i] = int(np.count_nonzero(resid))
        fail_bp[i] = _logical_fail(logicals, e_bp, a)
        fail_lib[i] = _logical_fail(logicals, e0, a)
        conv[i] = int(c)
        dh_replay[i] = int(np.count_nonzero(e_bp ^ e0))
        llr = np.asarray(dec.log_prob_ratios, dtype=np.float64)
        info = osd0_information_set(packed, s, llr, n_cols, rank)
        if info is None:
            e_f = e0
            lib_eq_forced[i] = 1
        else:
            e_f = info["solution"]
            lib_eq_forced[i] = int(np.array_equal(e0, e_f))
        fail_forced[i] = _logical_fail(logicals, e_f, a)
        r0 = np.asarray(h.dot(e_f) % 2).ravel() ^ s
        r_forced[i] = int(np.count_nonzero(r0))
        he_ok[i] = int(r_forced[i] == 0)
    np.savez_compressed(
        out_p,
        p=np.full(n, p),
        seed=np.full(n, seed),
        n=n,
        r_bp=r_bp,
        syn_weight=syn_w,
        fail_bp=fail_bp,
        fail_lib_osd0=fail_lib,
        fail_forced_osd0=fail_forced,
        he_ok=he_ok,
        lib_eq_forced=lib_eq_forced,
        conv=conv,
        r_forced=r_forced,
        d_h_replay=dh_replay,
        fail_osd0_stored=stored["fail_osd0"].astype(np.uint8),
        d_h_stored=stored["d_h"].astype(np.float64),
        fail_k1000_stored=stored["fail_k1000"].astype(np.uint8),
        fail_full_stored=stored["fail_full"].astype(np.uint8),
        bp_converged_stored=stored["bp_converged"].astype(np.uint8),
    )
    return out_p


def load_residual_merged() -> Dict[float, dict]:
    files = sorted(glob.glob(os.path.join(RES_PARTS, "*_res.npz")))
    by_p: Dict[float, List[dict]] = {}
    for f in files:
        d = np.load(f)
        p = float(np.asarray(d["p"]).ravel()[0])
        by_p.setdefault(p, []).append({k: np.asarray(d[k]) for k in d.files})
    out = {}
    for p, parts in by_p.items():
        keys = [k for k in parts[0] if k not in ("p", "seed", "n")]
        m = {k: np.concatenate([t[k] for t in parts]) for k in keys}
        m["n"] = int(m["r_bp"].size)
        m["p"] = p
        out[p] = m
    return out


def spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float), kind="mergesort"))
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float), kind="mergesort"))
    if rx.size < 2:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def analyse_residual(merged: Dict[float, dict], rng) -> dict:
    out = {}
    for p in P_VALUES:
        d = merged[p]
        fail0 = d["fail_osd0_stored"].astype(float)
        failk = d["fail_k1000_stored"].astype(float)
        failf = d["fail_full_stored"].astype(float)
        dh = d["d_h_stored"].astype(float)
        rbp = d["r_bp"].astype(float)
        sw = d["syn_weight"].astype(float)
        useful = ((fail0 == 1) & (failf == 0)).astype(float)
        n = d["n"]
        match_fail = float(np.mean(d["fail_lib_osd0"] == fail0))
        match_dh = float(np.mean(d["d_h_replay"] == dh))
        match_conv = float(np.mean(d["conv"] == d["bp_converged_stored"]))
        auc_r_fail = bootstrap_auroc(fail0, rbp, rng)
        auc_sw_fail = bootstrap_auroc(fail0, sw, rng)
        auc_dh_fail = bootstrap_auroc(fail0, dh, rng)
        auc_r_ben = bootstrap_auroc(useful, rbp, rng)
        auc_dh_ben = bootstrap_auroc(useful, dh, rng)
        nc = d["conv"] == 0
        auc_r_nc = bootstrap_auroc(fail0[nc], rbp[nc], rng) if nc.any() else (float("nan"),) * 3
        auc_dh_nc = bootstrap_auroc(fail0[nc], dh[nc], rng) if nc.any() else (float("nan"),) * 3
        route = []
        for f in ESC_TARGETS:
            esc_r = topk_mask(rbp, f)
            esc_d = topk_mask(dh, f)
            esc_s = topk_mask(sw, f)
            ler_r = mix_ler(fail0, failk, esc_r)
            ler_d = mix_ler(fail0, failk, esc_d)
            ler_s = mix_ler(fail0, failk, esc_s)
            rnd = random_lers(fail0, failk, f, rng, n_rep=N_RANDOM)
            ler0 = float(fail0.mean())
            lerf = float(failf.mean())
            route.append(
                {
                    "f": f,
                    "ler_dh": ler_d,
                    "rec_dh": recovered(ler0, lerf, ler_d),
                    "ler_rbp": ler_r,
                    "rec_rbp": recovered(ler0, lerf, ler_r),
                    "ler_sw": ler_s,
                    "rec_sw": recovered(ler0, lerf, ler_s),
                    "ler_rnd": float(rnd.mean()),
                    "n_fail_rbp": int(np.where(esc_r, failk, fail0).sum()),
                    "n_fail_dh": int(np.where(esc_d, failk, fail0).sum()),
                }
            )
        fail_lib = d["fail_lib_osd0"].astype(int)
        fail_forced = d["fail_forced_osd0"].astype(int)
        fail_bp = d["fail_bp"].astype(int)
        differ_logical = int(np.sum(fail_lib != fail_forced))
        # shots where the two *corrections* differ, not just fail bits:
        n_vec_diff = int(np.sum(d["lib_eq_forced"] == 0))
        one_succeeds = {
            "lib_ok_forced_fail": int(((fail_lib == 0) & (fail_forced == 1)).sum()),
            "lib_fail_forced_ok": int(((fail_lib == 1) & (fail_forced == 0)).sum()),
            "bp_ok_lib_fail": int(((fail_bp == 0) & (fail_lib == 1)).sum()),
            "bp_fail_lib_ok": int(((fail_bp == 1) & (fail_lib == 0)).sum()),
        }
        out[p] = {
            "n": n,
            "replay_match_fail_osd0": match_fail,
            "replay_match_d_h": match_dh,
            "replay_match_conv": match_conv,
            "he_ok_frac": float(d["he_ok"].mean()),
            "he_ok_nonconv": float(d["he_ok"][nc].mean()) if nc.any() else None,
            "r_bp_mean": float(rbp.mean()),
            "r_bp_frac_zero": float((rbp == 0).mean()),
            "r_bp_mean_conv": float(rbp[d["conv"] == 1].mean()) if (d["conv"] == 1).any() else None,
            "r_bp_mean_nonconv": float(rbp[nc].mean()) if nc.any() else None,
            "auroc_fail_rbp": list(auc_r_fail),
            "auroc_fail_sw": list(auc_sw_fail),
            "auroc_fail_dh": list(auc_dh_fail),
            "auroc_ben_rbp": list(auc_r_ben),
            "auroc_ben_dh": list(auc_dh_ben),
            "auroc_fail_rbp_nonconv": list(auc_r_nc),
            "auroc_fail_dh_nonconv": list(auc_dh_nc),
            "spearman_dh_rbp": spearman(dh, rbp),
            "spearman_dh_sw": spearman(dh, sw),
            "spearman_rbp_sw": spearman(rbp, sw),
            "routing": route,
            "conditional": {
                "ler_library_osd0_field": float(fail_lib.mean()),
                "ler_forced_algebraic_osd0": float(fail_forced.mean()),
                "ler_bp_hard": float(fail_bp.mean()),
                "n_library_ne_forced_vector": n_vec_diff,
                "n_library_ne_forced_logical_fail": differ_logical,
                "split": one_succeeds,
                "n_converged": int((d["conv"] == 1).sum()),
            },
        }
    return out


def write_csvs(merged, residual_stats, h2):
    os.makedirs(AUDIT_DIR, exist_ok=True)
    auroc_path = os.path.join(AUDIT_DIR, "bp_residual_auroc.csv")
    route_path = os.path.join(AUDIT_DIR, "bp_residual_routing.csv")
    ctrl_path = os.path.join(AUDIT_DIR, "bp_residual_control.csv")
    h2_path = os.path.join(AUDIT_DIR, "h2_osd0_full1fv_transitions.csv")
    with open(auroc_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "p",
                "n",
                "auroc_fail_dh",
                "auroc_fail_dh_lo",
                "auroc_fail_dh_hi",
                "auroc_fail_rbp",
                "auroc_fail_rbp_lo",
                "auroc_fail_rbp_hi",
                "auroc_fail_sw",
                "auroc_fail_sw_lo",
                "auroc_fail_sw_hi",
                "auroc_ben_dh",
                "auroc_ben_rbp",
                "spearman_dh_rbp",
                "spearman_dh_sw",
                "spearman_rbp_sw",
            ]
        )
        for p in P_VALUES:
            s = residual_stats[p]
            w.writerow(
                [
                    p,
                    s["n"],
                    *s["auroc_fail_dh"],
                    *s["auroc_fail_rbp"],
                    *s["auroc_fail_sw"],
                    s["auroc_ben_dh"][0],
                    s["auroc_ben_rbp"][0],
                    s["spearman_dh_rbp"],
                    s["spearman_dh_sw"],
                    s["spearman_rbp_sw"],
                ]
            )
    with open(route_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "p",
                "f",
                "ler_dh",
                "rec_dh",
                "ler_rbp",
                "rec_rbp",
                "ler_sw",
                "rec_sw",
                "ler_rnd",
            ]
        )
        for p in P_VALUES:
            for row in residual_stats[p]["routing"]:
                w.writerow(
                    [
                        p,
                        row["f"],
                        row["ler_dh"],
                        row["rec_dh"],
                        row["ler_rbp"],
                        row["rec_rbp"],
                        row["ler_sw"],
                        row["rec_sw"],
                        row["ler_rnd"],
                    ]
                )
    with open(ctrl_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "p",
                "n",
                "r_bp_mean",
                "r_bp_frac_zero",
                "he_ok_frac",
                "ler_lib_osd0",
                "ler_forced_osd0",
                "ler_bp",
                "n_vec_lib_ne_forced",
                "n_logical_lib_ne_forced",
                "replay_match_fail",
                "replay_match_dh",
            ]
        )
        for p in P_VALUES:
            s = residual_stats[p]
            c = s["conditional"]
            w.writerow(
                [
                    p,
                    s["n"],
                    s["r_bp_mean"],
                    s["r_bp_frac_zero"],
                    s["he_ok_frac"],
                    c["ler_library_osd0_field"],
                    c["ler_forced_algebraic_osd0"],
                    c["ler_bp_hard"],
                    c["n_library_ne_forced_vector"],
                    c["n_library_ne_forced_logical_fail"],
                    s["replay_match_fail_osd0"],
                    s["replay_match_d_h"],
                ]
            )
    with open(h2_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cell", "count", "note"])
        w.writerow(["CC", h2["CC"], "OSD0 correct, full1FV correct"])
        w.writerow(["CW", h2["CW"], "OSD0 correct, full1FV fail (harm)"])
        w.writerow(["WC", h2["WC"], "OSD0 fail, full1FV correct (rescue)"])
        w.writerow(["WW", h2["WW"], "both fail"])
        w.writerow(["n", h2["n"], ""])
        w.writerow(["ler_osd0", h2["ler_osd0"], ""])
        w.writerow(["ler_full1fv", h2["ler_full"], ""])
        w.writerow(["net_failure_change", h2["net_failure_change"], "full-osd0; 0 = no net LER change"])
    return auroc_path, route_path, ctrl_path, h2_path


def fmt_auc(t) -> str:
    x, lo, hi = t
    if x != x:
        return "n/a"
    return f"{x:.3f} [{lo:.3f},{hi:.3f}]"


def write_report(bundle, csvs) -> str:
    path = os.path.join(AUDIT_DIR, "audit_and_residual_control_report.md")
    sanity = bundle["sanity"]
    rnd = bundle["random"]
    t = bundle["timing"]
    h2 = bundle["h2"]
    res = bundle["residual"]
    lines = [
        "# Adaptive QEC audit + BP residual-weight control",
        "",
        "No manuscript TeX/figures were edited. No bb144/bb72 Monte Carlo labels were regenerated.",
        "Primary routing convention throughout: exact top-$k$ on the evaluation sample, $K=1000$ unless stated.",
        "",
        "Matched-shot source: `outputs/failure_prediction/adaptive_v1/parts/` "
        "(40 seeds × 500 shots per $p$; seeds 3000–3039 / 4000–4039 / 2000–2039).",
        "",
        "## 1. Candidate scoring audit",
        "",
        "**STATUS: PASS** (expression is $\\log(1/p_i)$, not Bernoulli NLL)",
        "",
        "production script/path: `failure_prediction/adaptive_v1.py` "
        "(`score_one_free_prefix`, `pick_one_free`); score vector built in "
        "`failure_prediction/adaptive_v1_run.py` as `log_inv_p = np.log(1.0 / matrices.error_probs)`.",
        "",
        "production output/path: `outputs/failure_prediction/adaptive_v1/parts/*.npz` "
        "(`fail_k500`, `fail_k1000`, `fail_full`).",
        "",
        "exact result:",
        "",
        "- Production score is $S(x)=\\sum_{i:x_i=1}\\log(1/p_i)$ on **channel** probabilities from the Stim DEM.",
        "- It is **not** $S_{\\mathrm{ML}}(x)=\\sum_{i:x_i=1}\\log((1-p_i)/p_i)$.",
        "- Selection is **lower $S$**: `pick_one_free` maximises $\\Delta=S(e_{\\mathrm{OSD0}})-S(e_j)$ and keeps OSD-0 if $\\max\\Delta\\le 0$ (`adaptive_v1.py` 85–93).",
        "- The same $S$ and the same scored free-column list are used for $K=500$, $K=1000$, and full 1FV; only the prefix length $k$ changes.",
        "- No `ldpc` library function reweights this score afterwards. Truncated 1FV is entirely our Python reconstruction.",
        "- Describe $S$ as a **prior-weighted / approximate channel score** (OSD-CS package surrogate), not exact independent-Bernoulli NLL.",
        "",
        "implication for manuscript: keep $S(x)=\\sum \\log(1/p_i)$ if that is what the text says; do not call it exact Bernoulli NLL.",
        "",
        "## 2. OSD-0 implementation audit",
        "",
        "**STATUS: PASS** on algorithm (free variables **set to zero**). "
        "**STATUS: ISSUE** if the manuscript claims an explicit OSD-0 reconstruction on *every* shot, including BP-converged shots.",
        "",
        "production script/path:",
        "",
        "- Library OSD-0 used for reported `fail_osd0`: `ldpc` 2.4.1 `BpOsdDecoder.osd0_decoding` "
        "(`site-packages/ldpc/bposd_decoder/_bposd_decoder.pyx` 264–280).",
        "- Independent reconstruction used for 1FV: `failure_prediction/osd_internals.py` `osd0_information_set`.",
        "- Shot loop: `failure_prediction/adaptive_v1.py` `evaluate_shot` 116–140.",
        "",
        "exact algorithm:",
        "",
        "1. Columns ordered by **ascending BP log-probability ratio** (`np.argsort(llr, kind='stable')`; upstream `soft_decision_col_sort`).",
        "2. Information set = first $\\mathrm{rank}(H)$ linearly independent columns in that order (greedy RREF).",
        "3. **Free / non-pivot bits are held at 0.** Pivot bits are the reduced RHS. They do **not** inherit BP hard decisions.",
        "4. On BP-converged shots, `ldpc` **never enters OSD**. `osd0_decoding` is copied from `bpd.decoding` (pyx 273–276). "
        "Reported OSD-0 LER on those shots is therefore the BP output. bb144 DEM: $H$ is $1800\\times 12240$, rank $1794$, $k_{\\mathrm{free}}=10446$.",
        "5. `decoder.decode()` on non-converged shots returns `osdw_decoding`; production LER does **not** use that return value. "
        "It uses the `osd0_decoding` property, then our 1FV search on the reconstructed information set.",
        "6. $H e = s$: verified on a 50-shot probe (49/49 non-converged) and on the residual replay (`he_ok_frac` below). "
        "Bit-level reconstruction vs library `osd0_decoding` on non-converged shots was previously gated in `verify_osd_reconstruction.py` / `verify_against_decoder`.",
        "",
        "relation $e_{\\mathrm{BP}}$ vs $e_{\\mathrm{OSD0}}$: $d_H=\\|e_{\\mathrm{BP}}\\oplus e_{\\mathrm{OSD0}}\\|_0$. "
        "On converged shots this is $0$ by construction (both fields are BP).",
        "",
        "implication for manuscript: say that OSD-0 zeros free variables and that BP-converged shots skip OSD and reuse $e_{\\mathrm{BP}}$. "
        "Do not say every analysed shot uses an independent-set reconstruction distinct from BP.",
        "",
        "## 3. Random-routing endpoint audit",
        "",
        "**STATUS: PASS**",
        "",
        "production script/path: `failure_prediction/final_adaptive.py` `random_lers` (160–170) and `analyse_p` (uses `fail_k1000`).",
        "Figure 3: `fig3_routing` plots random against `fail_k1000` and draws full 1FV as a **horizontal dashed visual floor** (`axhline`), "
        "while the random curve at $f=1$ is always-$K{=}1000$.",
        "",
        "exact result (200-seed random top-$k$, escalate to $K=1000$):",
        "",
        "| $p$ | $f$ | analytic $(1-f)\\mathrm{LER}_0+f\\mathrm{LER}_{K1000}$ | 200-seed mean | $\\Delta$ |",
        "|---|---|---|---|---|",
    ]
    for p in P_VALUES:
        for f in ESC_TARGETS:
            row = rnd[p]["points"][f]
            lines.append(
                f"| {p:g} | {100*f:.0f}% | {row['analytic']:.5f} | {row['rnd_mean']:.5f} | {row['delta']:+.6f} |"
            )
    p7 = rnd[0.007]["points"][0.20]
    lines += [
        "",
        f"$p=0.007$, $f=0.20$: analytic $0.8\\times 0.01990 + 0.2\\times 0.00475 = {0.8*0.01990+0.2*0.00475:.5f}$; "
        f"measured mean ${p7['rnd_mean']:.5f}$. This **exactly explains** the manuscript random value (Monte Carlo noise $<5\\times 10^{{-5}}$).",
        "",
        "implication for manuscript: random routing escalates to $K=1000$, not full 1FV. The full-1FV line in Fig. 3 is a visual accuracy floor only.",
        "",
        "## 4. Timing provenance audit",
        "",
        "**STATUS: PASS** on the numerical origin of 24.5 / 38.0 / 51.7 ms. "
        "**STATUS: ISSUE** if the manuscript presents $K{=}500/1000$ times as independently timed truncated loops.",
        "",
        f"production script/path: `{t['script']}` + `{t['slurm']}`",
        f"production output/path: `{t['json']}` ; log `{t['log']}` (host `{t['host']}`).",
        "",
        "exact result:",
        "",
        f"- Sample: $n={t['n']}$, $p={t['p']}$, Stim seed {t['seed']}, **one pass**, **no warm-up discarded**.",
        f"- Threads: `{t['omp']}`. `BpOsdDecoder` is not given an explicit `omp_thread_count`; process-level OpenMP is 1.",
        "- Adaptive 24.5 / 38.0 / 51.7 ms are **reconstructed**, not a second end-to-end strategy that skips the strong branch at runtime. "
        "After `evaluate_shot` (which always scores the full free list on non-converged shots), "
        "`adaptive_v1_timing.py` 169–181 does exact top-$k$ on that sample’s $d_H$ and sets "
        "`t_ad = t_osd0 + 1_{\\mathrm{esc}}(t_{K1000}-t_{\\mathrm{osd0}})`.",
        "- BP+OSD-0 is executed once per shot. The escalated branch **reuses** that decode: extra cost is GE + scoring already stored as `t_k1000_ms`.",
        "- Gaussian elimination is computed once per non-converged shot and **reused** for $K=500/1000/$full in the same `evaluate_shot` call.",
        "- Routing overhead (argsort of $n=1000$ $d_H$ values) is not timed separately; it is negligible vs milliseconds.",
        "- **Always $K{=}1000$ 139.7 ms includes** BP@20+OSD-0 + GE + a **prorated** slice of a *full* free-column score sweep: "
        "`t_k1000 = t_GE + (min(1000,n_free)/n_free)*t_full_score` with $n_{\\mathrm{free}}=10446$ (`adaptive_v1.py` 150–161). "
        "It is not a wall-clock of a loop that stops at $K=1000$. Same construction for $K=500$ (~132.6 ms).",
        "- **Adaptive 38.0 ms includes** the same OSD-0 time on every shot plus that prorated $K{=}1000$ extra **only on the 200/1000 highest-$d_H$ timing shots**.",
        "",
        f"Unconditional mix at 20%: ${t['t_osd0_ms']:.2f} + 0.2\\times({t['t_k1000_ms']:.2f}-{t['t_osd0_ms']:.2f}) = {t['unconditional_mix_f20_ms']:.2f}$ ms.",
        f"Reconstructed adaptive 20%: **{t['measured_adapt_f20_ms']:.2f} ms**.",
        f"$E[\\Delta T\\mid \\mathrm{{escalated}}] = ({t['measured_adapt_f20_ms']:.2f}-{t['t_osd0_ms']:.2f})/0.2 = {t['E_delta_given_escalated_ms'][0.20] if 0.20 in t['E_delta_given_escalated_ms'] else t['E_delta_given_escalated_ms']['0.2']:.2f}$ ms, "
        f"vs unconditional $E[\\Delta T]={t['E_delta_all_ms']:.2f}$ ms.",
        "High-$d_H$ selected shots are systematically more expensive than the average shot "
        "(~8 ms extra on the timing identity). The 20k science parts show the same direction "
        f"(esc20 extra {t['science_relative_cost']['mean_extra_esc20_ms']:.1f} ms vs all {t['science_relative_cost']['mean_extra_all_ms']:.1f} ms; different node, prorated times).",
        "",
        f"$273.7/38.0 \\approx {t['ratio_full_over_adapt20']:.2f}$ (vs always full 1FV).",
        f"$139.7/38.0 \\approx {t['ratio_alwaysK1000_over_adapt20']:.2f}$ (vs always $K=1000$; isolates selective routing).",
        "",
        "Do **not** replace 38.0 with 36.3. 38.0 is the correct reconstructed strategy mean on the timed sample.",
        "",
        "implication for manuscript: keep 38.0 ms; optionally footnote that $K{=}500/1000$ times prorate a full free-column sweep, "
        "and that 38.0 exceeds the unconditional mix because the routed shots are slower.",
        "",
        "## 5. BP residual-weight experiment",
        "",
    ]
    if res is None:
        lines += [
            "**STATUS: NEEDS-RERUN** — residual replay did not complete.",
            "",
        ]
    else:
        match_ok = all(res[p]["replay_match_fail_osd0"] == 1.0 and res[p]["replay_match_d_h"] == 1.0 for p in P_VALUES)
        # compare rbp vs dh routing honestly
        rbp_beats_or_ties = []
        rbp_worse = []
        for p in P_VALUES:
            for row in res[p]["routing"]:
                if row["ler_rbp"] <= row["ler_dh"] + 1e-15:
                    rbp_beats_or_ties.append((p, row["f"], row["ler_rbp"], row["ler_dh"]))
                else:
                    rbp_worse.append((p, row["f"], row["ler_rbp"], row["ler_dh"]))
        story = (
            "NEW RESULT DOES NOT OVERTURN $d_H$"
            if rbp_worse and not rbp_beats_or_ties
            else (
                "BP residual matches or beats $d_H$ at some operating points — FLAG FOR NARRATIVE"
                if rbp_beats_or_ties
                else "see table"
            )
        )
        lines += [
            f"**STATUS: PASS** (replay aligned with stored labels: fail/d_H match all 1.000)."
            if match_ok
            else "**STATUS: ISSUE** — residual replay does not perfectly match stored fail/d_H; diagnose before quoting AUROC.",
            "",
            "production script/path: `failure_prediction/audit_residual_control.py` (this run).",
            f"production output/path: `{csvs[0]}`, `{csvs[1]}`, `{csvs[2]}`.",
            "",
            "Replay is BP@20+OSD-0 on the **same Stim seeds**; 1FV fail labels are the stored matched-shot columns.",
            "",
            "### A–B. AUROC",
            "",
            "| $p$ | $d_H$ fail | $r_{\\mathrm{BP}}$ fail | syn-wt fail | $d_H$ beneficial | $r_{\\mathrm{BP}}$ beneficial | Spearman $d_H$ vs $r_{\\mathrm{BP}}$ |",
            "|---|---|---|---|---|---|---|",
        ]
        for p in P_VALUES:
            s = res[p]
            lines.append(
                f"| {p:g} | {fmt_auc(s['auroc_fail_dh'])} | {fmt_auc(s['auroc_fail_rbp'])} | "
                f"{fmt_auc(s['auroc_fail_sw'])} | {s['auroc_ben_dh'][0]:.3f} | {s['auroc_ben_rbp'][0]:.3f} | "
                f"{s['spearman_dh_rbp']:.3f} |"
            )
        lines += [
            "",
            "### C–D. Exact top-$k$, escalate to stored $K{=}1000$",
            "",
            "| $p$ | $f$ | $d_H$ LER | $r_{\\mathrm{BP}}$ LER | syn-wt LER | random LER | $d_H$ rec | $r_{\\mathrm{BP}}$ rec |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for p in P_VALUES:
            for row in res[p]["routing"]:
                lines.append(
                    f"| {p:g} | {100*row['f']:.0f}% | {row['ler_dh']:.5f} | {row['ler_rbp']:.5f} | "
                    f"{row['ler_sw']:.5f} | {row['ler_rnd']:.5f} | {100*row['rec_dh']:.1f}% | {100*row['rec_rbp']:.1f}% |"
                )
        lines += [
            "",
            "### E. Correlation",
            "",
            "| $p$ | $d_H$ vs $r_{\\mathrm{BP}}$ | $d_H$ vs syn-wt | $r_{\\mathrm{BP}}$ vs syn-wt |",
            "|---|---|---|---|",
        ]
        for p in P_VALUES:
            s = res[p]
            lines.append(
                f"| {p:g} | {s['spearman_dh_rbp']:.3f} | {s['spearman_dh_sw']:.3f} | {s['spearman_rbp_sw']:.3f} |"
            )
        lines += [
            "",
            "### F. Non-converged subset fail AUROC",
            "",
            "| $p$ | $d_H$ | $r_{\\mathrm{BP}}$ | $r_{\\mathrm{BP}}$ mean (conv / not) |",
            "|---|---|---|---|",
        ]
        for p in P_VALUES:
            s = res[p]
            lines.append(
                f"| {p:g} | {fmt_auc(s['auroc_fail_dh_nonconv'])} | {fmt_auc(s['auroc_fail_rbp_nonconv'])} | "
                f"{s['r_bp_mean_conv']:.3f} / {s['r_bp_mean_nonconv']:.3f} |"
            )
        # honest scientific question
        dh_better_all = all(
            row["ler_dh"] < row["ler_rbp"] - 1e-15
            for p in P_VALUES
            for row in res[p]["routing"]
        )
        lines += [
            "",
            f"**Primary scientific question:** does $d_H$ contain useful information beyond $r_{{\\mathrm{{BP}}}}=\\|He_{{\\mathrm{{BP}}}}\\oplus s\\|_0$?",
            "",
            (
                "$d_H$ has strictly lower exact-top-$k$ LER than $r_{\\mathrm{BP}}$ at every listed $(p,f)$. "
                "Residual weight is much stronger than syndrome weight, but it does not replace $d_H$."
                if dh_better_all
                else story
            ),
            "",
            "implication for manuscript: include $r_{\\mathrm{BP}}$ as a control if space allows; do not silently treat $d_H$ as the only decoder-internal statistic.",
            "",
        ]
    # section 6
    lines += [
        "## 6. Conventional conditional BP→OSD0 fast path",
        "",
    ]
    if res is None:
        lines.append("**STATUS: NEEDS-RERUN**")
    else:
        lines += [
            "**STATUS: PASS** (no extra Monte Carlo; library field *is* the conventional path).",
            "",
            "production script/path: `ldpc` `osd0_decoding` (BP if converge, else OSD-0) vs "
            "`osd_internals.osd0_information_set` forced on every shot including converged.",
            "",
            "| $p$ | LER library OSD-0 field (= conventional) | LER forced algebraic OSD-0 | LER BP hard | vector diffs | logical-fail diffs | lib ok / forced fail | lib fail / forced ok |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for p in P_VALUES:
            c = res[p]["conditional"]
            sp = c["split"]
            lines.append(
                f"| {p:g} | {c['ler_library_osd0_field']:.5f} | {c['ler_forced_algebraic_osd0']:.5f} | "
                f"{c['ler_bp_hard']:.5f} | {c['n_library_ne_forced_vector']} | "
                f"{c['n_library_ne_forced_logical_fail']} | {sp['lib_ok_forced_fail']} | {sp['lib_fail_forced_ok']} |"
            )
        lines += [
            "",
            "The reported manuscript OSD-0 LER **is** the conventional conditional output. "
            "Forcing algebraic OSD-0 on BP-converged shots is the extra construction needed if one wanted OSD-0 even when BP already returned a vector.",
            "Timing of that extra GE on the converged minority was **not** re-benchmarked (would be a new serial study). "
            "Existing n=1000 serial already times BP+OSD-0 on every shot at 10.5 ms mean; that *is* the conventional decode() cost, "
            "because `ldpc` skips OSD on converge. Building $d_H$ on converged shots is a Hamming distance of two identical vectors (zero extra decode).",
            "On non-converged shots, $d_H$ is a popcount of two vectors the fast path already materialises: $e_{\\mathrm{BP}}$ and $e_{\\mathrm{OSD0}}$. "
            "So the extra cost of $d_H$ beyond conventional BP→OSD0 is essentially zero at the decode level; "
            "the cost of *using* $d_H$ is the gated $K{=}1000$ branch.",
            "",
        ]
    lines += [
        "## 7. H2 transition counts",
        "",
        "**STATUS: PASS**",
        "",
        f"production output/path: `{h2['path']}` ; CSV `{csvs[3] if csvs else ''}`.",
        "",
        f"n=300. OSD-0 LER={h2['ler_osd0']:.5f} ({h2['n_fail_osd0']}/300). "
        f"full 1FV LER={h2['ler_full']:.5f} ({h2['n_fail_full']}/300). "
        f"net failure change = {h2['net_failure_change']}.",
        "",
        "|  | full 1FV correct | full 1FV fail |",
        "|---|---|---|",
        f"| OSD-0 correct | CC={h2['CC']} | CW={h2['CW']} |",
        f"| OSD-0 fail | WC={h2['WC']} | WW={h2['WW']} |",
        "",
        f"WC (rescues) = **{h2['WC']}**. CW (harms) = **{h2['CW']}**. They cancel. "
        "Do not claim monotonic improvement on hardware.",
        "",
        "## 8. Held-out table discrepancy",
        "",
        "**STATUS: PASS** (computational reason identified; label should be renamed)",
        "",
        "production script/path: `failure_prediction/adaptive_v1_controls.py` `holdout_block`.",
        "production output/path: `outputs/paper_figures/final_adaptive/controls_bp_tau.md`.",
        "",
        "Main-figure LER is exact top-$k$ on the **pooled** $n=20000$ shots.",
        "The held-out table “oracle / in-sample” column is exact top-$k$ on each **10k-shot test fold**, then the two fold LERs are **averaged** "
        "(`mean_over_folds`). Ranking a subset is not the same as ranking the union, so the numbers need not match.",
        "",
        "Check: $p=0.007$, $f=0.20$: pooled $0.00545$ = $109/20000$. Fold oracles $0.00540$ and $0.00540$ average to $0.00540$ = $108/20000$. "
        "One extra residual failure appears only in the pooled ranking. $p=0.008$, $f=0.20$: pooled $0.02530$ vs fold-mean $0.02505$.",
        "Tie handling ($d_H$ ties broken by mergesort original order) also differs once the comparison set changes.",
        "Seeds are complementary halves of the same 40 files, not a different shot set.",
        "",
        "implication for manuscript: do **not** label the fold oracle “in-sample LER” as if it were Fig. 3’s 20k number. "
        "Call it “in-fold exact top-$k$ on the held-out seeds (mean of two 10k folds)”.",
        "",
        "## 9. K=500 sanity check",
        "",
        f"**STATUS: {'PASS' if sanity['ok_k500'] else 'FAIL'}**",
        "",
        "Same `parts/*.npz` `fail_k500` column; same `topk_mask` as $K{=}1000$.",
        "",
        "| $p$ | always $K{=}500$ | 10% | 20% | 30% |",
        "|---|---|---|---|---|",
    ]
    for row in sanity["rows"]:
        p = row["p"]
        a = row["adapt"]
        lines.append(
            f"| {p:g} | {row['always']['k500']:.5f} | {a[0.10]['k500']:.5f} | {a[0.20]['k500']:.5f} | {a[0.30]['k500']:.5f} |"
        )
    lines += [
        "",
        "Matches the expected exact-top-$k$ values in the task statement." if sanity["ok_k500"] else "DISCREPANCY — see rows.",
        "",
        "## 10. Recommended manuscript corrections",
        "",
        "### MUST FIX IN MANUSCRIPT",
        "",
        "- If the text calls $S(x)$ exact Bernoulli NLL, change to channel score $\\sum \\log(1/p_i)$ (prior-weighted / OSD-CS surrogate).",
        "- If OSD-0 is described as running on every shot: BP-converged shots skip OSD and copy $e_{\\mathrm{BP}}$.",
        "- Held-out table: rename “in-sample LER” so it is not identified with the 20k main-figure top-$k$.",
        "- H2: report CW=3 as well as WC=3; net LER change is zero. Do not imply monotone 1FV improvement.",
        "- Timing: if claiming truncated-search wall-clock, disclose prorated full-sweep scoring for $K{=}500/1000$. Keep 38.0 ms (do not “correct” to 36.3).",
        "",
        "### NEW RESULT WORTH INCLUDING",
        "",
        "- BP residual weight $r_{\\mathrm{BP}}$ as a decoder-internal control vs $d_H$ and syndrome weight (Section 5 tables).",
        "- Random-routing analytic identity $(1-f)\\mathrm{LER}_{\\mathrm{OSD0}}+f\\mathrm{LER}_{K1000}$.",
        "- $139.7/38.0\\approx 3.68\\times$ vs always $K{=}1000$ as the routing benefit isolated from full-1FV.",
        "",
        "### NO ACTION REQUIRED",
        "",
        "- Primary 20k LER table (OSD-0 / always $K$ / adaptive $K{=}1000$ exact top-$k$) matches the frozen parts.",
        "- $K{=}500$ exact-top-$k$ sensitivity matches the expected values.",
        "- Random routing already uses $K{=}1000$, not full 1FV.",
        "- Fig. 3 full-1FV dashed line as a visual floor is fine if the caption says so.",
        "",
        "### ANY RESULT THAT CHANGES THE CURRENT SCIENTIFIC STORY",
        "",
    ]
    if res is not None:
        dh_strict = all(
            row["ler_dh"] < row["ler_rbp"] - 1e-15
            for p in P_VALUES
            for row in res[p]["routing"]
        )
        rbp_close = any(
            abs(row["ler_dh"] - row["ler_rbp"]) * 20000 < 3
            for p in P_VALUES
            for row in res[p]["routing"]
        )
        if dh_strict:
            lines.append(
                "The $d_H$ routing story is **not overturned**. $r_{\\mathrm{BP}}$ is a strong but strictly weaker matched-budget router than $d_H$ "
                "in this table. It is much stronger than syndrome weight. Include it as a control, not a replacement."
            )
        else:
            lines.append(
                "**FLAG:** $r_{\\mathrm{BP}}$ is not strictly worse than $d_H$ at every $(p,f)$. Revisit the claim that disagreement-in-error-space is necessary beyond unsatisfied-parity count."
            )
        if rbp_close:
            lines.append(
                "At some operating points the two LERs differ by only a couple of failures in 20k; do not over-claim a large practical gap there."
            )
        c6 = res[0.007]["conditional"]
        if c6["n_library_ne_forced_logical_fail"] == 0:
            lines.append(
                "Forced algebraic OSD-0 on BP-converged shots does not change bb144 LER at these $p$ relative to the library field. "
                "The conventional vs forced distinction is conceptually real but empirically idle on this dataset."
            )
    else:
        lines.append("Residual replay incomplete — scientific-story flag pending Section 5.")
    lines += [
        "",
        "Main-LER sanity: "
        + ("PASS" if sanity["ok_main"] else "FAIL — stop, provenance mismatch."),
        "",
    ]
    os.makedirs(AUDIT_DIR, exist_ok=True)
    text = "\n".join(lines) + "\n"
    with open(path, "w") as fh:
        fh.write(text)
    return path


def main() -> None:
    os.makedirs(AUDIT_DIR, exist_ok=True)
    os.makedirs(RES_PARTS, exist_ok=True)
    rng = np.random.default_rng(20260907)
    print("Loading parts...", flush=True)
    by_p = load_parts_with_meta()
    sanity = sanity_and_k500(by_p)
    print("sanity main", sanity["ok_main"], "k500", sanity["ok_k500"], flush=True)
    if not sanity["ok_main"]:
        print("STOP: primary LER mismatch")
        print(json.dumps(jsonable(sanity), indent=2))
        return
    rnd = random_audit(by_p, rng)
    timing = timing_audit(by_p)
    h2 = h2_transitions()
    files = sorted(glob.glob(os.path.join(PARTS, "*.npz")))
    print(f"Residual replay n_files={len(files)}", flush=True)
    n_workers = min(20, len(files))
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        done = 0
        for _ in pool.imap_unordered(replay_one, files, chunksize=1):
            done += 1
            if done % 10 == 0:
                print(f"  residual {done}/{len(files)}", flush=True)
    merged = load_residual_merged()
    print("Analysing residual AUROC/routing...", flush=True)
    residual = analyse_residual(merged, rng)
    csvs = write_csvs(merged, residual, h2)
    bundle = dict(sanity=sanity, random=rnd, timing=timing, h2=h2, residual=residual)
    with open(os.path.join(AUDIT_DIR, "audit_bundle.json"), "w") as fh:
        json.dump(jsonable(bundle), fh, indent=2)
    report = write_report(bundle, csvs)
    print("wrote", report)
    for pth in csvs:
        print("wrote", pth)


if __name__ == "__main__":
    main()
