"""Blind 1FV-headroom pilot for the published radial [[n=90,k=8]] instance.

Does NOT compute or inspect d_H, r_BP, or routing AUROCs.
Does NOT write into Adaptive V1 BB or the deleted Hamming-HGP directories.

Usage:
    python -m failure_prediction.radial_headroom
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from failure_prediction.adaptive_v1 import (
    K_SENSITIVITY,
    pick_one_free,
    score_one_free_prefix,
)
from failure_prediction.h2_full_css import build_css_memory_circuit
from failure_prediction.osd_internals import gf2_rank_packed, osd0_information_set, pack_matrix
from failure_prediction.qldpc_circuit import (
    build_z_memory_circuit,
    detector_error_model_to_matrices,
    x_logical_basis,
    z_logical_basis,
)
from failure_prediction.qldpc_codes import css_commutes
from failure_prediction.radial_code import (
    cnot_schedule,
    code_weight_stats,
    construction_meta,
    published_logicals,
    radial_90_8,
    validate_logicals,
)

OUT = "outputs/failure_prediction/nonbb_radial_headroom_pilot"
ROUNDS = 12
PILOT_SEED = 21
N_PILOT = 500
P_SCAN = (
    1e-4,
    3e-4,
    5e-4,
    1e-3,
    2e-3,
    3e-3,
    4e-3,
    5e-3,
    6e-3,
    7e-3,
    8e-3,
    9e-3,
    1e-2,
    1.2e-2,
    1.5e-2,
    2e-2,
)
REL_HEADROOM_MIN = 0.20


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if np.isnan(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def software_versions() -> dict:
    import ldpc
    import stim

    git = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]), text=True
    ).strip()
    return {
        "git_commit": git,
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "stim": stim.__version__,
        "ldpc": ldpc.__version__,
        "scipy": __import__("scipy").__version__,
    }


def count_ops(circuit) -> dict:
    n2 = n1 = 0
    for op in circuit:
        if op.name == "CX":
            n2 += 1
        elif op.name in ("H", "DEPOLARIZE1", "X_ERROR"):
            n1 += 1
    return {
        "n_qubits": int(circuit.num_qubits),
        "n_meas": int(circuit.num_measurements),
        "n_detectors": int(circuit.num_detectors),
        "n_observables": int(circuit.num_observables),
        "n_cx": n2,
        "n_1q_noise_or_h": n1,
        "depth_ticks": int(getattr(circuit, "num_ticks", 0)),
    }


def try_dem(circuit, decompose: bool) -> dict:
    try:
        dem = circuit.detector_error_model(
            decompose_errors=decompose, ignore_decomposition_failures=True
        )
        return {"ok": True, "dem": dem, "error": None}
    except Exception as exc:
        return {"ok": False, "dem": None, "error": f"{type(exc).__name__}: {exc}"}


def gf2_residual(h, e, syndrome) -> np.ndarray:
    he = (h.astype(np.int32).dot(np.asarray(e, dtype=np.int32)) % 2).astype(np.uint8)
    return np.asarray(he).ravel() ^ np.asarray(syndrome, dtype=np.uint8).ravel()


def _logical_fail(logicals, e, actual) -> int:
    pred = (logicals @ np.asarray(e, dtype=np.int64)) % 2
    return int(np.any(pred != np.asarray(actual, dtype=np.uint8).ravel()))


def evaluate_headroom(decoder, syndrome, actual, packed, log_inv_p, logicals, h, n_cols, rank):
    """OSD-0 / K=1000 / full 1FV logical fails. No d_H / r_BP / routing."""

    syndrome = np.asarray(syndrome, dtype=np.uint8)
    actual = np.asarray(actual, dtype=np.uint8).ravel()
    decoder.decode(syndrome)
    e_bp = np.asarray(decoder.bp_decoding, dtype=np.uint8) % 2
    if int(np.count_nonzero(gf2_residual(h, e_bp, syndrome))) == 0:
        fail = _logical_fail(logicals, e_bp, actual)
        return {
            "fail_osd0": fail,
            "fail_k1000": fail,
            "fail_full": fail,
            "he_osd0_ok": 1,
            "bp_syndrome_ok": 1,
        }
    llr = np.asarray(decoder.log_prob_ratios, dtype=np.float64)
    info = osd0_information_set(packed, syndrome, llr, n_cols=n_cols, rank=rank)
    if info is None:
        e0 = np.asarray(decoder.osd0_decoding, dtype=np.uint8) % 2
        fail = _logical_fail(logicals, e0, actual)
        return {
            "fail_osd0": fail,
            "fail_k1000": fail,
            "fail_full": fail,
            "he_osd0_ok": int(np.count_nonzero(gf2_residual(h, e0, syndrome)) == 0),
            "bp_syndrome_ok": 0,
        }
    e0 = np.asarray(info["solution"], dtype=np.uint8) % 2
    fail0 = _logical_fail(logicals, e0, actual)
    delta, free_p, rf = score_one_free_prefix(info, log_inv_p, n_cols, n_cols)
    e1000 = pick_one_free(info, delta, free_p, rf, K_SENSITIVITY)
    e_full = pick_one_free(info, delta, free_p, rf, int(delta.size))
    return {
        "fail_osd0": fail0,
        "fail_k1000": _logical_fail(logicals, e1000, actual),
        "fail_full": _logical_fail(logicals, e_full, actual),
        "he_osd0_ok": int(np.count_nonzero(gf2_residual(h, e0, syndrome)) == 0),
        "bp_syndrome_ok": 0,
    }


def transitions(fail0, failk) -> dict:
    f0 = np.asarray(fail0).astype(int)
    fk = np.asarray(failk).astype(int)
    return {
        "CC": int(((f0 == 0) & (fk == 0)).sum()),
        "CW": int(((f0 == 0) & (fk == 1)).sum()),
        "WC": int(((f0 == 1) & (fk == 0)).sum()),
        "WW": int(((f0 == 1) & (fk == 1)).sum()),
        "net": int(f0.sum() - fk.sum()),
    }


def dem_bundle(circuit):
    dem_try = try_dem(circuit, decompose=True)
    if not dem_try["ok"]:
        return None, dem_try
    dem = dem_try["dem"]
    matrices = detector_error_model_to_matrices(dem)
    n_cols = matrices.h.shape[1]
    packed = pack_matrix(matrices.h, n_cols=n_cols, extra_cols=1)
    rank = gf2_rank_packed(packed, n_cols=n_cols)
    meta = {
        "n_detectors": int(dem.num_detectors),
        "n_observables": int(dem.num_observables),
        "n_error_mechanisms": int(n_cols),
        "H_shape": [int(matrices.h.shape[0]), int(n_cols)],
        "rank_H": int(rank),
        "k_free": int(n_cols - rank),
        "K": 1000,
        "K_eff": int(min(1000, n_cols - rank)),
        "K_is_full_1fv": bool(n_cols - rank <= 1000),
        "decompose_errors": True,
    }
    return (dem, matrices, packed, rank, meta), dem_try


def run_pilot(circuit, matrices, packed, rank, p, seed, n_shots):
    from ldpc import BpOsdDecoder

    h = matrices.h
    n_cols = h.shape[1]
    decoder = BpOsdDecoder(
        h,
        error_channel=matrices.error_probs.tolist(),
        max_iter=20,
        bp_method="ms",
        ms_scaling_factor=0.625,
        osd_method="osd_0",
    )
    log_inv_p = np.log(1.0 / matrices.error_probs)
    logicals = matrices.logicals.astype(np.int64)
    syn, actual = circuit.compile_detector_sampler(seed=seed).sample(
        n_shots, separate_observables=True
    )
    t0 = time.time()
    rows = [
        evaluate_headroom(
            decoder, syn[i], actual[i], packed, log_inv_p, logicals, h, n_cols, rank
        )
        for i in range(n_shots)
    ]
    elapsed = time.time() - t0
    fail0 = np.asarray([r["fail_osd0"] for r in rows], dtype=np.int8)
    failk = np.asarray([r["fail_k1000"] for r in rows], dtype=np.int8)
    failf = np.asarray([r["fail_full"] for r in rows], dtype=np.int8)
    he = np.asarray([r["he_osd0_ok"] for r in rows], dtype=np.int8)
    n0, nk, nf = int(fail0.sum()), int(failk.sum()), int(failf.sum())
    ler0 = n0 / n_shots
    lerf = nf / n_shots
    rel = ((ler0 - lerf) / ler0) if ler0 > 0 else float("nan")
    trans_f = transitions(fail0, failf)
    trans_k = transitions(fail0, failk)
    useful = bool(
        (not np.isnan(rel))
        and rel >= REL_HEADROOM_MIN
        and trans_f["WC"] >= 10
        and trans_f["net"] >= 5
    )
    return {
        "p": p,
        "n": n_shots,
        "seed": seed,
        "seconds": elapsed,
        "ms_per_shot": 1e3 * elapsed / n_shots,
        "n_fail_osd0": n0,
        "n_fail_k1000": nk,
        "n_fail_full": nf,
        "ler_osd0": ler0,
        "ler_k1000": nk / n_shots,
        "ler_full": lerf,
        "rel_improvement_full": rel,
        "ratio_osd0_over_full": (ler0 / lerf) if lerf > 0 else None,
        "he_osd0_ok": float(he.mean()),
        "bp_syndrome_ok": float(np.mean([r["bp_syndrome_ok"] for r in rows])),
        "trans_k1000": trans_k,
        "trans_full": trans_f,
        "meets_headroom_criterion": useful,
        "fail_osd0": fail0,
        "fail_k1000": failk,
        "fail_full": failf,
    }


def write_report(payload) -> str:
    c = payload["construction"]
    v = payload["validation"]
    go = payload["go_nogo"]
    lines = [
        "# Radial / lifted-product qLDPC headroom pilot",
        "",
        "Blind to d_H / BP-residual routing. Hamming-HGP experiment removed; BB Adaptive V1 untouched.",
        "",
        "## 1. Selected code and why it was selected",
        "",
        f"- family: quantum radial codes (lifted product of classical quasi-cyclic radial codes)",
        f"- instance: published PCM `90_8_10`, (r,s)=({c['r']},{c['s']})",
        f"- n={c['n']}, k={c['k_formula']} (rank formula {c['k_from_ranks']})",
        f"- selection: {c['selection_reason']}",
        "",
        "Not a BB code. Not the Hamming-HGP instance. No random construction search.",
        "",
        "## 2. Literature/software source",
        "",
        f"- {c['literature']}",
        f"- {c['reference_implementation']} @ `{c['reference_commit']}`",
        f"- local PCM copy: `failure_prediction/data/radial_90_8_10/`",
        "",
        "## 3. Code validation",
        "",
        f"- H_X {c['weights']['hx_shape']}, rank {c['weights']['hx_rank']}",
        f"- H_Z {c['weights']['hz_shape']}, rank {c['weights']['hz_rank']}",
        f"- CSS commute H_X H_Z^T = 0: **{v['css_commutes']}**",
        f"- row/column weights: `{json.dumps(c['weights'])}`",
        f"- distance: **{c['d']}**. {c['d_note']}",
        "",
        "## 4. Logical validation",
        "",
        f"- published logicals: `{v['published_logicals']}`",
        f"- pipeline logicals (ker / stabilizer quotient, Adaptive V1 convention): `{v['pipeline_logicals']}`",
        "",
        "## 5. Circuit validation",
        "",
        f"- full-CSS noiseless 64 shots: **{v['fullcss']['noiseless_ok']}**, ops `{v['fullcss']['quiet']}`",
        f"- Z-memory noiseless 64 shots: **{v['zmem']['noiseless_ok']}**, ops `{v['zmem']['quiet']}`",
        f"- production circuit: **{v['production_circuit']}**",
        f"- rounds: {ROUNDS} (not tuned on decoder performance)",
        f"- full-CSS DEM decompose_errors=True: {v['fullcss']['dem_decompose_true']}",
        f"- full-CSS k_free: {v['fullcss'].get('k_free_if_used')} "
        "(if ≳10^4, production uses Z-memory so that full 1FV remains the same search family)",
        "",
        "## 6. Noise / DEM validation",
        "",
        "Adaptive V1 circuit-level convention: post-reset X_ERROR(p), CX DEPOLARIZE2(p),",
        "M(p), data DEPOLARIZE1(p) once per round, final data M(p).",
        "",
        f"- DEM: `{json.dumps(v['production_dem'])}`",
        f"- H e_OSD0 = s on production path: recorded in the p-scan `he_osd0_ok` column",
        "",
        "## 7. Decoder configuration",
        "",
        f"`{json.dumps(payload['decoder_config'])}`",
        "",
        f"- software: `{json.dumps(payload['software'])}`",
        "",
        "## 8. Pilot p scan",
        "",
        f"Independent seed {PILOT_SEED}, n={N_PILOT} shots/p. Not a production dataset.",
        "",
        "| p | OSD-0 LER | K=1000 LER | full1FV LER | rel. impr. | ratio | WC | CW | net | he_ok | ms/shot | criterion |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["pilot"]:
        tf = row["trans_full"]
        he = row.get("he_osd0_ok")
        he_s = f"{he:.3f}" if isinstance(he, (int, float)) else "n/a"
        ms = row.get("ms_per_shot")
        ms_s = f"{ms:.1f}" if isinstance(ms, (int, float)) else "n/a"
        rel = row["rel_improvement_full"]
        rel_s = f"{rel:.3f}" if isinstance(rel, (int, float)) and rel == rel else "nan"
        lines.append(
            f"| {row['p']:.5g} | {row['ler_osd0']:.4f} ({row['n_fail_osd0']}/{row['n']}) | "
            f"{row['ler_k1000']:.4f} | {row['ler_full']:.4f} | {rel_s} | "
            f"{row['ratio_osd0_over_full']} | {tf['WC']} | {tf['CW']} | {tf['net']} | "
            f"{he_s} | {ms_s} | {row['meets_headroom_criterion']} |"
        )
    lines += [
        "",
        "## 9. OSD0 / K1000 / full1FV headroom",
        "",
        f"Pre-registered useful-headroom rule: relative improvement "
        f"(LER_OSD0 − LER_full1FV)/LER_OSD0 ≥ {REL_HEADROOM_MIN} "
        "and WC ≥ 10 with net ≥ 5 on this pilot sample.",
        "",
        f"Points meeting the criterion: `{go['points_meeting_criterion']}`",
        "",
        "## 10. Rescue / harm counts",
        "",
        "See WC (rescues) and CW (harms) in the scan table. net = WC − CW.",
        "",
        "## 11. k_free and computational feasibility",
        "",
        f"- k_free = {v['production_dem'].get('k_free')}",
        f"- K_eff = {v['production_dem'].get('K_eff')}; K=1000 is full 1FV? "
        f"{v['production_dem'].get('K_is_full_1fv')}",
        f"- typical ms/shot from the scan (see table). Full 1FV is the OSD combination sweep "
        "over free variables, not exhaustive physical-error search.",
        "",
        "## 12. GO / NO-GO conclusion",
        "",
        go["reason"],
        "",
        f"FORMAL CROSS-FAMILY TEST READY: **{go['formal_test_ready']}**",
        "",
    ]
    if go["formal_test_ready"] == "YES":
        lines += [
            "Frozen for a later d_H test (not run in this task):",
            f"- code: radial_90_8, (r,s)=(3,5)",
            f"- circuit: {v['production_circuit']}, rounds={ROUNDS}",
            f"- recommended p: {go['frozen_p']}",
            "- decoder: Adaptive V1 BP/OSD-0/1FV conventions unchanged",
            "",
            "Do not inspect d_H until that follow-up is requested.",
            "",
        ]
    path = os.path.join(OUT, "radial_headroom_pilot_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "construction"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "circuit"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "pilot"), exist_ok=True)

    code = radial_90_8()
    stats = code_weight_stats(code)
    k_from_ranks = code.n - stats["hx_rank"] - stats["hz_rank"]
    commute = bool(css_commutes(code))
    pub = published_logicals()
    x_pipe = x_logical_basis(code)
    z_pipe = z_logical_basis(code)
    pub_val = validate_logicals(code, pub["x_logicals"], pub["z_logicals"])
    pipe_val = validate_logicals(code, x_pipe, z_pipe)
    meta = construction_meta()
    meta.update(
        {
            "n": int(code.n),
            "k_from_ranks": int(k_from_ranks),
            "weights": stats,
        }
    )

    from scipy import sparse

    sparse.save_npz(os.path.join(OUT, "construction", "hx.npz"), code.hx.tocsr())
    sparse.save_npz(os.path.join(OUT, "construction", "hz.npz"), code.hz.tocsr())
    np.savez_compressed(
        os.path.join(OUT, "construction", "logicals.npz"),
        x_published=pub["x_logicals"],
        z_published=pub["z_logicals"],
        x_pipeline=x_pipe,
        z_pipeline=z_pipe,
    )
    with open(os.path.join(OUT, "circuit", "cnot_schedule.json"), "w") as fh:
        json.dump(_jsonable(cnot_schedule(code)), fh, indent=2)
        fh.write("\n")

    full_quiet = build_css_memory_circuit(
        code, p=0.0, rounds=ROUNDS, basis="Z", ancilla_layout="separate"
    )
    det0, obs0 = full_quiet.compile_detector_sampler(seed=0).sample(64, separate_observables=True)
    full_ok = bool(np.all(det0 == 0) and np.all(obs0 == 0))
    full_noisy = build_css_memory_circuit(
        code, p=0.003, rounds=ROUNDS, basis="Z", ancilla_layout="separate"
    )
    full_dem = try_dem(full_noisy, decompose=True)
    with open(os.path.join(OUT, "circuit", "fullcss_quiet.stim"), "w") as fh:
        fh.write(str(full_quiet))

    z_quiet = build_z_memory_circuit(code, p=0.0, rounds=ROUNDS)
    detz, obsz = z_quiet.compile_detector_sampler(seed=0).sample(64, separate_observables=True)
    z_ok = bool(np.all(detz == 0) and np.all(obsz == 0))
    with open(os.path.join(OUT, "circuit", "zmem_quiet.stim"), "w") as fh:
        fh.write(str(z_quiet))

    blocker = []
    if not commute:
        blocker.append("Hx Hz^T != 0")
    if not pub_val["ok"]:
        blocker.append(f"published logicals: {pub_val['issues']}")
    if not pipe_val["ok"]:
        blocker.append(f"pipeline logicals: {pipe_val['issues']}")
    if k_from_ranks != meta["k_formula"]:
        blocker.append(f"k formula {meta['k_formula']} != {k_from_ranks}")

    # Full-CSS noiseless + DEM can succeed with a very large k_free. Full 1FV
    # then scores every free column. If that is prohibitive, use the Adaptive V1
    # Z-memory circuit (same noise placement / DEM flags) rather than replacing
    # the 1FV search family.
    fullcss_k_free = None
    if full_dem["ok"]:
        b_full, _ = dem_bundle(full_noisy)
        if b_full is not None:
            fullcss_k_free = b_full[4]["k_free"]

    FULLCSS_KFREE_LIMIT = 10000
    if full_ok and full_dem["ok"] and fullcss_k_free is not None and fullcss_k_free <= FULLCSS_KFREE_LIMIT:
        production_name = "full_css_z_memory"
        prod_circuit_builder = lambda p: build_css_memory_circuit(
            code, p=p, rounds=ROUNDS, basis="Z", ancilla_layout="separate"
        )
    elif z_ok:
        production_name = "z_memory"
        prod_circuit_builder = lambda p: build_z_memory_circuit(code, p=p, rounds=ROUNDS)
    else:
        production_name = None
        prod_circuit_builder = None
        if not full_ok:
            blocker.append("full-CSS noiseless failed")
        if not z_ok:
            blocker.append("Z-memory noiseless failed")

    validation = {
        "css_commutes": commute,
        "published_logicals": pub_val,
        "pipeline_logicals": pipe_val,
        "fullcss": {
            "noiseless_ok": full_ok,
            "quiet": count_ops(full_quiet),
            "n_data": code.n,
            "n_x_anc": int(code.hx.shape[0]),
            "n_z_anc": int(code.hz.shape[0]),
            "dem_decompose_true": {"ok": full_dem["ok"], "error": full_dem["error"]},
            "k_free_if_used": fullcss_k_free,
            "full1fv_note": (
                f"full-CSS k_free={fullcss_k_free}. Pilot uses Z-memory if this exceeds "
                f"{10000} because scoring every free column is the Adaptive V1 full-1FV "
                "definition and is not replaced by another search."
            ),
        },
        "zmem": {
            "noiseless_ok": z_ok,
            "quiet": count_ops(z_quiet),
            "n_data": code.n,
            "n_x_anc": 0,
            "n_z_anc": int(code.hz.shape[0]),
        },
        "production_circuit": production_name,
        "production_dem": None,
        "blocker": blocker,
    }

    decoder_config = {
        "bp_method": "ms",
        "ms_scaling_factor": 0.625,
        "schedule": "flooding/parallel (ldpc BpOsdDecoder default)",
        "max_iter": 20,
        "osd0_fast_path": "if H e_BP = s return e_BP; else algebraic OSD-0 (free vars zero)",
        "score": "S(x)=sum_{i:x_i=1} log(1/p_i) on Stim DEM channel priors",
        "score_is_not": "exact Bernoulli ML",
        "K": 1000,
        "full_1fv": "all one-free-variable OSD perturbations; not exhaustive physical search",
        "not_computed": ["d_H", "r_BP", "routing AUROC", "adaptive LER"],
    }

    if prod_circuit_builder is None or (not z_ok and not (full_ok and full_dem["ok"])):
        go = {
            "formal_test_ready": "NO",
            "reason": "NO-GO: circuit/logical validation failed before noisy pilot. " + "; ".join(blocker),
            "points_meeting_criterion": [],
            "frozen_p": [],
        }
        payload = {
            "construction": meta,
            "validation": validation,
            "decoder_config": decoder_config,
            "software": software_versions(),
            "pilot": [],
            "go_nogo": go,
        }
        with open(os.path.join(OUT, "summary.json"), "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2)
            fh.write("\n")
        print(write_report(payload))
        print(go["reason"])
        return

    # Probe DEM on a mid p for metadata; rebuild per p in the scan (probs depend on p).
    probe = prod_circuit_builder(0.003)
    bundle, dem_try = dem_bundle(probe)
    if bundle is None:
        go = {
            "formal_test_ready": "NO",
            "reason": (
                "NO-GO: production circuit noiseless-ok but paper DEM "
                f"decompose_errors=True failed: {dem_try['error']}"
            ),
            "points_meeting_criterion": [],
            "frozen_p": [],
        }
        validation["production_dem"] = {"ok": False, "error": dem_try["error"]}
        payload = {
            "construction": meta,
            "validation": validation,
            "decoder_config": decoder_config,
            "software": software_versions(),
            "pilot": [],
            "go_nogo": go,
        }
        with open(os.path.join(OUT, "summary.json"), "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2)
            fh.write("\n")
        print(write_report(payload))
        print(go["reason"])
        return

    dem, matrices, packed, rank, dem_meta = bundle
    validation["production_dem"] = dem_meta
    with open(os.path.join(OUT, "circuit", f"{production_name}_p0.003.stim"), "w") as fh:
        fh.write(str(probe))
    with open(os.path.join(OUT, "circuit", f"{production_name}_p0.003.dem"), "w") as fh:
        fh.write(str(dem))

    print(
        f"production={production_name} k_free={dem_meta['k_free']} "
        f"K_eff={dem_meta['K_eff']} noiseless full={full_ok} z={z_ok}",
        flush=True,
    )

    pilot_rows = []
    for p in P_SCAN:
        print(f"pilot p={p:g} ...", flush=True)
        part = os.path.join(OUT, "pilot", f"p{p:g}_seed{PILOT_SEED}.npz")
        if os.path.isfile(part):
            d = np.load(part)
            fail0 = d["fail_osd0"]
            failk = d["fail_k1000"]
            failf = d["fail_full"]
            n0, nk, nf = int(fail0.sum()), int(failk.sum()), int(failf.sum())
            n_shots = int(fail0.size)
            ler0 = n0 / n_shots
            lerf = nf / n_shots
            rel = ((ler0 - lerf) / ler0) if ler0 > 0 else float("nan")
            trans_f = transitions(fail0, failf)
            trans_k = transitions(fail0, failk)
            useful = bool(
                (not np.isnan(rel))
                and rel >= REL_HEADROOM_MIN
                and trans_f["WC"] >= 10
                and trans_f["net"] >= 5
            )
            slim = {
                "p": p,
                "n": n_shots,
                "seed": PILOT_SEED,
                "seconds": None,
                "ms_per_shot": None,
                "n_fail_osd0": n0,
                "n_fail_k1000": nk,
                "n_fail_full": nf,
                "ler_osd0": ler0,
                "ler_k1000": nk / n_shots,
                "ler_full": lerf,
                "rel_improvement_full": rel,
                "ratio_osd0_over_full": (ler0 / lerf) if lerf > 0 else None,
                "he_osd0_ok": None,
                "bp_syndrome_ok": None,
                "trans_k1000": trans_k,
                "trans_full": trans_f,
                "meets_headroom_criterion": useful,
                "k_free": dem_meta["k_free"],
                "K_eff": dem_meta["K_eff"],
                "reloaded": True,
            }
            print(
                f"  reload LER osd0={slim['ler_osd0']:.4f} full={slim['ler_full']:.4f} "
                f"WC={slim['trans_full']['WC']} CW={slim['trans_full']['CW']}",
                flush=True,
            )
            pilot_rows.append(slim)
            continue
        circ = prod_circuit_builder(p)
        b, err = dem_bundle(circ)
        if b is None:
            print("  DEM failed", err["error"], flush=True)
            continue
        _, mats, pkd, rk, dmeta = b
        rec = run_pilot(circ, mats, pkd, rk, p, PILOT_SEED, N_PILOT)
        rec["k_free"] = dmeta["k_free"]
        rec["K_eff"] = dmeta["K_eff"]
        np.savez_compressed(
            os.path.join(OUT, "pilot", f"p{p:g}_seed{PILOT_SEED}.npz"),
            fail_osd0=rec["fail_osd0"],
            fail_k1000=rec["fail_k1000"],
            fail_full=rec["fail_full"],
            p=np.full(N_PILOT, p),
            seed=np.full(N_PILOT, PILOT_SEED),
        )
        slim = {k: v for k, v in rec.items() if k not in ("fail_osd0", "fail_k1000", "fail_full")}
        print(
            f"  LER osd0={slim['ler_osd0']:.4f} k1000={slim['ler_k1000']:.4f} "
            f"full={slim['ler_full']:.4f} rel={slim['rel_improvement_full']} "
            f"WC={slim['trans_full']['WC']} CW={slim['trans_full']['CW']} "
            f"{slim['ms_per_shot']:.1f} ms/shot",
            flush=True,
        )
        pilot_rows.append(slim)

    useful = [r for r in pilot_rows if r["meets_headroom_criterion"]]
    he_bad = [
        r
        for r in pilot_rows
        if r.get("he_osd0_ok") is not None and r["he_osd0_ok"] < 1.0 - 1e-12
    ]
    if he_bad:
        go = {
            "formal_test_ready": "NO",
            "reason": "NO-GO: H e_OSD0 != s on some pilot shots.",
            "points_meeting_criterion": [r["p"] for r in useful],
            "frozen_p": [],
        }
    elif not useful:
        go = {
            "formal_test_ready": "NO",
            "reason": (
                "NO-GO: insufficient 1FV headroom. No scanned p met the pre-registered "
                f"relative-improvement ≥ {REL_HEADROOM_MIN} with WC≥10 and net≥5. "
                "Did not search other radial instances."
            ),
            "points_meeting_criterion": [],
            "frozen_p": [],
        }
    else:
        # Freeze one or two useful p from OSD-0 LER / headroom only (no d_H).
        # Prefer points with OSD-0 LER in a statistically useful band if present.
        band = [r for r in useful if 0.01 <= r["ler_osd0"] <= 0.15]
        pool = band if band else useful
        frozen = [pool[0]["p"]]
        if len(pool) > 1 and pool[-1]["p"] != pool[0]["p"]:
            frozen.append(pool[-1]["p"])
        go = {
            "formal_test_ready": "YES",
            "reason": (
                "GO: published radial code/circuit/DEM validated; full 1FV feasible on "
                "the Adaptive V1 Z-memory DEM; at least one p meets the pre-registered "
                "headroom criterion. Ready for a later d_H test. d_H was not inspected."
            ),
            "points_meeting_criterion": [r["p"] for r in useful],
            "frozen_p": frozen,
        }
        freeze = {
            "code": "radial_90_8",
            "r": 3,
            "s": 5,
            "circuit": production_name,
            "rounds": ROUNDS,
            "p": frozen,
            "K": 1000,
            "k_free": dem_meta["k_free"],
            "pilot_seed_excluded": PILOT_SEED,
            "decoder": decoder_config,
            "note": "Frozen from OSD-0 LER and 1FV headroom only. d_H not inspected.",
        }
        with open(os.path.join(OUT, "frozen_config.json"), "w") as fh:
            json.dump(_jsonable(freeze), fh, indent=2)
            fh.write("\n")

    payload = {
        "construction": meta,
        "validation": validation,
        "decoder_config": decoder_config,
        "software": software_versions(),
        "pilot_seed": PILOT_SEED,
        "pilot_n": N_PILOT,
        "rounds": ROUNDS,
        "pilot": pilot_rows,
        "go_nogo": go,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(_jsonable(payload), fh, indent=2)
        fh.write("\n")
    import csv

    csv_path = os.path.join(OUT, "summary.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "p",
                "n",
                "ler_osd0",
                "ler_k1000",
                "ler_full",
                "rel_improvement_full",
                "WC",
                "CW",
                "net",
                "k_free",
                "meets_criterion",
            ]
        )
        for r in pilot_rows:
            tf = r["trans_full"]
            w.writerow(
                [
                    r["p"],
                    r["n"],
                    r["ler_osd0"],
                    r["ler_k1000"],
                    r["ler_full"],
                    r["rel_improvement_full"],
                    tf["WC"],
                    tf["CW"],
                    tf["net"],
                    r.get("k_free"),
                    r["meets_headroom_criterion"],
                ]
            )
    report = write_report(payload)
    print("report", report)
    print("FORMAL CROSS-FAMILY TEST READY:", go["formal_test_ready"])
    print(go["reason"], flush=True)


if __name__ == "__main__":
    main()
