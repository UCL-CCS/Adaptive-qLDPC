"""Write the frozen radial_90_8 formal-experiment config. Run once before shots."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from failure_prediction.qldpc_circuit import build_z_memory_circuit
from failure_prediction.radial_code import DATA, construction_meta, published_logicals, radial_90_8
from failure_prediction.radial_headroom import dem_bundle

OUT = Path("outputs/failure_prediction/nonbb_radial_formal")
CFG_NAME = "radial_90_8_frozen_config.json"
ROUNDS = 12


def write_dem_provenance(p_values) -> None:
    """Circuit/DEM artefacts. Does not alter the frozen scientific JSON."""
    dem_dir = OUT / "circuit_dem"
    dem_dir.mkdir(exist_ok=True)
    meta = {}
    code = radial_90_8()
    for p in p_values:
        circuit = build_z_memory_circuit(code, p=p, rounds=ROUNDS)
        stim_path = dem_dir / f"circuit_p{p:g}.stim"
        dem_path = dem_dir / f"dem_p{p:g}.dem"
        if not stim_path.is_file():
            stim_path.write_text(str(circuit))
        bundle, dem_try = dem_bundle(circuit)
        if bundle is None:
            raise SystemExit(f"DEM failed at p={p}: {dem_try}")
        dem, matrices, packed, rank, dem_meta = bundle
        if not dem_path.is_file():
            dem_path.write_text(str(dem))
        h = matrices.h
        meta[f"{p:g}"] = {
            **dem_meta,
            "circuit_path": str(stim_path),
            "dem_path": str(dem_path),
            "H_sha256": sha256_array(np.ascontiguousarray(h.astype(np.uint8))),
            "error_probs_sha256": sha256_array(
                np.ascontiguousarray(matrices.error_probs.astype(np.float64))
            ),
            "logicals_shape": list(matrices.logicals.shape),
            "n_detectors_circuit": int(circuit.num_detectors),
            "n_observables_circuit": int(circuit.num_observables),
        }
    (OUT / "dem_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(a)
    return sha256_bytes(arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode())


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calib").mkdir(exist_ok=True)
    (OUT / "test").mkdir(exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    code = radial_90_8()
    logs = published_logicals()
    meta = construction_meta()
    import ldpc
    import stim

    git = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]), text=True
    ).strip()
    cfg = {
        "code": "radial_90_8",
        "pcm_path": str(DATA),
        "pcm_source": "PCMs/90_8_10 from tRowans/radial-codes-public",
        "reference_commit": meta["reference_commit"],
        "r": 3,
        "s": 5,
        "n": int(code.n),
        "k": 8,
        "d": "unknown",
        "hashes": {
            "hx_csv": sha256_file(DATA / "hx.csv"),
            "hz_csv": sha256_file(DATA / "hz.csv"),
            "lx_csv": sha256_file(DATA / "lx.csv"),
            "lz_csv": sha256_file(DATA / "lz.csv"),
            "hx_array": sha256_array(code.hx.toarray().astype(np.uint8)),
            "hz_array": sha256_array(code.hz.toarray().astype(np.uint8)),
            "lx_array": sha256_array(logs["x_logicals"]),
            "lz_array": sha256_array(logs["z_logicals"]),
        },
        "circuit": {
            "builder": "failure_prediction.qldpc_circuit.build_z_memory_circuit",
            "basis": "Z",
            "rounds": 12,
            "noise": (
                "Adaptive V1 Z-memory: post-reset X_ERROR(p) on data and Z-ancillas; "
                "CX DEPOLARIZE2(p); M(p); data DEPOLARIZE1(p) once per round; final data M(p)"
            ),
            "dem": {
                "decompose_errors": True,
                "ignore_decomposition_failures": True,
            },
            "not_used": "full-CSS production circuit (noiseless-ok in pilot; k_free too large for full 1FV)",
        },
        "p": [0.008, 0.009],
        "p_selection": "blind OSD-0 vs full-1FV headroom pilot; seed 21 excluded from formal data",
        "pilot_seed_excluded": 21,
        "decoder": {
            "bp_method": "ms",
            "ms_scaling_factor": 0.625,
            "schedule": "flooding/parallel (ldpc BpOsdDecoder default)",
            "max_iter": 20,
            "osd0_fast_path": "if H e_BP = s return e_BP; else algebraic OSD-0 (free variables zero)",
            "score": "S(x)=sum_{i:x_i=1} log(1/p_i) on Stim DEM channel priors; lower is better via argmax Δ",
            "score_is_not": "exact Bernoulli ML",
            "K": 1000,
            "k_free_pilot": 3289,
            "full_1fv": "all one-free-variable OSD perturbations in the same LPR ordering",
            "tie_break_topk": "numpy.argsort(-score, kind='mergesort')",
            "tie_break_1fv": "argmax Δ; keep OSD-0 if Δ<=0",
        },
        "shots_per_part": 500,
        "calib_shots_per_p": 5000,
        "test_shots_per_p": 20000,
        "calib_seeds": {
            "0.008": list(range(9000, 9010)),
            "0.009": list(range(9200, 9210)),
        },
        "test_seeds": {
            "0.008": list(range(9100, 9140)),
            "0.009": list(range(9300, 9340)),
        },
        "rng_bootstrap": 20260908,
        "n_boot": 800,
        "n_random": 200,
        "software": {
            "git_commit": git,
            "python": sys.version.replace("\n", " "),
            "numpy": np.__version__,
            "stim": stim.__version__,
            "ldpc": ldpc.__version__,
            "scipy": __import__("scipy").__version__,
        },
        "logical_endpoint": "any of k Z-memory observables misidentified",
    }
    path = OUT / CFG_NAME
    if path.is_file():
        old = json.loads(path.read_text())
        # scientific freeze: do not change seeds/p/decoder if file already exists
        sci = (
            old.get("p"),
            old.get("calib_seeds"),
            old.get("test_seeds"),
            old.get("decoder"),
            old.get("circuit", {}).get("rounds"),
        )
        new = (
            cfg["p"],
            cfg["calib_seeds"],
            cfg["test_seeds"],
            cfg["decoder"],
            cfg["circuit"]["rounds"],
        )
        if sci != new:
            raise SystemExit(f"refusing to overwrite frozen scientific config at {path}")
        print("frozen config already present; scientific fields unchanged")
        write_dem_provenance(old["p"])
        if not (OUT / "pcm_hashes.json").is_file():
            (OUT / "pcm_hashes.json").write_text(json.dumps(old.get("hashes", {}), indent=2) + "\n")
        if not (OUT / "seeds.json").is_file():
            (OUT / "seeds.json").write_text(
                json.dumps(
                    {
                        "pilot_seed_excluded": old.get("pilot_seed_excluded"),
                        "calib_seeds": old.get("calib_seeds"),
                        "test_seeds": old.get("test_seeds"),
                        "rng_bootstrap": old.get("rng_bootstrap"),
                    },
                    indent=2,
                )
                + "\n"
            )
        return
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("wrote", path)
    (OUT / "pcm_hashes.json").write_text(json.dumps(cfg["hashes"], indent=2) + "\n")
    (OUT / "seeds.json").write_text(
        json.dumps(
            {
                "pilot_seed_excluded": cfg["pilot_seed_excluded"],
                "calib_seeds": cfg["calib_seeds"],
                "test_seeds": cfg["test_seeds"],
                "rng_bootstrap": cfg["rng_bootstrap"],
            },
            indent=2,
        )
        + "\n"
    )
    write_dem_provenance(cfg["p"])
    print("wrote DEM provenance")


if __name__ == "__main__":
    main()
