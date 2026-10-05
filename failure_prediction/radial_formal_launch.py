"""Launch frozen radial formal calib/test parts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

OUT = Path("outputs/failure_prediction/nonbb_radial_formal")
CFG = OUT / "radial_90_8_frozen_config.json"
SHOTS = 500
WORKERS = int(os.environ.get("NONBB_WORKERS", "40"))


def _jobs(cfg: dict):
    jobs = []
    for split, key in (("calib", "calib_seeds"), ("test", "test_seeds")):
        for p_str, seeds in cfg[key].items():
            p = float(p_str)
            for seed in seeds:
                out = OUT / split / f"p{p:g}_seed{seed}.npz"
                jobs.append((split, p, int(seed), str(out)))
    return jobs


def _run(job):
    split, p, seed, out = job
    if os.path.isfile(out):
        return ("skip", split, p, seed, out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "failure_prediction.radial_formal_run",
        "--p",
        repr(p),
        "--seed",
        str(seed),
        "--shots",
        str(SHOTS),
        "--split",
        split,
        "--out",
        out,
    ]
    subprocess.check_call(cmd, cwd=str(Path(__file__).resolve().parents[1]))
    return ("ok", split, p, seed, out)


def main() -> None:
    cfg = json.loads(CFG.read_text())
    jobs = _jobs(cfg)
    print(f"launch {len(jobs)} parts with {WORKERS} workers", flush=True)
    n_ok = n_skip = n_fail = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(_run, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                status, split, p, seed, out = fut.result()
            except Exception as exc:
                n_fail += 1
                print("FAIL", exc, flush=True)
                continue
            if status == "skip":
                n_skip += 1
            else:
                n_ok += 1
            print(f"{status} {split} p={p:g} seed={seed}", flush=True)
    print(f"done ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
