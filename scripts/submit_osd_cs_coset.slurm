#!/usr/bin/env bash
# Coset-aware aggregation on the existing 1527-shot trajectory set.
# No new Monte Carlo. One exclusive wolpy node, one worker per seed.
#
#   sbatch submit_osd_cs_coset.slurm

#SBATCH -J osdcs_coset
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/osd_cs_coset/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/osd_cs_coset/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/osd_cs_coset/logs
mkdir -p outputs/failure_prediction/osd_cs_coset/parts

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate base
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

python - <<'PY'
import numpy as np
t = np.load("outputs/failure_prediction/osd_cs_mechanism/trajectories.npz")
seeds = sorted(set(t["seed"].astype(int)))
print("seeds", len(seeds), seeds[0], seeds[-1])
open("outputs/failure_prediction/osd_cs_coset/logs/seeds.txt", "w").write(
    "\n".join(str(s) for s in seeds) + "\n"
)
PY

mapfile -t SEEDS < outputs/failure_prediction/osd_cs_coset/logs/seeds.txt
echo "[$(date)] host=$(hostname) n_seeds=${#SEEDS[@]}"

for sd in "${SEEDS[@]}"; do
    python -m failure_prediction.osd_cs_coset \
        --seed-min "${sd}" \
        --seed-max $((sd + 1)) \
        --out "outputs/failure_prediction/osd_cs_coset/parts/seed_${sd}.npz" \
        > "outputs/failure_prediction/osd_cs_coset/logs/seed_${sd}.log" 2>&1 &
done
wait
echo "[$(date)] workers finished"

python - <<'PY'
import glob
import numpy as np

parts = sorted(glob.glob("outputs/failure_prediction/osd_cs_coset/parts/seed_*.npz"))
tables = [np.load(p) for p in parts]
keys = tables[0].files
merged = {k: np.concatenate([t[k] for t in tables]) for k in keys}
out = "outputs/failure_prediction/osd_cs_coset/coset.npz"
np.savez_compressed(out, **merged)
print(f"merged {len(parts)} -> {out} n={merged['outcome'].size}")
PY

echo "[$(date)] Done"
