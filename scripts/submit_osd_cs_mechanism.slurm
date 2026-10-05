#!/usr/bin/env bash
# Replay WC/WW/CW (+ a modest high-d CC control) from the existing 20k paired
# dataset and reconstruct the OSD-CS candidate landscape offline.
#
# wolpy is OverSubscribe=EXCLUSIVE, so this takes one node and fans out one
# worker per shard. Each worker only decodes the selected shots of that shard
# (OSD-0, not the CS sweep).
#
#   sbatch submit_osd_cs_mechanism.slurm

#SBATCH -J osdcs_mech
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/osd_cs_mechanism/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/osd_cs_mechanism/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=02:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/osd_cs_mechanism/logs
mkdir -p outputs/failure_prediction/osd_cs_mechanism/parts

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

N_SHARDS=40
echo "[$(date)] host=$(hostname) shards=${N_SHARDS}"

for s in $(seq 0 $((N_SHARDS - 1))); do
    python -m failure_prediction.osd_cs_mechanism \
        --shard-start "${s}" \
        --shard-stop $((s + 1)) \
        --n-cc-per-shard 6 \
        --out "outputs/failure_prediction/osd_cs_mechanism/parts/part_${s}.npz" \
        > "outputs/failure_prediction/osd_cs_mechanism/logs/part_${s}.log" 2>&1 &
done
wait
echo "[$(date)] workers finished"

python - <<'PY'
import glob, os
import numpy as np

parts = sorted(glob.glob("outputs/failure_prediction/osd_cs_mechanism/parts/part_*.npz"))
tables = [np.load(p, allow_pickle=True) for p in parts]
skip = {"rec_idx", "rec_score", "rec_correct", "rec_offsets", "label_agreement", "rank", "n_cols"}
scalar = [k for k in tables[0].files if k not in skip]
merged = {k: np.concatenate([t[k] for t in tables]) for k in scalar}
rec_idx, rec_score, rec_correct, offsets = [], [], [], [0]
agree = np.zeros(2, dtype=np.int64)
for t in tables:
    rec_idx.append(t["rec_idx"])
    rec_score.append(t["rec_score"])
    rec_correct.append(t["rec_correct"])
    base = offsets[-1]
    off = t["rec_offsets"]
    offsets.extend((off[1:] + base).tolist())
    agree += t["label_agreement"]
merged["rec_idx"] = np.concatenate(rec_idx)
merged["rec_score"] = np.concatenate(rec_score)
merged["rec_correct"] = np.concatenate(rec_correct)
merged["rec_offsets"] = np.asarray(offsets, dtype=np.int64)
merged["label_agreement"] = agree
merged["rank"] = tables[0]["rank"]
merged["n_cols"] = tables[0]["n_cols"]
out = "outputs/failure_prediction/osd_cs_mechanism/trajectories.npz"
np.savez_compressed(out, **merged)
print(f"merged {len(parts)} parts -> {out}  shots={merged['outcome'].size}  label_agree={agree}")
PY

echo "[$(date)] Done"
