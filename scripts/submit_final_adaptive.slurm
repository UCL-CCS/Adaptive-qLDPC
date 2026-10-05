#!/usr/bin/env bash
# Final Adaptive V1 close-out: n=1000 serial timing + minimal bb72 control.
# Does not generate a new bb144 Monte Carlo set.
#
#   sbatch submit_final_adaptive.slurm

#SBATCH -J av1_final
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/adaptive_v1/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/adaptive_v1/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=02:00:00

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/adaptive_v1/logs
mkdir -p outputs/failure_prediction/adaptive_v1_bb72/parts
mkdir -p outputs/failure_prediction/adaptive_v1_bb72/logs

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

echo "[$(date)] host=$(hostname)  n=1000 timing then bb72 control"

python -m failure_prediction.adaptive_v1_timing \
    --code bb_144_12_12 --p 0.007 --rounds 24 --shots 1000 --seed 9001 \
    --skip-package \
    --out outputs/failure_prediction/adaptive_v1/serial_timing_n1000.json

WORKERS=20
SHOTS=500
WORKER_LOG_DIR=outputs/failure_prediction/adaptive_v1_bb72/logs/workers_${SLURM_JOB_ID}
mkdir -p "${WORKER_LOG_DIR}"
echo "[$(date)] bb72 control workers=${WORKERS} shots=${SHOTS} p=0.006,0.008 r=12"

fail=0
pids=()
for w in $(seq 0 $((WORKERS - 1))); do
    (
        set -euo pipefail
        python -m failure_prediction.adaptive_v1_run \
            --code bb_72_12_6 --p 0.006 --rounds 12 \
            --shots "${SHOTS}" --seed $((5000 + w)) --max-iter 20 \
            --out "outputs/failure_prediction/adaptive_v1_bb72/parts/p0.006_seed$((5000 + w)).npz"
        python -m failure_prediction.adaptive_v1_run \
            --code bb_72_12_6 --p 0.008 --rounds 12 \
            --shots "${SHOTS}" --seed $((6000 + w)) --max-iter 20 \
            --out "outputs/failure_prediction/adaptive_v1_bb72/parts/p0.008_seed$((6000 + w)).npz"
    ) > "${WORKER_LOG_DIR}/w${w}.log" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done

n_npz=$(ls -1 outputs/failure_prediction/adaptive_v1_bb72/parts/*.npz 2>/dev/null | wc -l)
echo "[$(date)] bb72 workers finished  npz=${n_npz}  fail=${fail}"

if [ "${fail}" -ne 0 ]; then
    echo "[$(date)] WARNING: at least one bb72 worker failed; inspect ${WORKER_LOG_DIR}"
    exit 1
fi
echo "[$(date)] Done"
