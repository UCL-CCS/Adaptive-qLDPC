#!/usr/bin/env bash
# Adaptive Decoder V1 validation on bb_144_12_12.
#
# Scope: implement and measure the adaptive pipeline. Do not extend OSD scoring.
#
#   fast     BP@20 + OSD-0
#   router   d_H(e_BP, e_OSD0)   (already produced by the fast path)
#   strong   truncated one-free-variable CS, K=500 (K=1000 sensitivity)
#
# wolpy is OverSubscribe=EXCLUSIVE: one job = one node. Fan out 40 workers
# across cores. Each worker handles three matched (p, seed) blocks of 500 shots.
# p=0.008 reuses Stim seeds 2000-2039 from the 20k arbitration set.
#
#   sbatch submit_adaptive_v1.slurm

#SBATCH -J adaptive_v1
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/adaptive_v1/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/adaptive_v1/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=08:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/adaptive_v1/logs
mkdir -p outputs/failure_prediction/adaptive_v1/parts

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

WORKERS="${WORKERS:-40}"
SHOTS="${SHOTS_PER_WORKER:-500}"
echo "[$(date)] host=$(hostname) workers=${WORKERS} shots/worker=${SHOTS}"

WORKER_LOG_DIR=outputs/failure_prediction/adaptive_v1/logs/workers_${SLURM_JOB_ID}
mkdir -p "${WORKER_LOG_DIR}"

fail=0
pids=()
for w in $(seq 0 $((WORKERS - 1))); do
    (
        set -euo pipefail
        python -m failure_prediction.adaptive_v1_run \
            --code bb_144_12_12 --p 0.006 --rounds 24 \
            --shots "${SHOTS}" --seed $((3000 + w)) --max-iter 20 \
            --out "outputs/failure_prediction/adaptive_v1/parts/p0.006_seed$((3000 + w)).npz"
        python -m failure_prediction.adaptive_v1_run \
            --code bb_144_12_12 --p 0.007 --rounds 24 \
            --shots "${SHOTS}" --seed $((4000 + w)) --max-iter 20 \
            --out "outputs/failure_prediction/adaptive_v1/parts/p0.007_seed$((4000 + w)).npz"
        python -m failure_prediction.adaptive_v1_run \
            --code bb_144_12_12 --p 0.008 --rounds 24 \
            --shots "${SHOTS}" --seed $((2000 + w)) --max-iter 20 \
            --out "outputs/failure_prediction/adaptive_v1/parts/p0.008_seed$((2000 + w)).npz"
    ) > "${WORKER_LOG_DIR}/w${w}.log" 2>&1 &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        fail=1
    fi
done

n_npz=$(ls -1 outputs/failure_prediction/adaptive_v1/parts/*.npz 2>/dev/null | wc -l)
echo "[$(date)] workers finished  npz=${n_npz}  fail=${fail}"

echo "[$(date)] serial timing (same node, no contention)"
python -m failure_prediction.adaptive_v1_timing \
    --code bb_144_12_12 --p 0.007 --rounds 24 --shots 200 --seed 9001 \
    --out outputs/failure_prediction/adaptive_v1/serial_timing.json

echo "[$(date)] analysis"
python -m failure_prediction.adaptive_v1_analysis \
    --dir outputs/failure_prediction/adaptive_v1 \
    --out-dir outputs/failure_prediction/adaptive_v1

if [ "${fail}" -ne 0 ]; then
    echo "[$(date)] WARNING: at least one worker failed; inspect ${WORKER_LOG_DIR}"
    exit 1
fi
echo "[$(date)] Done"
