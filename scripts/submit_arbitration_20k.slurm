#!/usr/bin/env bash
# Paired fast/strong decoder outcomes for the arbitration mechanism analysis.
#
#   code   bb_144_12_12, p=0.008, r=24
#   fast   BP@20 + OSD-0
#   strong BP@20 + OSD-CS-2
#
# Both arms come from one BpOsdDecoder(osd_method="osd_cs", osd_order=2) call via
# its osd0_decoding / osdw_decoding outputs, so the pairing is exact by
# construction rather than by matching sampler seeds.
#
# The wolpy partition is OverSubscribe=EXCLUSIVE, so a job holds a whole node
# whatever it asks for. A job array would therefore need one node per task and
# would sit in the queue; instead this takes a single node and fans out worker
# processes across its cores. Each worker draws an independent shot set.
#
# CPU only: these decoders have no GPU path and the OSD-CS sweep is the
# bottleneck at roughly 1.3 s/shot single-threaded.
#
#   sbatch submit_arbitration_20k.slurm

#SBATCH -J arb20k
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/arbitration_mechanism/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/arbitration_mechanism/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --time=04:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/arbitration_mechanism/logs
mkdir -p outputs/failure_prediction/arbitration_mechanism/shards

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate base
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
# Each worker is single-threaded; let the fan-out provide the parallelism.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

WORKERS="${WORKERS:-40}"
SHOTS_PER_WORKER="${SHOTS_PER_WORKER:-500}"
TOTAL=$((WORKERS * SHOTS_PER_WORKER))

echo "[$(date)] host=$(hostname) workers=${WORKERS} x ${SHOTS_PER_WORKER} shots = ${TOTAL}"

WORKER_LOG_DIR=outputs/failure_prediction/arbitration_mechanism/logs/workers_${SLURM_JOB_ID}
mkdir -p "${WORKER_LOG_DIR}"

for w in $(seq 0 $((WORKERS - 1))); do
    SEED=$((2000 + w))
    python -m failure_prediction.arbitration \
        --code bb_144_12_12 \
        --p 0.008 \
        --rounds 24 \
        --shots "${SHOTS_PER_WORKER}" \
        --seed "${SEED}" \
        --max-iter 20 \
        --out-dir outputs/failure_prediction/arbitration_mechanism/shards \
        --progress-every 100 \
        > "${WORKER_LOG_DIR}/w${w}.log" 2>&1 &
done

wait
echo "[$(date)] all workers finished"

ls -1 outputs/failure_prediction/arbitration_mechanism/shards/*.npz | wc -l \
    | xargs -I{} echo "shards written: {}"
echo "[$(date)] Done"
