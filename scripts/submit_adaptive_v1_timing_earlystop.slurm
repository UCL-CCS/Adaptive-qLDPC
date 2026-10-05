#!/usr/bin/env bash
# True early-stop K=500 / K=1000 serial timing. Does not overwrite the prorated
# serial_timing_n1000.json.
#
#   sbatch submit_adaptive_v1_timing_earlystop.slurm

#SBATCH -J av1_estop
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/adaptive_v1/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/adaptive_v1/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/adaptive_v1/logs
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate base
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "[$(date)] host=$(hostname) early-stop n=1000"
python -m failure_prediction.adaptive_v1_timing_earlystop \
    --code bb_144_12_12 --p 0.007 --rounds 24 --shots 1000 --seed 9001 \
    --out outputs/failure_prediction/adaptive_v1/serial_timing_earlystop_n1000.json
echo "[$(date)] Done"
