#!/usr/bin/env bash
# Serial n=1000 Adaptive V1 timing on one exclusive wolpy node.
# Skips package OSD-CS. Does not decode a new science dataset.
#
#   sbatch submit_adaptive_v1_timing_n1000.slurm

#SBATCH -J av1_time1k
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/adaptive_v1/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/adaptive_v1/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

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

echo "[$(date)] host=$(hostname) n=1000 skip-package"
python -m failure_prediction.adaptive_v1_timing \
    --code bb_144_12_12 --p 0.007 --rounds 24 --shots 1000 --seed 9001 \
    --skip-package \
    --out outputs/failure_prediction/adaptive_v1/serial_timing_n1000.json
echo "[$(date)] Done"
