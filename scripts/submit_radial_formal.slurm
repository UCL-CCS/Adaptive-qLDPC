#!/usr/bin/env bash
# Formal radial_90_8: 5k calib + 20k test at frozen p=0.008, 0.009.
# Does not write Adaptive V1 BB or the headroom-pilot directory.

#SBATCH -J radial_formal
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/nonbb_radial_formal/logs/%x_%j.out
#SBATCH -e outputs/failure_prediction/nonbb_radial_formal/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/nonbb_radial_formal/{logs,calib,test}

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
export NONBB_WORKERS="${SLURM_CPUS_PER_TASK:-40}"

echo "[$(date)] host=$(hostname) git=$(git rev-parse --short HEAD)"
python -m failure_prediction.radial_formal_freeze
python -m failure_prediction.radial_formal_launch
python -m failure_prediction.radial_formal_analyze
echo "[$(date)] Done"
