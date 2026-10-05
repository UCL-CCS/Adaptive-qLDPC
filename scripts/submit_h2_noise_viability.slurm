#!/usr/bin/env bash
# Zero-HQC Stim noise-amplification viability on bb_36.
# Does not submit Quantinuum / Nexus jobs.
#
#   sbatch submit_h2_noise_viability.slurm

#SBATCH -J h2_viab
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH -o outputs/failure_prediction/h2_noise_viability/%x_%j.out
#SBATCH -e outputs/failure_prediction/h2_noise_viability/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/h2_noise_viability

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

echo "[$(date)] host=$(hostname)  H2 viability  0 HQC"
python -m failure_prediction.h2_noise_viability --workers 7
echo "[$(date)] Done"
