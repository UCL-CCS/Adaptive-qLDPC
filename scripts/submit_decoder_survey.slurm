#!/usr/bin/env bash
# Decoder (cost, LER) frontier survey.
#
# Stage B found that BP@20 + OSD-0 matches BP+OSD-CS order 2/4/8 at a sixth of
# the cost, so the fast/strong pair we had picked was dominated by a third
# decoder and adaptive routing had nothing to save. This survey looks for a
# noise point and code where some (fast, strong) pair on the Pareto frontier
# has both a real LER gap and a real cost gap.
#
# Tasks 0-5 : bb_72_12_6,   r=12, 50k shots, p sweep
# Tasks 6-8 : bb_144_12_12, r=24,  1k shots, cost probe (this code is far slower)
#
# CPU only; ldpc decoders are single-threaded C++.
#
#   sbatch submit_decoder_survey.slurm

#SBATCH -J decoder_survey
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH --array=0-8
#SBATCH -o outputs/failure_prediction/decoder_survey/logs/%x_%A_%a.out
#SBATCH -e outputs/failure_prediction/decoder_survey/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs/failure_prediction/decoder_survey/logs

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    eval "$(conda shell.bash hook)"
fi
conda activate base
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

CODES=(bb_72_12_6 bb_72_12_6 bb_72_12_6 bb_72_12_6 bb_72_12_6 bb_72_12_6 \
       bb_144_12_12 bb_144_12_12 bb_144_12_12)
PS=(0.002 0.003 0.004 0.005 0.006 0.008 0.003 0.005 0.008)
SHOTS=(50000 50000 50000 50000 50000 50000 1000 1000 1000)
ROUNDS=(12 12 12 12 12 12 24 24 24)

I="${SLURM_ARRAY_TASK_ID}"
CODE="${CODES[$I]}"
P="${PS[$I]}"
N="${SHOTS[$I]}"
R="${ROUNDS[$I]}"

echo "[$(date)] host=$(hostname) task=${I}"
echo "code=${CODE} p=${P} rounds=${R} shots=${N}"

python -m failure_prediction.decoder_survey \
    --code "${CODE}" \
    --p "${P}" \
    --rounds "${R}" \
    --shots "${N}" \
    --seed 7 \
    --progress-every $(( N / 5 ))

echo "[$(date)] Done"
