#!/usr/bin/env bash
# bb_144_12_12 decoder-gap pilot at p = 0.006 / 0.007.
#
# Question: does the OSD-0 -> OSD-CS logical error gap grow progressively with p,
# or does it only appear once BP stops converging near threshold?
#
# Known so far (1000 shots each, r=24):
#   p=0.003  OSD-0 0/1000   OSD-CS 0/1000   BP@20 conv 0.585
#   p=0.005  OSD-0 1/1000   OSD-CS 0/1000   BP@20 conv 0.216
#   p=0.008  OSD-0 68/1000  OSD-CS 17/1000  BP@20 conv 0.004   <- 4x gap, 58x cost
#
# Decoder set trimmed to what answers the question: one plain BP for the
# convergence rate, two cheap candidates, one expensive reference. Dropping
# osdcs8 roughly halves the runtime, which pays for more shots instead.
#
#   sbatch submit_bb144_gap_pilot.slurm

#SBATCH -J bb144_gap_pilot
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH --array=0-1
#SBATCH -o outputs/failure_prediction/decoder_survey/logs/%x_%A_%a.out
#SBATCH -e outputs/failure_prediction/decoder_survey/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=06:00:00

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

PS=(0.006 0.007)
P="${PS[$SLURM_ARRAY_TASK_ID]}"
SHOTS="${SHOTS:-3000}"

echo "[$(date)] host=$(hostname) task=${SLURM_ARRAY_TASK_ID}"
echo "code=bb_144_12_12 p=${P} rounds=24 shots=${SHOTS}"

python -m failure_prediction.decoder_survey \
    --code bb_144_12_12 \
    --p "${P}" \
    --rounds 24 \
    --shots "${SHOTS}" \
    --seed 7 \
    --decoders bp20,bp20_osd0,bp20_lsd,bp50_osdcs2 \
    --progress-every 500

echo "[$(date)] Done"
