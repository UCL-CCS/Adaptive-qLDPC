#!/usr/bin/env bash
# BP-effort control: is the OSD-0 / OSD-CS separation driven by poor BP
# convergence, or by the size of the OSD candidate space?
#
# Code and noise are held fixed (bb_144_12_12, p=0.008, r=24) so the error
# distribution is identical; only BP max_iter varies. Each task runs the
# OSD-0 and OSD-CS arms at *matched* max_iter, which is what makes the gap
# measurable per effort level.
#
# All tasks use the same seed and the same circuit, so Stim's deterministic
# sampler gives every task the identical shot set. The comparison across
# max_iter is therefore paired as well as the comparison within it.
#
# Prediction under the candidate-space mechanism: the gap ratio is flat in
# max_iter. Under the poor-BP-convergence mechanism: the gap shrinks as
# max_iter grows and BP converges more often.
#
#   sbatch submit_bb144_bpiter_control.slurm

#SBATCH -J bb144_bpiter
#SBATCH -p wolpy
#SBATCH -A di49mer
#SBATCH --array=0-2
#SBATCH -o outputs/failure_prediction/decoder_survey/logs/%x_%A_%a.out
#SBATCH -e outputs/failure_prediction/decoder_survey/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00

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

ITERS=(10 20 50)
M="${ITERS[$SLURM_ARRAY_TASK_ID]}"
SHOTS="${SHOTS:-3000}"
# The survey names its output by (code, p, rounds, shots, seed), which is
# identical across these three tasks, so give each max_iter its own directory.
OUT_DIR="outputs/failure_prediction/decoder_survey/bpiter_control/maxiter${M}"

echo "[$(date)] host=$(hostname) task=${SLURM_ARRAY_TASK_ID}"
echo "code=bb_144_12_12 p=0.008 rounds=24 shots=${SHOTS} max_iter=${M}"

python -m failure_prediction.decoder_survey \
    --code bb_144_12_12 \
    --p 0.008 \
    --rounds 24 \
    --shots "${SHOTS}" \
    --seed 7 \
    --decoders "bp${M}_osd0,bp${M}_osdcs2" \
    --out-dir "${OUT_DIR}" \
    --progress-every 500

echo "[$(date)] Done"
