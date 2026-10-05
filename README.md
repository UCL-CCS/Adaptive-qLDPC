# Adaptive qLDPC

Adaptive decoding for quantum LDPC codes.

Frozen method:

```
BP@20 + OSD-0  →  d_H(e_BP, e_OSD0)  →  OSD-0 or truncated 1FV search (K=1000)
```

`d_H` is the Hamming distance between the BP hard decision and the OSD-0 reconstruction. It is produced by the fast path; no extra decode is required to route. The expensive branch perturbs one OSD free variable at a time (first `K` in LPR order) using the prior-weighted surrogate

```
S(x) = sum_{i: x_i = 1} log(1/p_i)
```

on Stim DEM channel probabilities. Full one-free-variable search is the accuracy reference, not an exhaustive physical-error search.

Primary simulation: bivariate bicycle `bb_144_12_12`, circuit-level Z-memory. Cross-family check: radial / lifted-product `radial_90_8` (`n=90`, `k=8`, distance unknown).

## Layout

```
failure_prediction/     Adaptive V1 decoder, circuits, radial/H2/mechanism code
scripts/                Slurm launchers for the paper experiments
outputs/paper_figures/  Publication figures and compact summaries
```

Raw 20k-shot `.npz` dumps are not in this repository. They can be regenerated with the scripts below.

## Setup

Python 3.11+ with:

```bash
pip install -r requirements.txt
```

Run from the repository root:

```bash
export PYTHONPATH=.
```

## Adaptive Decoder V1 (BB)

```bash
python -m failure_prediction.adaptive_v1_run \
    --code bb_144_12_12 --p 0.007 --rounds 24 \
    --shots 500 --seed 3000 --max-iter 20 \
    --out outputs/failure_prediction/adaptive_v1/parts/p0.007_seed3000.npz

python -m failure_prediction.final_adaptive
python -m failure_prediction.paper_figures_adaptive_v1
```

Cluster: `sbatch scripts/submit_adaptive_v1.slurm` then `sbatch scripts/submit_final_adaptive.slurm`.

## Radial cross-family check

```bash
python -m failure_prediction.radial_formal_freeze
python -m failure_prediction.radial_formal_launch
python -m failure_prediction.radial_formal_analyze
python -m failure_prediction.fig_radial_cross_family
```

PCM: `failure_prediction/data/radial_90_8_10` (Scruby–Hillmann–Roffe companion matrices). `K=1000` is a real truncation (`k_free = 3289`).

## Supporting studies

- Decoder cost/LER survey: `failure_prediction/decoder_survey.py`
- OSD-CS mechanism / coset: `osd_cs_mechanism.py`, `osd_cs_coset.py`
- BP–OSD disagreement vs residual: `audit_residual_control.py`
- Hardware-scale screen: `h2_*.py`

## Frozen scientific choices

Do not retune from `d_H` after seeing routing results:

- BB: `bb_144_12_12`, 24-round Z-memory, `p ∈ {0.006, 0.007, 0.008}`
- Radial: `radial_90_8`, 12-round Z-memory, `p ∈ {0.008, 0.009}`
- BP: normalised min-sum, scale `0.625`, 20 iterations, flooding schedule
- Search score `S(x)` as above; `K = 1000`
