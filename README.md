# Adaptive qLDPC

Adaptive decoding for quantum LDPC codes.

```
BP@20 + OSD-0  →  d_H(e_BP, e_OSD0)  →  OSD-0 or truncated 1FV search (K=1000)
```

`d_H` is the Hamming distance between the BP hard decision and the OSD-0 reconstruction. The expensive branch perturbs one OSD free variable at a time using

```
S(x) = sum_{i: x_i = 1} log(1/p_i)
```

on Stim DEM channel probabilities.

Primary code: bivariate bicycle `bb_144_12_12`. Cross-family check: radial / lifted-product `radial_90_8` (`n=90`, `k=8`, distance unknown).

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=.
```

Run from the repository root.

## Decode shots

```bash
python -m failure_prediction.adaptive_v1_run \
    --code bb_144_12_12 --p 0.007 --rounds 24 \
    --shots 500 --seed 3000 --max-iter 20 \
    --out outputs/failure_prediction/adaptive_v1/parts/p0.007_seed3000.npz
```

Cluster: `sbatch scripts/submit_adaptive_v1.slurm`

## Radial cross-family check

```bash
python -m failure_prediction.radial_formal_freeze
python -m failure_prediction.radial_formal_launch
python -m failure_prediction.radial_formal_analyze
```

PCM: `failure_prediction/data/radial_90_8_10`.
