# Radial / lifted-product qLDPC headroom pilot

Blind to d_H / BP-residual routing. Hamming-HGP experiment removed; BB Adaptive V1 untouched.

## 1. Selected code and why it was selected

- family: quantum radial codes (lifted product of classical quasi-cyclic radial codes)
- instance: published PCM `90_8_10`, (r,s)=(3,5)
- n=90, k=8 (rank formula 8)
- selection: Smallest published PCM in the paper companion repository. Chosen before any decoder or headroom evaluation.

Not a BB code. Not the Hamming-HGP instance. No random construction search.

## 2. Literature/software source

- Scruby, Hillmann, Roffe, arXiv:2406.14445 / PRX Quantum
- https://github.com/tRowans/radial-codes-public @ `12e67584db0fae3dbf8b4447596066e26ad2514c`
- local PCM copy: `failure_prediction/data/radial_90_8_10/`

## 3. Code validation

- H_X [45, 90], rank 41
- H_Z [45, 90], rank 41
- CSS commute H_X H_Z^T = 0: **True**
- row/column weights: `{"hx_shape": [45, 90], "hz_shape": [45, 90], "hx_rank": 41, "hz_rank": 41, "hx_row_weights": {"min": 6, "max": 6, "mean": 6.0, "unique": [6]}, "hz_row_weights": {"min": 6, "max": 6, "mean": 6.0, "unique": [6]}, "hx_col_degrees": {"min": 3, "max": 3, "mean": 3.0, "unique": [3]}, "hz_col_degrees": {"min": 3, "max": 3, "mean": 3.0, "unique": [3]}}`
- distance: **unknown**. Source directory is named 90_8_10 and the family bound is d≤2s=10. Distance is not independently verified in this repository.

## 4. Logical validation

- published logicals: `{'ok': True, 'issues': [], 'k_x': 8, 'k_z': 8, 'pairing_rank': 8, 'n_anticommuting_pairs': 18}`
- pipeline logicals (ker / stabilizer quotient, Adaptive V1 convention): `{'ok': True, 'issues': [], 'k_x': 8, 'k_z': 8, 'pairing_rank': 8, 'n_anticommuting_pairs': 31}`

## 5. Circuit validation

- full-CSS noiseless 64 shots: **True**, ops `{'n_qubits': 180, 'n_meas': 1170, 'n_detectors': 1080, 'n_observables': 8, 'n_cx': 1080, 'n_1q_noise_or_h': 24, 'depth_ticks': 1129}`
- Z-memory noiseless 64 shots: **True**, ops `{'n_qubits': 135, 'n_meas': 630, 'n_detectors': 585, 'n_observables': 8, 'n_cx': 540, 'n_1q_noise_or_h': 0, 'depth_ticks': 565}`
- production circuit: **full_css_z_memory**
- rounds: 12 (not tuned on decoder performance)
- full-CSS DEM decompose_errors=True: {'ok': True, 'error': None}

## 6. Noise / DEM validation

Adaptive V1 circuit-level convention: post-reset X_ERROR(p), CX DEPOLARIZE2(p),
M(p), data DEPOLARIZE1(p) once per round, final data M(p).

- DEM: `{"n_detectors": 1080, "n_observables": 8, "n_error_mechanisms": 42090, "H_shape": [1080, 42090], "rank_H": 1076, "k_free": 41014, "K": 1000, "K_eff": 1000, "K_is_full_1fv": false, "decompose_errors": true}`
- H e_OSD0 = s on production path: recorded in the p-scan `he_osd0_ok` column

## 7. Decoder configuration

`{"bp_method": "ms", "ms_scaling_factor": 0.625, "schedule": "flooding/parallel (ldpc BpOsdDecoder default)", "max_iter": 20, "osd0_fast_path": "if H e_BP = s return e_BP; else algebraic OSD-0 (free vars zero)", "score": "S(x)=sum_{i:x_i=1} log(1/p_i) on Stim DEM channel priors", "score_is_not": "exact Bernoulli ML", "K": 1000, "full_1fv": "all one-free-variable OSD perturbations; not exhaustive physical search", "not_computed": ["d_H", "r_BP", "routing AUROC", "adaptive LER"]}`

- software: `{"git_commit": "7af1d4b36cc3cff2c3c2bce1f1e08a47cf86d0ed", "python": "3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:28:48) [GCC 14.3.0]", "numpy": "2.4.3", "stim": "1.16.0", "ldpc": "2.4.1", "scipy": "1.18.0"}`

## 8. Pilot p scan

Independent seed 21, n=500 shots/p. Not a production dataset.

| p | OSD-0 LER | K=1000 LER | full1FV LER | rel. impr. | ratio | WC | CW | net | he_ok | ms/shot | criterion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0001 | 0.0000 (0/500) | 0.0000 | 0.0000 | nan | None | 0 | 0 | 0 | 1.000 | 265.3 | False |
| 0.0003 | 0.0000 (0/500) | 0.0000 | 0.0000 | nan | None | 0 | 0 | 0 | 1.000 | 311.9 | False |
| 0.0005 | 0.0000 (0/500) | 0.0000 | 0.0000 | nan | None | 0 | 0 | 0 | 1.000 | 403.7 | False |
| 0.001 | 0.0100 (5/500) | 0.0080 | 0.0060 | 0.4 | 1.6666666666666667 | 3 | 1 | 2 | 1.000 | 568.8 | False |
| 0.002 | 0.1140 (57/500) | 0.0420 | 0.0440 | 0.6140350877192983 | 2.5909090909090913 | 35 | 0 | 35 | 1.000 | 793.3 | True |
| 0.003 | 0.3160 (158/500) | 0.2060 | 0.2000 | 0.36708860759493667 | 1.5799999999999998 | 58 | 0 | 58 | 1.000 | 1138.9 | True |
| 0.004 | 0.6380 (319/500) | 0.5060 | 0.5120 | 0.1974921630094044 | 1.24609375 | 65 | 2 | 63 | 1.000 | 1696.7 | False |
| 0.005 | 0.8660 (433/500) | 0.7940 | 0.7880 | 0.09006928406466508 | 1.098984771573604 | 40 | 1 | 39 | 1.000 | 2272.3 | False |
| 0.006 | 0.9500 (475/500) | 0.9260 | 0.9160 | 0.03578947368421044 | 1.0371179039301308 | 20 | 3 | 17 | 1.000 | 2939.4 | False |

## 9. OSD0 / K1000 / full1FV headroom

Pre-registered useful-headroom rule: relative improvement (LER_OSD0 − LER_full1FV)/LER_OSD0 ≥ 0.2 and WC ≥ 10 with net ≥ 5 on this pilot sample.

Points meeting the criterion: `[0.002, 0.003]`

## 10. Rescue / harm counts

See WC (rescues) and CW (harms) in the scan table. net = WC − CW.

## 11. k_free and computational feasibility

- k_free = 41014
- K_eff = 1000; K=1000 is full 1FV? False
- typical ms/shot from the scan (see table). Full 1FV is the OSD combination sweep over free variables, not exhaustive physical-error search.

## 12. GO / NO-GO conclusion

GO: published radial code/circuit/DEM validated; full 1FV feasible; at least one p meets the pre-registered headroom criterion. Ready for a later d_H test. d_H was not inspected.

FORMAL CROSS-FAMILY TEST READY: **YES**

Frozen for a later d_H test (not run in this task):
- code: radial_90_8, (r,s)=(3,5)
- circuit: full_css_z_memory, rounds=12
- recommended p: [0.002]
- decoder: Adaptive V1 BP/OSD-0/1FV conventions unchanged

Do not inspect d_H until that follow-up is requested.

