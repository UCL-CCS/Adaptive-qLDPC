# Formal cross-family validation: radial / lifted-product `radial_90_8`

Z-memory circuit-level Adaptive V1 pipeline. Pilot seed 21 excluded. Manuscript not edited.
Distance is **unknown**; this is not labelled [[90,8,10]].

## 1. Frozen configuration and provenance

- code: `radial_90_8`, (r,s)=(3,5), n=90, k=8, d=unknown
- PCM: `/home/di49mer/QIML/QEC/failure_prediction/data/radial_90_8_10` hashes `{"hx_csv": "110c48289e2023ff4dff2ecd809f14360d327242309384cf507f30ac51d84c4a", "hz_csv": "6ab78c47750150aa0ce050531c03e93590c56ccd89ea4ebb6f09b8625aea7754", "lx_csv": "5a8c90c5503c7f95d78eb7fdac2e1bb7996d9d6093f4c63151f42ded81a8b4f7", "lz_csv": "0d9ad5aba7c9adb3626755339b5492eb8856d7fb3972d2a7666ea4506acec8e2", "hx_array": "b8a57173104da2bc2876398582df65a12861249398f4d6c1ef13ed2a028ac3c3", "hz_array": "eafdaae0e44e413214e949e830cc424fc99078d0cea966617eabbc4587566ba9", "lx_array": "be9f829b2ca9b14a97db0f8665783db51beb018e90fde3eee638cb6c4b041fd8", "lz_array": "9113a141bbc652e8358339c418f9f67fd33d74c4f93cfa56561970dd15b11e6c"}`
- circuit: `{'builder': 'failure_prediction.qldpc_circuit.build_z_memory_circuit', 'basis': 'Z', 'rounds': 12, 'noise': 'Adaptive V1 Z-memory: post-reset X_ERROR(p) on data and Z-ancillas; CX DEPOLARIZE2(p); M(p); data DEPOLARIZE1(p) once per round; final data M(p)', 'dem': {'decompose_errors': True, 'ignore_decomposition_failures': True}, 'not_used': 'full-CSS production circuit (noiseless-ok in pilot; k_free too large for full 1FV)'}`
- p: `[0.008, 0.009]` (frozen from blind headroom; not retuned)
- decoder: `{"bp_method": "ms", "ms_scaling_factor": 0.625, "schedule": "flooding/parallel (ldpc BpOsdDecoder default)", "max_iter": 20, "osd0_fast_path": "if H e_BP = s return e_BP; else algebraic OSD-0 (free variables zero)", "score": "S(x)=sum_{i:x_i=1} log(1/p_i) on Stim DEM channel priors; lower is better via argmax \u0394", "score_is_not": "exact Bernoulli ML", "K": 1000, "k_free_pilot": 3289, "full_1fv": "all one-free-variable OSD perturbations in the same LPR ordering", "tie_break_topk": "numpy.argsort(-score, kind='mergesort')", "tie_break_1fv": "argmax \u0394; keep OSD-0 if \u0394<=0"}`
- software: `{"git_commit": "7af1d4b36cc3cff2c3c2bce1f1e08a47cf86d0ed", "python": "3.13.11 | packaged by Anaconda, Inc. | (main, Dec 10 2025, 21:28:48) [GCC 14.3.0]", "numpy": "2.4.3", "stim": "1.16.0", "ldpc": "2.4.1", "scipy": "1.18.0"}`

## 2. Formal dataset

- calib seeds: `{'0.008': [9000, 9001, 9002, 9003, 9004, 9005, 9006, 9007, 9008, 9009], '0.009': [9200, 9201, 9202, 9203, 9204, 9205, 9206, 9207, 9208, 9209]}` (5,000 / p)
- test seeds: `{'0.008': [9100, 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108, 9109, 9110, 9111, 9112, 9113, 9114, 9115, 9116, 9117, 9118, 9119, 9120, 9121, 9122, 9123, 9124, 9125, 9126, 9127, 9128, 9129, 9130, 9131, 9132, 9133, 9134, 9135, 9136, 9137, 9138, 9139], '0.009': [9300, 9301, 9302, 9303, 9304, 9305, 9306, 9307, 9308, 9309, 9310, 9311, 9312, 9313, 9314, 9315, 9316, 9317, 9318, 9319, 9320, 9321, 9322, 9323, 9324, 9325, 9326, 9327, 9328, 9329, 9330, 9331, 9332, 9333, 9334, 9335, 9336, 9337, 9338, 9339]}` (20,000 / p)
- excluded pilot seed: 21
- bootstrap seed 20260908, N_BOOT=800, N_RANDOM=200

## 3. Formal decoder headroom

| p | OSD-0 | always K=1000 | full 1FV | k_free | K_eff |
|---|---|---|---|---|---|
| 0.008 | 0.05130 [0.04840,0.05420] (1026/20000) | 0.02095 [0.01900,0.02290] (419/20000) | 0.02090 [0.01910,0.02290] (418/20000) | 3289 | 1000 |
| 0.009 | 0.10160 [0.09765,0.10570] (2032/20000) | 0.04660 [0.04375,0.04970] (932/20000) | 0.04655 [0.04395,0.04970] (931/20000) | 3289 | 1000 |

| p | branch | CC | CW | WC | WW | net |
|---|---|---|---|---|---|---|
| 0.008 | K1000 | 18924 | 50 | 657 | 369 | 607 |
| 0.008 | full1FV | 18922 | 52 | 660 | 366 | 608 |
| 0.009 | K1000 | 17882 | 86 | 1186 | 846 | 1100 |
| 0.009 | full1FV | 17880 | 88 | 1189 | 843 | 1101 |

H e_OSD0 = s rate: p=0.008 → 100.00%, p=0.009 → 100.00%.

## 4. Risk discrimination

Preregistered orientation: larger score = higher predicted OSD-0 failure risk. `-d_H` AUROC is post hoc only.

| p | BP conv / He_BP=s | d_H AUROC | r_BP AUROC | w_s AUROC | −d_H AUROC (post hoc) | Spearman(d_H,r_BP) |
|---|---|---|---|---|---|---|
| 0.008 | 21.04% / 21.04% | 0.951 [0.946,0.956] | 0.832 [0.821,0.842] | 0.707 [0.691,0.723] | 0.049 [0.044,0.054] | 0.953 |
| 0.009 | 12.45% / 12.45% | 0.940 [0.935,0.944] | 0.809 [0.801,0.818] | 0.701 [0.690,0.713] | 0.060 [0.056,0.064] | 0.931 |

## 5. Exact matched-budget routing

Primary endpoint: f=20%, escalate to K=1000. Tie-break: stable argsort of −score.

| p | f | d_H LER | r_BP LER | w_s LER | random LER | R_dH |
|---|---|---|---|---|---|---|
| 0.008 | 10% | 0.02705 (541/20000) 0.02705 [0.02450,0.02920] | 0.03905 (781/20000) | 0.04365 (873/20000) | 0.04828 | 0.798 |
| 0.008 | 20% | 0.02275 (455/20000) 0.02275 [0.02060,0.02445] | 0.03290 (658/20000) | 0.03860 (772/20000) | 0.04525 | 0.939 |
| 0.008 | 30% | 0.02145 (429/20000) 0.02145 [0.01950,0.02350] | 0.02805 (561/20000) | 0.03390 (678/20000) | 0.04225 | 0.982 |
| 0.009 | 10% | 0.06585 (1317/20000) 0.06585 [0.06215,0.06975] | 0.08430 (1686/20000) | 0.08870 (1774/20000) | 0.09613 | 0.649 |
| 0.009 | 20% | 0.05390 (1078/20000) 0.05390 [0.05065,0.05740] | 0.07275 (1455/20000) | 0.07975 (1595/20000) | 0.09054 | 0.866 |
| 0.009 | 30% | 0.04950 (990/20000) 0.04950 [0.04660,0.05235] | 0.06495 (1299/20000) | 0.07345 (1469/20000) | 0.08503 | 0.946 |

## 6. d_H versus BP residual

| p | f | ΔLER (d_H − r_BP) 95% paired CI | Δ fails |
|---|---|---|---|
| 0.008 | 10% | -0.01200 [-0.01360,-0.01060] | -240 |
| 0.008 | 20% | -0.01015 [-0.01195,-0.00880] | -203 |
| 0.008 | 30% | -0.00660 [-0.00775,-0.00525] | -132 |
| 0.009 | 10% | -0.01845 [-0.02010,-0.01670] | -369 |
| 0.009 | 20% | -0.01885 [-0.02060,-0.01690] | -377 |
| 0.009 | 30% | -0.01545 [-0.01710,-0.01375] | -309 |

## 7. Rescue/harm analysis

Exact top-20% routed subset (escalated shots only):

| p | score | n_esc | WC | CW | CC | WW | net |
|---|---|---|---|---|---|---|---|
| 0.008 | d_H | 4000 | 611 | 40 | 3017 | 332 | 571 |
| 0.008 | r_BP | 4000 | 392 | 24 | 3335 | 249 | 368 |
| 0.008 | w_s | 4000 | 270 | 16 | 3535 | 179 | 254 |
| 0.009 | d_H | 4000 | 1007 | 53 | 2218 | 722 | 954 |
| 0.009 | r_BP | 4000 | 625 | 48 | 2801 | 526 | 577 |
| 0.009 | w_s | 4000 | 472 | 35 | 3083 | 410 | 437 |

Secondary d_H bins (test set):

| p | bin | n | OSD-0 LER | WC | CW |
|---|---|---|---|---|---|
| 0.008 | [-0.5,5.0) | 8764 | 0.00011 | 0 | 1 |
| 0.008 | [5.0,12.0) | 6810 | 0.00954 | 35 | 8 |
| 0.008 | [12.0,17.0) | 2345 | 0.06780 | 99 | 13 |
| 0.008 | [17.0,21.0) | 957 | 0.19854 | 125 | 10 |
| 0.008 | [21.0,64.5) | 1124 | 0.54359 | 398 | 18 |
| 0.009 | [-0.5,9.0) | 9976 | 0.00291 | 12 | 3 |
| 0.009 | [9.0,17.0) | 5753 | 0.04033 | 140 | 27 |
| 0.009 | [17.0,23.0) | 2102 | 0.20171 | 251 | 22 |
| 0.009 | [23.0,29.0) | 1090 | 0.44128 | 308 | 23 |
| 0.009 | [29.0,65.5) | 1079 | 0.80259 | 475 | 11 |

## 8. Held-out calibration

τ from 5k calib targeting 20%; freeze; apply to 20k test. Not matched-budget if realised f differs.

| p | signal | τ | calib f | test f | test LER |
|---|---|---|---|---|---|
| 0.008 | d_h | 12 | 22.66% | 22.13% | 0.02225 (445/20000) |
| 0.008 | r_bp | 19 | 20.58% | 20.20% | 0.03285 (657/20000) |
| 0.008 | syn_weight | 91 | 22.06% | 21.52% | 0.03765 (753/20000) |
| 0.009 | d_h | 18 | 20.10% | 19.09% | 0.05500 (1100/20000) |
| 0.009 | r_bp | 25 | 21.14% | 20.11% | 0.07260 (1452/20000) |
| 0.009 | syn_weight | 100 | 21.82% | 20.87% | 0.07915 (1583/20000) |

Tie rule: escalate iff `score >= τ`, `τ = quantile(calib, 0.80)`.

## 9. Statistical uncertainty

LER and AUROC: nonparametric bootstrap, N_BOOT=800, 95% percentile intervals. Strategy comparisons: paired bootstrap of the same shots. Random routing: 200 seeds. Raw failure counts are reported with every LER.

## 10. Reproducibility paths

- root: `outputs/failure_prediction/nonbb_radial_formal`
- frozen config: `outputs/failure_prediction/nonbb_radial_formal/radial_90_8_frozen_config.json`
- figure: `outputs/failure_prediction/nonbb_radial_formal/radial_cross_family_ler_vs_escalation.png`
- per-shot: `calib/*.npz`, `test/*.npz`
- masks: `routing_masks_p*.npz`
- summaries: `summary.json`, `summary.csv`, `bootstrap_core.json`
- this report: `outputs/failure_prediction/nonbb_radial_formal/radial_cross_family_formal_report.md`

## 11. Final scientific interpretation

On both frozen physical error rates, larger `d_H` predicts higher OSD-0 logical-failure risk in the same orientation as the BB study (AUROC 0.951 and 0.940; CIs well above 0.5). The post-hoc `−d_H` AUROCs are 0.049 and 0.060 and are not used as the primary result.

Formal 1FV headroom is large and stable: OSD-0 LER 5.130% → full 1FV 2.090% at p=0.008 (WC=660, CW=52, net=608) and 10.160% → 4.655% at p=0.009 (WC=1189, CW=88, net=1101). Always-K=1000 is essentially identical to full 1FV (419 vs 418 failures; 932 vs 931), so K=1000 is a genuine truncated search that still captures almost all one-free-variable headroom.

Exact top-20% routing to K=1000 recovers most of that gap when ranked by `d_H` (R=0.939 and 0.866). Residual ranking recovers less (R=0.605 and 0.525); syndrome weight and random recover still less. Paired bootstrap ΔLER = LER(d_H)−LER(r_BP) at 20% is −0.01015 [−0.01195, −0.00880] and −0.01885 [−0.02060, −0.01690] (203 and 377 fewer failures). Inside the routed 20% subset, `d_H` identifies more rescuable OSD-0 failures (WC=611 and 1007) than residual (392 and 625) or syndrome weight (270 and 472).

This is a structurally distinct radial / lifted-product code, not a BB cousin. The result supports adding it as a cross-family validation. The manuscript has not been edited.

### Explicit answers

1. Does d_H predict OSD-0 logical failure on radial_90_8? **Yes.** AUROC 0.951 [0.946, 0.956] at p=0.008 and 0.940 [0.935, 0.944] at p=0.009.
2. Is the orientation the same as on BB? **Yes.** Larger `d_H` predicts higher OSD-0 failure risk. The sign was not flipped.
3. Does d_H outperform BP residual? **Yes.** At every frozen p and budget, paired ΔLER is negative with 95% CIs excluding 0. Residual AUROC is high (0.832 / 0.809) but strictly weaker than `d_H`.
4. Does K=1000 retain useful formal headroom? **Yes.** Relative to OSD-0, always-K=1000 cuts failures 1026→419 and 2032→932, matching full 1FV to within one shot.
5. At exact 20% escalation, does d_H routing lower LER relative to residual, syndrome weight and random? **Yes.** p=0.008: 0.02275 vs 0.03290 / 0.03860 / 0.04525. p=0.009: 0.05390 vs 0.07275 / 0.07975 / 0.09054. Bootstrap CIs do not overlap.
6. What fraction of the full1FV headroom does the 20% policy recover? **R(d_H)=0.939 at p=0.008 and 0.866 at p=0.009.** Residual recovers 0.605 and 0.525; syndrome weight 0.418 and 0.397; random ≈0.20.
7. Does a calibration-derived threshold transfer to the independent test set? **Yes, approximately, but it is not a matched-budget comparison.** Discrete-score `τ` targeting 20% realises 22.13% / 19.09% (`d_H`) on test; test LERs (0.02225 / 0.05500) are close to exact top-20%. Exact top-k remains primary.
8. Is the result strong enough to support adding this code as a cross-family validation in the paper? **YES.** Do not edit the manuscript in this step.

CROSS-FAMILY RESULT: **STRONG TRANSFER**

RECOMMEND MAIN-MANUSCRIPT INCLUSION: **YES**

