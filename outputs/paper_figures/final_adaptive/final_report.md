# Final Adaptive Decoder V1 — simulation close-out

Frozen method: BP@20 + OSD-0 → $d_H$ → truncated 1FV, $K=1000$.
bb144, $p\in\{0.006,0.007,0.008\}$, $n=20000$ matched shots.

## Q1–Q2  Routing and recovered gap ($K=1000$)

### p=0.006  n=20000  OSD-0 fails=95
- OSD-0 LER 0.00475 [0.00380,0.00570]
- full 1FV LER 0.00105 [0.00060,0.00155]
- always $K=1000$ LER 0.00115
- headroom 4.52×
- $d_H$ fail AUROC 0.958 [0.941,0.974]
- $d_H$ beneficial AUROC 0.965 [0.949,0.978]
- BP@20 converged 9.18% (not used as a 10–30% router)

- **10%**  $d_H$ LER 0.00160 [0.00100,0.00210]  rec 85.1% [77.8,96.6]  | syn-wt LER 0.00390 [0.00305,0.00480] rec 23.0%  | random LER 0.00440 [0.00415,0.00465] rec 9.3%  — beats random, beats syn-wt
- **20%**  $d_H$ LER 0.00135 [0.00085,0.00190]  rec 91.9% [84.7,98.8]  | syn-wt LER 0.00365 [0.00285,0.00445] rec 29.7%  | random LER 0.00402 [0.00370,0.00435] rec 19.7%  — beats random, beats syn-wt
- **30%**  $d_H$ LER 0.00120 [0.00075,0.00170]  rec 95.9% [89.0,101.6]  | syn-wt LER 0.00290 [0.00215,0.00360] rec 50.0%  | random LER 0.00368 [0.00330,0.00405] rec 28.9%  — beats random, beats syn-wt

### p=0.007  n=20000  OSD-0 fails=398
- OSD-0 LER 0.01990 [0.01800,0.02190]
- full 1FV LER 0.00410 [0.00320,0.00510]
- always $K=1000$ LER 0.00475
- headroom 4.85×
- $d_H$ fail AUROC 0.958 [0.951,0.965]
- $d_H$ beneficial AUROC 0.957 [0.949,0.964]
- BP@20 converged 3.06% (not used as a 10–30% router)

- **10%**  $d_H$ LER 0.00710 [0.00595,0.00830]  rec 81.0% [76.0,85.6]  | syn-wt LER 0.01595 [0.01405,0.01775] rec 25.0%  | random LER 0.01837 [0.01785,0.01885] rec 9.7%  — beats random, beats syn-wt
- **20%**  $d_H$ LER 0.00545 [0.00445,0.00635]  rec 91.5% [88.3,95.4]  | syn-wt LER 0.01385 [0.01240,0.01560] rec 38.3%  | random LER 0.01687 [0.01625,0.01750] rec 19.2%  — beats random, beats syn-wt
- **30%**  $d_H$ LER 0.00500 [0.00405,0.00600]  rec 94.3% [91.0,97.5]  | syn-wt LER 0.01230 [0.01080,0.01380] rec 48.1%  | random LER 0.01533 [0.01445,0.01610] rec 28.9%  — beats random, beats syn-wt

### p=0.008  n=20000  OSD-0 fails=1255
- OSD-0 LER 0.06275 [0.05960,0.06595]
- full 1FV LER 0.01865 [0.01680,0.02060]
- always $K=1000$ LER 0.02040
- headroom 3.36×
- $d_H$ fail AUROC 0.937 [0.930,0.943]
- $d_H$ beneficial AUROC 0.931 [0.924,0.937]
- BP@20 converged 0.68% (not used as a 10–30% router)

- **10%**  $d_H$ LER 0.03295 [0.03025,0.03575]  rec 67.6% [64.4,70.9]  | syn-wt LER 0.05400 [0.05110,0.05720] rec 19.8%  | random LER 0.05856 [0.05775,0.05945] rec 9.5%  — beats random, beats syn-wt
- **20%**  $d_H$ LER 0.02530 [0.02305,0.02760]  rec 84.9% [82.1,88.0]  | syn-wt LER 0.04840 [0.04540,0.05145] rec 32.5%  | random LER 0.05430 [0.05295,0.05546] rec 19.2%  — beats random, beats syn-wt
- **30%**  $d_H$ LER 0.02215 [0.02020,0.02445]  rec 92.1% [89.5,94.4]  | syn-wt LER 0.04295 [0.04045,0.04580] rec 44.9%  | random LER 0.05002 [0.04875,0.05130] rec 28.9%  — beats random, beats syn-wt

## Q3  Measured serial runtime

Source: `outputs/failure_prediction/adaptive_v1/serial_timing_n1000.json`
- bp20_osd0: mean 10.51 ms  median 9.53 ms  CI [10.15,10.94]
- always_trunc_k500: mean 132.60 ms  median 133.54 ms  CI [130.64,134.66]
- always_trunc_k1000: mean 139.69 ms  median 140.83 ms  CI [137.58,141.88]
- always_full_1fv: mean 273.65 ms  median 278.93 ms  CI [270.25,277.12]
- adapt_k1000_f10: mean 24.47 ms  median 9.53 ms  CI [21.63,27.50]
- adapt_k1000_f20: mean 38.02 ms  median 9.53 ms  CI [34.52,41.47]
- adapt_k1000_f30: mean 51.66 ms  median 9.54 ms  CI [47.80,55.77]
- Mix-model adaptive 20% ≈ 36.3 ms vs full 1FV 273.7 ms (7.5× cheaper). Wall-clock, not candidate-count.

## Q4  bb72 control

Hypothesis: little fast-to-strong headroom ⇒ little absolute adaptive benefit. Not a generalisation claim.
- p=0.006 n=10000  OSD-0=0.04060 [0.03680,0.04460]  1FV=0.03500 [0.03140,0.03860]  headroom 1.16×  fail AUROC 0.831 [0.809,0.853]
  20% $d_H$ LER=0.03580 [0.03210,0.03890]  rec 85.7% of available gap  (absolute drop 0.00480 vs available 0.00560)
  30% $d_H$ LER=0.03530 [0.03190,0.03900]  rec 94.6% of available gap  (absolute drop 0.00530 vs available 0.00560)
  absolute 20% drop 0.00480 vs available 0.00560
- p=0.008 n=10000  OSD-0=0.13390 [0.12670,0.14060]  1FV=0.11010 [0.10350,0.11640]  headroom 1.22×  fail AUROC 0.839 [0.828,0.850]
  20% $d_H$ LER=0.11400 [0.10760,0.12060]  rec 83.6% of available gap  (absolute drop 0.01990 vs available 0.02380)
  30% $d_H$ LER=0.11210 [0.10609,0.11820]  rec 91.6% of available gap  (absolute drop 0.02180 vs available 0.02380)
  absolute 20% drop 0.01990 vs available 0.02380

## Q5  Statistical support

No flagged interval is wide enough to overturn a central claim.
p=0.006 has only 95 OSD-0 failures; LER CIs are wider than at p=0.007/0.008, but the adaptive vs OSD-0 and vs-random/syn-weight gaps remain one-sided.
The p=0.006 30% recovered-gap interval upper bound slightly exceeds 100% (bootstrap with 21 full-1FV failures); the point estimate is 95.9% and the lower bound is 89%.
At p=0.008 and 10% escalation the recovered fraction is 67.6% [64.4, 70.9] — most of the gap is recovered at 20–30% (85% / 92%), not at 10%.
No additional Monte Carlo was generated: existing n=20000 is sufficient for the claims.

## Q6  Zero-cost controls (existing 20k shots; no new Monte Carlo)

Source: `outputs/paper_figures/final_adaptive/controls_bp_tau.md`

### BP@20 converge flag vs $d_H$

Score $=1-$ converge (non-convergence = high risk). Every OSD-0 failure is in the non-converged majority. The converged minority has **0** OSD-0 failures, so it is a perfect “skip” on 0.7–9% of shots — not a 10–30% router. Fail AUROC of the binary flag sits at $0.50$–$0.55$ because that is what a rare perfect-negative class produces. $d_H$ stays at $0.94$–$0.96$, including on the non-converged subset alone. Escalating every non-converged shot is $f_{\mathrm{esc}}=91$–$99\%$ and recovers the always-$K{=}1000$ LER.

| $p$ | conv. | fails in conv. / not | BP-flag AUROC | $d_H$ AUROC | $d_H$ AUROC on non-conv. |
|---|---|---|---|---|---|
| 0.006 | 9.18% | 0 / 95 | 0.546 [0.544, 0.548] | 0.958 [0.941, 0.973] | 0.954 [0.936, 0.969] |
| 0.007 | 3.06% | 0 / 398 | 0.516 [0.514, 0.517] | 0.958 [0.951, 0.965] | 0.957 [0.949, 0.964] |
| 0.008 | 0.68% | 0 / 1255 | 0.504 [0.503, 0.504] | 0.937 [0.931, 0.942] | 0.936 [0.930, 0.942] |

### Held-out $\tau$ (20 seeds calibrate, 20 evaluate; then swap)

$\tau$ = $k$-th largest $d_H$ on calibration seeds, $k=\mathrm{round}(f n_{\mathrm{cal}})$. Test: escalate iff $d_H\ge\tau$. Oracle: in-sample exact top-$k$ on the same test seeds. Mean $\Delta$ LER (hold-out minus oracle) is $\le 0$ within shot noise; recovered-gap tracks the in-sample curves.

| $p$ | $f$ | mean $\tau$ | mean $f_{\mathrm{esc}}$ test | mean hold-out LER | mean oracle LER | mean $\Delta$ |
|---|---|---|---|---|---|---|
| 0.006 | 20% | 14 | 20.9% | 0.00135 | 0.00135 | 0 |
| 0.007 | 20% | 21 | 21.7% | 0.00525 | 0.00540 | $-0.00015$ |
| 0.008 | 20% | 32 | 20.8% | 0.02475 | 0.02505 | $-0.00030$ |

The paper’s main curves remain budget-controlled exact top-$k$. This table is the independent-calibration check.

Notes:
- Routing uses exact top-$k$ on matched shots ($k=\mathrm{round}(f n)$), so LER at ~20% can differ slightly from the earlier $\tau$-quantile sweep.
- Random routing: 200 independent seeds, mean + 2.5/97.5 percentiles.
- Serial timing is n=1000, one process, wolpy16, package OSD-CS skipped. Absolute milliseconds are node-dependent; relative Pareto shape is the claim.
- On bb72, $K=1000$ already saturates full 1FV LER. The router still works (fail AUROC ~0.83) but relative LER stays ~0.85–0.88 of OSD-0, versus ~0.27–0.40 on bb144.

## Verdict

**EXPERIMENTAL PHASE COMPLETE.** $d_H$ outperforms random and syndrome-weight routing at the same $K=1000$ budget; Adaptive V1 recovers most of the available OSD-0→full-1FV gap at 10–30% escalation; measured serial cost remains favourable; bb72 behaves as a low-headroom control. Freeze experiments and begin writing.

## Figures

- `outputs/paper_figures/final_adaptive/fig1_method_schematic.png`
- `outputs/paper_figures/final_adaptive/fig2_router_dh.png`
- `outputs/paper_figures/final_adaptive/fig3_routing_budget.png`
- `outputs/paper_figures/final_adaptive/fig4_runtime_pareto.png`
- `outputs/paper_figures/final_adaptive/fig5_bb72_control.png`
