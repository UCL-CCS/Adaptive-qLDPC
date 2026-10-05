# Adaptive V1 controls: BP-converge AUROC and held-out $\tau$

Existing bb144 $n=20000$ matched shots. No new Monte Carlo.

## 1. BP@20 converge flag as a router

Score $=1-$ `BP@20.converge` (non-convergence = high OSD-0-failure risk).

| $p$ | conv. frac. | OSD-0 fails in conv. / not | LER $\mid$ conv. | LER $\mid$ not | BP-flag fail AUROC | $d_H$ fail AUROC | $d_H$ AUROC on non-conv. | natural $f_{\mathrm{esc}}$ (escalate all non-conv.) | natural LER |
|---|---|---|---|---|---|---|---|---|---|
| 0.006 | 9.18% (1835/20000) | 0 / 95 | 0.00000 | 0.00523 | 0.546 [0.544,0.548] | 0.958 [0.941,0.973] | 0.954 [0.936,0.969] | 90.83% | 0.00115 |
| 0.007 | 3.06% (611/20000) | 0 / 398 | 0.00000 | 0.02053 | 0.516 [0.514,0.517] | 0.958 [0.951,0.965] | 0.957 [0.949,0.964] | 96.95% | 0.00475 |
| 0.008 | 0.68% (136/20000) | 0 / 1255 | 0.00000 | 0.06318 | 0.504 [0.503,0.504] | 0.937 [0.931,0.942] | 0.936 [0.930,0.942] | 99.32% | 0.02040 |

The binary flag cannot implement a 10–30% compute budget: not-converged is 91–99% of shots.
Escalating every non-converged shot is essentially always-$K{=}1000$.
$d_H$ remains predictive on the non-converged majority.

## 2. Held-out $\tau$ (20 seeds calibrate, 20 evaluate; then swap)

Calibrate $\tau$ as the $k$-th largest $d_H$ on the calibration seeds, $k=\mathrm{round}(f n_{\mathrm{cal}})$. Test policy: escalate iff $d_H \ge \tau$. Oracle on the same test seeds is in-sample exact top-$k$.

### $p=0.006$

| fold | $f$ | $\tau_{\mathrm{cal}}$ | $f_{\mathrm{esc}}$ test | hold-out LER | oracle top-$k$ LER | $\Delta$ LER | hold-out rec. | oracle rec. |
|---|---|---|---|---|---|---|---|---|
| seeds_first20_cal | 10% | 18 | 10.39% | 0.00160 [0.00099,0.00260] | 0.00180 | -0.00020 | 86.5% | 81.1% |
| seeds_first20_cal | 20% | 14 | 20.86% | 0.00150 [0.00091,0.00247] | 0.00150 | +0.00000 | 89.2% | 89.2% |
| seeds_first20_cal | 30% | 11 | 33.11% | 0.00140 [0.00083,0.00235] | 0.00140 | +0.00000 | 91.9% | 91.9% |
| seeds_last20_cal | 10% | 18 | 10.42% | 0.00140 [0.00083,0.00235] | 0.00140 | +0.00000 | 89.2% | 89.2% |
| seeds_last20_cal | 20% | 14 | 20.96% | 0.00120 [0.00069,0.00210] | 0.00120 | +0.00000 | 94.6% | 94.6% |
| seeds_last20_cal | 30% | 11 | 33.51% | 0.00100 [0.00054,0.00184] | 0.00100 | +0.00000 | 100.0% | 100.0% |

Mean over the two complementary folds:

| $f$ | mean $\tau$ | mean $f_{\mathrm{esc}}$ | mean hold-out LER | mean oracle LER | mean $\Delta$ | mean rec. hold-out |
|---|---|---|---|---|---|---|
| 10% | 18.0 | 10.41% | 0.00150 | 0.00160 | -0.00010 | 87.8% |
| 20% | 14.0 | 20.91% | 0.00135 | 0.00135 | +0.00000 | 91.9% |
| 30% | 11.0 | 33.31% | 0.00120 | 0.00120 | +0.00000 | 95.9% |

### $p=0.007$

| fold | $f$ | $\tau_{\mathrm{cal}}$ | $f_{\mathrm{esc}}$ test | hold-out LER | oracle top-$k$ LER | $\Delta$ LER | hold-out rec. | oracle rec. |
|---|---|---|---|---|---|---|---|---|
| seeds_first20_cal | 10% | 27 | 9.96% | 0.00790 [0.00634,0.00983] | 0.00790 | +0.00000 | 79.2% | 79.2% |
| seeds_first20_cal | 20% | 21 | 21.45% | 0.00530 [0.00405,0.00693] | 0.00540 | -0.00010 | 95.6% | 95.0% |
| seeds_first20_cal | 30% | 18 | 30.38% | 0.00500 [0.00379,0.00659] | 0.00500 | +0.00000 | 97.5% | 97.5% |
| seeds_last20_cal | 10% | 26 | 11.41% | 0.00620 [0.00484,0.00794] | 0.00650 | -0.00030 | 83.4% | 81.5% |
| seeds_last20_cal | 20% | 21 | 22.00% | 0.00520 [0.00397,0.00681] | 0.00540 | -0.00020 | 89.8% | 88.5% |
| seeds_last20_cal | 30% | 18 | 31.10% | 0.00500 [0.00379,0.00659] | 0.00510 | -0.00010 | 91.1% | 90.4% |

Mean over the two complementary folds:

| $f$ | mean $\tau$ | mean $f_{\mathrm{esc}}$ | mean hold-out LER | mean oracle LER | mean $\Delta$ | mean rec. hold-out |
|---|---|---|---|---|---|---|
| 10% | 26.5 | 10.69% | 0.00705 | 0.00720 | -0.00015 | 81.3% |
| 20% | 21.0 | 21.73% | 0.00525 | 0.00540 | -0.00015 | 92.7% |
| 30% | 18.0 | 30.74% | 0.00500 | 0.00505 | -0.00005 | 94.3% |

### $p=0.008$

| fold | $f$ | $\tau_{\mathrm{cal}}$ | $f_{\mathrm{esc}}$ test | hold-out LER | oracle top-$k$ LER | $\Delta$ LER | hold-out rec. | oracle rec. |
|---|---|---|---|---|---|---|---|---|
| seeds_first20_cal | 10% | 40 | 9.81% | 0.03230 [0.02901,0.03595] | 0.03210 | +0.00020 | 68.3% | 68.7% |
| seeds_first20_cal | 20% | 32 | 20.81% | 0.02410 [0.02127,0.02729] | 0.02440 | -0.00030 | 86.5% | 85.8% |
| seeds_first20_cal | 30% | 27 | 31.91% | 0.02140 [0.01874,0.02443] | 0.02160 | -0.00020 | 92.5% | 92.0% |
| seeds_last20_cal | 10% | 39 | 11.00% | 0.03260 [0.02929,0.03626] | 0.03380 | -0.00120 | 69.1% | 66.4% |
| seeds_last20_cal | 20% | 32 | 20.85% | 0.02540 [0.02249,0.02867] | 0.02570 | -0.00030 | 85.8% | 85.2% |
| seeds_last20_cal | 30% | 27 | 31.46% | 0.02270 [0.01996,0.02581] | 0.02290 | -0.00020 | 92.1% | 91.6% |

Mean over the two complementary folds:

| $f$ | mean $\tau$ | mean $f_{\mathrm{esc}}$ | mean hold-out LER | mean oracle LER | mean $\Delta$ | mean rec. hold-out |
|---|---|---|---|---|---|---|
| 10% | 39.5 | 10.41% | 0.03245 | 0.03295 | -0.00050 | 68.7% |
| 20% | 32.0 | 20.83% | 0.02475 | 0.02505 | -0.00030 | 86.2% |
| 30% | 27.0 | 31.68% | 0.02205 | 0.02225 | -0.00020 | 92.3% |

In-sample top-$k$ on the evaluation seeds does not materially beat a $\tau$ frozen on the complementary seeds.
The paper's main curves remain budget-controlled exact top-$k$; this table is the independent-calibration check.
