# Adaptive QEC audit + BP residual-weight control

No manuscript TeX/figures were edited. No bb144/bb72 Monte Carlo labels were regenerated.
Primary routing convention throughout: exact top-$k$ on the evaluation sample, $K=1000$ unless stated.

Matched-shot source: `outputs/failure_prediction/adaptive_v1/parts/` (40 seeds × 500 shots per $p$; seeds 3000–3039 / 4000–4039 / 2000–2039).

## 1. Candidate scoring audit

**STATUS: PASS** (expression is $\log(1/p_i)$, not Bernoulli NLL)

production script/path: `failure_prediction/adaptive_v1.py` (`score_one_free_prefix`, `pick_one_free`); score vector built in `failure_prediction/adaptive_v1_run.py` as `log_inv_p = np.log(1.0 / matrices.error_probs)`.

production output/path: `outputs/failure_prediction/adaptive_v1/parts/*.npz` (`fail_k500`, `fail_k1000`, `fail_full`).

exact result:

- Production score is $S(x)=\sum_{i:x_i=1}\log(1/p_i)$ on **channel** probabilities from the Stim DEM.
- It is **not** $S_{\mathrm{ML}}(x)=\sum_{i:x_i=1}\log((1-p_i)/p_i)$.
- Selection is **lower $S$**: `pick_one_free` maximises $\Delta=S(e_{\mathrm{OSD0}})-S(e_j)$ and keeps OSD-0 if $\max\Delta\le 0$ (`adaptive_v1.py` 85–93).
- The same $S$ and the same scored free-column list are used for $K=500$, $K=1000$, and full 1FV; only the prefix length $k$ changes.
- No `ldpc` library function reweights this score afterwards. Truncated 1FV is entirely our Python reconstruction.
- Describe $S$ as a **prior-weighted / approximate channel score** (OSD-CS package surrogate), not exact independent-Bernoulli NLL.

implication for manuscript: keep $S(x)=\sum \log(1/p_i)$ if that is what the text says; do not call it exact Bernoulli NLL.

## 2. OSD-0 implementation audit

**STATUS: PASS** on algorithm (free variables **set to zero**). **STATUS: ISSUE** if the manuscript claims an explicit OSD-0 reconstruction on *every* shot, including BP-converged shots.

production script/path:

- Library OSD-0 used for reported `fail_osd0`: `ldpc` 2.4.1 `BpOsdDecoder.osd0_decoding` (`site-packages/ldpc/bposd_decoder/_bposd_decoder.pyx` 264–280).
- Independent reconstruction used for 1FV: `failure_prediction/osd_internals.py` `osd0_information_set`.
- Shot loop: `failure_prediction/adaptive_v1.py` `evaluate_shot` 116–140.

exact algorithm:

1. Columns ordered by **ascending BP log-probability ratio** (`np.argsort(llr, kind='stable')`; upstream `soft_decision_col_sort`).
2. Information set = first $\mathrm{rank}(H)$ linearly independent columns in that order (greedy RREF).
3. **Free / non-pivot bits are held at 0.** Pivot bits are the reduced RHS. They do **not** inherit BP hard decisions.
4. On BP-converged shots, `ldpc` **never enters OSD**. `osd0_decoding` is copied from `bpd.decoding` (pyx 273–276). Reported OSD-0 LER on those shots is therefore the BP output. bb144 DEM: $H$ is $1800\times 12240$, rank $1794$, $k_{\mathrm{free}}=10446$.
5. `decoder.decode()` on non-converged shots returns `osdw_decoding`; production LER does **not** use that return value. It uses the `osd0_decoding` property, then our 1FV search on the reconstructed information set.
6. $H e = s$: verified on a 50-shot probe (49/49 non-converged) and on the residual replay (`he_ok_frac` below). Bit-level reconstruction vs library `osd0_decoding` on non-converged shots was previously gated in `verify_osd_reconstruction.py` / `verify_against_decoder`.

relation $e_{\mathrm{BP}}$ vs $e_{\mathrm{OSD0}}$: $d_H=\|e_{\mathrm{BP}}\oplus e_{\mathrm{OSD0}}\|_0$. On converged shots this is $0$ by construction (both fields are BP).

implication for manuscript: say that OSD-0 zeros free variables and that BP-converged shots skip OSD and reuse $e_{\mathrm{BP}}$. Do not say every analysed shot uses an independent-set reconstruction distinct from BP.

## 3. Random-routing endpoint audit

**STATUS: PASS**

production script/path: `failure_prediction/final_adaptive.py` `random_lers` (160–170) and `analyse_p` (uses `fail_k1000`).
Figure 3: `fig3_routing` plots random against `fail_k1000` and draws full 1FV as a **horizontal dashed visual floor** (`axhline`), while the random curve at $f=1$ is always-$K{=}1000$.

exact result (200-seed random top-$k$, escalate to $K=1000$):

| $p$ | $f$ | analytic $(1-f)\mathrm{LER}_0+f\mathrm{LER}_{K1000}$ | 200-seed mean | $\Delta$ |
|---|---|---|---|---|
| 0.006 | 10% | 0.00439 | 0.00439 | -0.000004 |
| 0.006 | 20% | 0.00403 | 0.00405 | +0.000022 |
| 0.006 | 30% | 0.00367 | 0.00365 | -0.000017 |
| 0.007 | 10% | 0.01839 | 0.01840 | +0.000010 |
| 0.007 | 20% | 0.01687 | 0.01690 | +0.000035 |
| 0.007 | 30% | 0.01535 | 0.01538 | +0.000029 |
| 0.008 | 10% | 0.05852 | 0.05848 | -0.000038 |
| 0.008 | 20% | 0.05428 | 0.05428 | +0.000000 |
| 0.008 | 30% | 0.05004 | 0.05000 | -0.000041 |

$p=0.007$, $f=0.20$: analytic $0.8\times 0.01990 + 0.2\times 0.00475 = 0.01687$; measured mean $0.01690$. This **exactly explains** the manuscript random value (Monte Carlo noise $<5\times 10^{-5}$).

implication for manuscript: random routing escalates to $K=1000$, not full 1FV. The full-1FV line in Fig. 3 is a visual accuracy floor only.

## 4. Timing provenance audit

**STATUS: PASS** on the numerical origin of 24.5 / 38.0 / 51.7 ms. **STATUS: ISSUE** if the manuscript presents $K{=}500/1000$ times as independently timed truncated loops.

production script/path: `failure_prediction/adaptive_v1_timing.py` + `scripts/submit_final_adaptive.slurm`
production output/path: `outputs/failure_prediction/adaptive_v1/serial_timing_n1000.json` ; log `outputs/failure_prediction/adaptive_v1/logs/av1_final_20304.out` (host `wolpy16`).

exact result:

- Sample: $n=1000$, $p=0.007$, Stim seed 9001, **one pass**, **no warm-up discarded**.
- Threads: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`. `BpOsdDecoder` is not given an explicit `omp_thread_count`; process-level OpenMP is 1.
- Adaptive 24.5 / 38.0 / 51.7 ms are **reconstructed**, not a second end-to-end strategy that skips the strong branch at runtime. After `evaluate_shot` (which always scores the full free list on non-converged shots), `adaptive_v1_timing.py` 169–181 does exact top-$k$ on that sample’s $d_H$ and sets `t_ad = t_osd0 + 1_{\mathrm{esc}}(t_{K1000}-t_{\mathrm{osd0}})`.
- BP+OSD-0 is executed once per shot. The escalated branch **reuses** that decode: extra cost is GE + scoring already stored as `t_k1000_ms`.
- Gaussian elimination is computed once per non-converged shot and **reused** for $K=500/1000/$full in the same `evaluate_shot` call.
- Routing overhead (argsort of $n=1000$ $d_H$ values) is not timed separately; it is negligible vs milliseconds.
- **Always $K{=}1000$ 139.7 ms includes** BP@20+OSD-0 + GE + a **prorated** slice of a *full* free-column score sweep: `t_k1000 = t_GE + (min(1000,n_free)/n_free)*t_full_score` with $n_{\mathrm{free}}=10446$ (`adaptive_v1.py` 150–161). It is not a wall-clock of a loop that stops at $K=1000$. Same construction for $K=500$ (~132.6 ms).
- **Adaptive 38.0 ms includes** the same OSD-0 time on every shot plus that prorated $K{=}1000$ extra **only on the 200/1000 highest-$d_H$ timing shots**.

Unconditional mix at 20%: $10.51 + 0.2\times(139.69-10.51) = 36.34$ ms.
Reconstructed adaptive 20%: **38.02 ms**.
$E[\Delta T\mid \mathrm{escalated}] = (38.02-10.51)/0.2 = 137.58$ ms, vs unconditional $E[\Delta T]=129.18$ ms.
High-$d_H$ selected shots are systematically more expensive than the average shot (~8 ms extra on the timing identity). The 20k science parts show the same direction (esc20 extra 241.7 ms vs all 227.1 ms; different node, prorated times).

$273.7/38.0 \approx 7.20$ (vs always full 1FV).
$139.7/38.0 \approx 3.67$ (vs always $K=1000$; isolates selective routing).

Do **not** replace 38.0 with 36.3. 38.0 is the correct reconstructed strategy mean on the timed sample.

implication for manuscript: keep 38.0 ms; optionally footnote that $K{=}500/1000$ times prorate a full free-column sweep, and that 38.0 exceeds the unconditional mix because the routed shots are slower.

## 5. BP residual-weight experiment

**STATUS: PASS** (replay aligned with stored labels: fail/d_H match all 1.000).

production script/path: `failure_prediction/audit_residual_control.py` (this run).
production output/path: `outputs/paper_figures/final_adaptive/audit/bp_residual_auroc.csv`, `outputs/paper_figures/final_adaptive/audit/bp_residual_routing.csv`, `outputs/paper_figures/final_adaptive/audit/bp_residual_control.csv`.

Replay is BP@20+OSD-0 on the **same Stim seeds**; 1FV fail labels are the stored matched-shot columns.

### A–B. AUROC

| $p$ | $d_H$ fail | $r_{\mathrm{BP}}$ fail | syn-wt fail | $d_H$ beneficial | $r_{\mathrm{BP}}$ beneficial | Spearman $d_H$ vs $r_{\mathrm{BP}}$ |
|---|---|---|---|---|---|---|
| 0.006 | 0.958 [0.942,0.973] | 0.842 [0.807,0.872] | 0.654 [0.590,0.710] | 0.965 | 0.839 | 0.956 |
| 0.007 | 0.958 [0.951,0.965] | 0.795 [0.775,0.814] | 0.657 [0.630,0.685] | 0.957 | 0.783 | 0.943 |
| 0.008 | 0.937 [0.931,0.943] | 0.769 [0.757,0.782] | 0.659 [0.644,0.673] | 0.931 | 0.749 | 0.927 |

### C–D. Exact top-$k$, escalate to stored $K{=}1000$

| $p$ | $f$ | $d_H$ LER | $r_{\mathrm{BP}}$ LER | syn-wt LER | random LER | $d_H$ rec | $r_{\mathrm{BP}}$ rec |
|---|---|---|---|---|---|---|---|
| 0.006 | 10% | 0.00160 | 0.00325 | 0.00390 | 0.00438 | 85.1% | 40.5% |
| 0.006 | 20% | 0.00135 | 0.00220 | 0.00365 | 0.00405 | 91.9% | 68.9% |
| 0.006 | 30% | 0.00120 | 0.00175 | 0.00290 | 0.00365 | 95.9% | 81.1% |
| 0.007 | 10% | 0.00710 | 0.01395 | 0.01595 | 0.01841 | 81.0% | 37.7% |
| 0.007 | 20% | 0.00545 | 0.01140 | 0.01385 | 0.01686 | 91.5% | 53.8% |
| 0.007 | 30% | 0.00500 | 0.00950 | 0.01230 | 0.01531 | 94.3% | 65.8% |
| 0.008 | 10% | 0.03295 | 0.04990 | 0.05400 | 0.05849 | 67.6% | 29.1% |
| 0.008 | 20% | 0.02530 | 0.04170 | 0.04840 | 0.05426 | 84.9% | 47.7% |
| 0.008 | 30% | 0.02215 | 0.03545 | 0.04295 | 0.05008 | 92.1% | 61.9% |

### E. Correlation

| $p$ | $d_H$ vs $r_{\mathrm{BP}}$ | $d_H$ vs syn-wt | $r_{\mathrm{BP}}$ vs syn-wt |
|---|---|---|---|
| 0.006 | 0.956 | 0.319 | 0.329 |
| 0.007 | 0.943 | 0.366 | 0.381 |
| 0.008 | 0.927 | 0.409 | 0.431 |

### F. Non-converged subset fail AUROC

| $p$ | $d_H$ | $r_{\mathrm{BP}}$ | $r_{\mathrm{BP}}$ mean (conv / not) |
|---|---|---|---|
| 0.006 | 0.954 [0.936,0.971] | 0.825 [0.790,0.859] | 0.000 / 16.423 |
| 0.007 | 0.957 [0.949,0.964] | 0.789 [0.767,0.808] | 0.000 / 23.813 |
| 0.008 | 0.936 [0.930,0.942] | 0.768 [0.755,0.780] | 0.000 / 34.472 |

**Primary scientific question:** does $d_H$ contain useful information beyond $r_{\mathrm{BP}}=\|He_{\mathrm{BP}}\oplus s\|_0$?

$d_H$ has strictly lower exact-top-$k$ LER than $r_{\mathrm{BP}}$ at every listed $(p,f)$. Residual weight is much stronger than syndrome weight, but it does not replace $d_H$.

implication for manuscript: include $r_{\mathrm{BP}}$ as a control if space allows; do not silently treat $d_H$ as the only decoder-internal statistic.

## 6. Conventional conditional BP→OSD0 fast path

**STATUS: PASS** (no extra Monte Carlo; library field *is* the conventional path).

production script/path: `ldpc` `osd0_decoding` (BP if converge, else OSD-0) vs `osd_internals.osd0_information_set` forced on every shot including converged.

| $p$ | LER library OSD-0 field (= conventional) | LER forced algebraic OSD-0 | LER BP hard | vector diffs | logical-fail diffs | lib ok / forced fail | lib fail / forced ok |
|---|---|---|---|---|---|---|---|
| 0.006 | 0.00475 | 0.00475 | 0.79140 | 0 | 0 | 0 | 0 |
| 0.007 | 0.01990 | 0.01990 | 0.90145 | 0 | 0 | 0 | 0 |
| 0.008 | 0.06275 | 0.06275 | 0.96045 | 0 | 0 | 0 | 0 |

The reported manuscript OSD-0 LER **is** the conventional conditional output. Forcing algebraic OSD-0 on BP-converged shots is the extra construction needed if one wanted OSD-0 even when BP already returned a vector.
Timing of that extra GE on the converged minority was **not** re-benchmarked (would be a new serial study). Existing n=1000 serial already times BP+OSD-0 on every shot at 10.5 ms mean; that *is* the conventional decode() cost, because `ldpc` skips OSD on converge. Building $d_H$ on converged shots is a Hamming distance of two identical vectors (zero extra decode).
On non-converged shots, $d_H$ is a popcount of two vectors the fast path already materialises: $e_{\mathrm{BP}}$ and $e_{\mathrm{OSD0}}$. So the extra cost of $d_H$ beyond conventional BP→OSD0 is essentially zero at the decode level; the cost of *using* $d_H$ is the gated $K{=}1000$ branch.

## 7. H2 transition counts

**STATUS: PASS**

production output/path: `outputs/failure_prediction/h2_full_css_hardware/decoded_shots.npz` ; CSV `outputs/paper_figures/final_adaptive/audit/h2_osd0_full1fv_transitions.csv`.

n=300. OSD-0 LER=0.27667 (83/300). full 1FV LER=0.27667 (83/300). net failure change = 0.

|  | full 1FV correct | full 1FV fail |
|---|---|---|
| OSD-0 correct | CC=214 | CW=3 |
| OSD-0 fail | WC=3 | WW=80 |

WC (rescues) = **3**. CW (harms) = **3**. They cancel. Do not claim monotonic improvement on hardware.

## 8. Held-out table discrepancy

**STATUS: PASS** (computational reason identified; label should be renamed)

production script/path: `failure_prediction/adaptive_v1_controls.py` `holdout_block`.
production output/path: `outputs/paper_figures/final_adaptive/controls_bp_tau.md`.

Main-figure LER is exact top-$k$ on the **pooled** $n=20000$ shots.
The held-out table “oracle / in-sample” column is exact top-$k$ on each **10k-shot test fold**, then the two fold LERs are **averaged** (`mean_over_folds`). Ranking a subset is not the same as ranking the union, so the numbers need not match.

Check: $p=0.007$, $f=0.20$: pooled $0.00545$ = $109/20000$. Fold oracles $0.00540$ and $0.00540$ average to $0.00540$ = $108/20000$. One extra residual failure appears only in the pooled ranking. $p=0.008$, $f=0.20$: pooled $0.02530$ vs fold-mean $0.02505$.
Tie handling ($d_H$ ties broken by mergesort original order) also differs once the comparison set changes.
Seeds are complementary halves of the same 40 files, not a different shot set.

implication for manuscript: do **not** label the fold oracle “in-sample LER” as if it were Fig. 3’s 20k number. Call it “in-fold exact top-$k$ on the held-out seeds (mean of two 10k folds)”.

## 9. K=500 sanity check

**STATUS: PASS**

Same `parts/*.npz` `fail_k500` column; same `topk_mask` as $K{=}1000$.

| $p$ | always $K{=}500$ | 10% | 20% | 30% |
|---|---|---|---|---|
| 0.006 | 0.00145 | 0.00185 | 0.00160 | 0.00145 |
| 0.007 | 0.00685 | 0.00915 | 0.00755 | 0.00710 |
| 0.008 | 0.02645 | 0.03800 | 0.03095 | 0.02810 |

Matches the expected exact-top-$k$ values in the task statement.

## 10. Recommended manuscript corrections

### MUST FIX IN MANUSCRIPT

- If the text calls $S(x)$ exact Bernoulli NLL, change to channel score $\sum \log(1/p_i)$ (prior-weighted / OSD-CS surrogate).
- If OSD-0 is described as running on every shot: BP-converged shots skip OSD and copy $e_{\mathrm{BP}}$.
- Held-out table: rename “in-sample LER” so it is not identified with the 20k main-figure top-$k$.
- H2: report CW=3 as well as WC=3; net LER change is zero. Do not imply monotone 1FV improvement.
- Timing: if claiming truncated-search wall-clock, disclose prorated full-sweep scoring for $K{=}500/1000$. Keep 38.0 ms (do not “correct” to 36.3).

### NEW RESULT WORTH INCLUDING

- BP residual weight $r_{\mathrm{BP}}$ as a decoder-internal control vs $d_H$ and syndrome weight (Section 5 tables).
- Random-routing analytic identity $(1-f)\mathrm{LER}_{\mathrm{OSD0}}+f\mathrm{LER}_{K1000}$.
- $139.7/38.0\approx 3.68\times$ vs always $K{=}1000$ as the routing benefit isolated from full-1FV.

### NO ACTION REQUIRED

- Primary 20k LER table (OSD-0 / always $K$ / adaptive $K{=}1000$ exact top-$k$) matches the frozen parts.
- $K{=}500$ exact-top-$k$ sensitivity matches the expected values.
- Random routing already uses $K{=}1000$, not full 1FV.
- Fig. 3 full-1FV dashed line as a visual floor is fine if the caption says so.

### ANY RESULT THAT CHANGES THE CURRENT SCIENTIFIC STORY

The $d_H$ routing story is **not overturned**. $r_{\mathrm{BP}}$ is a strong but strictly weaker matched-budget router than $d_H$ in this table. It is much stronger than syndrome weight. Include it as a control, not a replacement.
Forced algebraic OSD-0 on BP-converged shots does not change bb144 LER at these $p$ relative to the library field. The conventional vs forced distinction is conceptually real but empirically idle on this dataset.

Main-LER sanity: PASS

