# Adaptive Decoder V1 — key results

Code: `bb_144_12_12`. Noise: circuit-level depolarizing, \(p\in\{0.006,0.007,0.008\}\), \(n=20000\) matched shots per \(p\). Algorithm frozen.

## A. Method

```
BP@20 + OSD-0
    →  d_H(e_BP, e_OSD0)     # already produced by the fast path
    →  if d_H <  τ: return OSD-0
       if d_H ≥  τ: truncated one-free-variable search, K=1000
```

\(K=500\) is a sensitivity point. Full one-free-variable search is the accuracy reference. No ML router, no new OSD objective.

## B. Main result

A cheap BP–OSD-0 disagreement statistic concentrates extra OSD search on a minority of shots and recovers most of the OSD-0 → full-1FV accuracy gain.

At ~20% escalation (\(K=1000\)):

| \(p\) | Adaptive LER | gap recovered |
|---|---|---|
| 0.006 | 0.00135 | 91.9% |
| 0.007 | 0.00525 | 92.7% |
| 0.008 | 0.02475 | 86.2% |

At ~30% escalation (\(K=1000\)):

| \(p\) | Adaptive LER | gap recovered |
|---|---|---|
| 0.006 | 0.00120 | 95.9% |
| 0.007 | 0.00500 | 94.3% |
| 0.008 | 0.02205 | 92.3% |

OSD-0 LER: 0.00475 / 0.01990 / 0.06275. Full 1FV LER: 0.00105 / 0.00410 / 0.01865.

## C. Runtime (measured, same node)

Serial wall-clock on wolpy01 (\(p=0.007\), \(n=200\)):

| decoder | mean ms / shot |
|---|---|
| BP@20 + OSD-0 | 22.4 |
| always \(K=500\) | 248.7 |
| always \(K=1000\) | 258.9 |
| full 1FV | 450.5 |
| package OSD-CS order 2 | 1379.6 |

Adaptive cost is \(t_{\mathrm{OSD0}} + f_{\mathrm{esc}}(t_{K=1000}-t_{\mathrm{OSD0}})\), using those serial times — not a candidate-count estimate.

At ~20% escalation this is ~72–74 ms / shot, about **6×** cheaper than always-on full 1FV (450 ms), while recovering 86–93% of the accuracy gap. \(K=1000\) costs only ~10 ms more than \(K=500\) when always on (~4%), because Gaussian elimination dominates the strong branch.

Package OSD-CS is a practical reference only; it is not the accuracy baseline (implementation overhead, not search physics).

## D. Router evidence

\(d_H(\hat e_{\mathrm{BP}},\hat e_{\mathrm{OSD0}})\) vs OSD-0 failure / beneficial escalation:

| \(p\) | fail AUROC | beneficial AUROC |
|---|---|---|
| 0.006 | 0.958 | 0.965 |
| 0.007 | 0.958 | 0.957 |
| 0.008 | 0.937 | 0.931 |

This is an empirical disagreement statistic on this code and these \(p\), not a claimed universal hardness metric.

## E. Why \(K=1000\) is primary

At matched ~20% escalation, \(K=1000\) vs \(K=500\):

- \(p=0.006\): 0.00135 vs 0.00160
- \(p=0.007\): 0.00525 vs 0.00735
- \(p=0.008\): 0.02475 vs 0.03050

The extra serial cost is small; the LER gain is not, especially at \(p=0.007\) and \(0.008\).

## F. Fixed-threshold note (not a new algorithm)

\(\tau^*=21\), calibrated for ~20% escalation at \(p=0.007\), transfers as:

- \(p=0.006\): \(f_{\mathrm{esc}}=5.7\%\)
- \(p=0.007\): \(f_{\mathrm{esc}}=21.7\%\)
- \(p=0.008\): \(f_{\mathrm{esc}}=49.2\%\)

A fixed risk threshold does **not** hold compute budget constant. It spends more strong-decoder work as the noise regime hardens (risk-controlled mode). Budget-controlled mode instead chooses \(\tau\) per \(p\) to hit a target \(f_{\mathrm{esc}}\). Figures 2–3 use the latter.

## G. Scope / caveats

- Demonstrated primarily on `bb_144_12_12` at three \(p\).
- Figure 5 is a headroom control from existing bb72 survey data, **not** an Adaptive V1 run on bb72.
- Serial timing is one node, one \(p\), \(n=200\); Figure 3 mixes that cost model with 20k-shot LER.
- \(d_H\) is not claimed universal; mechanism studies (boundary negative result, WC/WW, coset aggregation) are supporting analysis and are closed.
- Real-time QPU–HPC integration has not been performed.

## Candidate claims (for later drafting; do not overstate)

1. On `bb_144_12_12`, BP versus OSD-0 Hamming disagreement is a cheap online statistic that ranks shots by OSD-0 failure risk (AUROC 0.93–0.96 at \(p=0.006\)–\(0.008\)).
2. Routing a truncated one-free-variable OSD search with this statistic recovers most of the OSD-0 → full-search accuracy gain while escalating only a minority of shots.
3. Around 20% escalation with \(K=1000\) recovers 92%, 93%, and 86% of that gain at the three tested \(p\); around 30% recovers 96%, 94%, and 92%.
4. Measured serial cost of this adaptive rule is set by OSD-0 plus a gated copy of the truncated-search work, not by the size of the full candidate list.
5. \(K=1000\) is a better operating point than \(K=500\) on this code: almost the same measured strong-branch time, substantially lower LER at \(p=0.007\) and \(0.008\).
6. Adaptive decoding is only as useful as the fast–strong accuracy gap: existing bb72 data show ~1.2× headroom, versus ~3.4–4.9× on bb144.
7. A single numerical threshold on \(d_H\) implements a risk-controlled policy whose escalation fraction grows with \(p\); a target compute budget requires retuning \(\tau\).

## Figures

| file | point |
|---|---|
| `fig1_adaptive_v1_schematic.png` | Pipeline: always BP+OSD-0+\(d_H\); only high-\(d_H\) shots enter \(K=1000\). |
| `fig2_ler_vs_escalation.png` | Main result: LER vs \(f_{\mathrm{esc}}\) at three \(p\). |
| `fig3_ler_vs_runtime.png` | Same trade-off on measured serial milliseconds. |
| `fig4_dh_router.png` | Why the router works: \(d_H\) separates OSD-0 fail from correct. |
| `fig5_bb72_headroom_control.png` | Control: bb72 has little fast–strong headroom; bb144 has several×. |
