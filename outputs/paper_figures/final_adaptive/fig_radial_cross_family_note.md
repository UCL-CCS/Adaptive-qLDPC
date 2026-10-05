# Radial cross-family main-text figure

Source: frozen formal experiment `outputs/failure_prediction/nonbb_radial_formal/`.
No shots were regenerated. No decoder, budget, or routing policy was changed.
Manuscript not edited.

## Files

- PNG: `outputs/paper_figures/final_adaptive/fig_radial_cross_family.png`
- PDF: `outputs/paper_figures/final_adaptive/fig_radial_cross_family.pdf`
- tidy plotting CSV: `outputs/paper_figures/final_adaptive/fig_radial_cross_family.csv`

## What the figure shows

Primary comparison: **exact matched-budget routing** (stable `argsort` of $-$score; escalate exactly `round(f n)` shots to $K=1000$). The primary endpoint is **$f_{\mathrm{esc}}=20\%$** (vertical guide). Code: radial / lifted-product `radial_90_8`, $n=90$, $k=8$, **distance unknown**. $K=1000$ is a real truncation ($k_{\mathrm{free}}=3289$). Full 1FV differs from always $K=1000$ by **one shot** at each $p$ (418 vs 419 failures at $p=0.008$; 931 vs 932 at $p=0.009$); the two reference lines therefore overlap on the plot and must not be read as a visual gap.

Preregistered scores, larger = higher predicted OSD-0 risk: $d_H$ AUROC $0.951$ / $0.940$; residual AUROC $0.832$ / $0.809$. Exact top-20% LER: $p=0.008$: $d_H$ $0.02275$, residual $0.03290$, syndrome $0.03860$, random $0.04525$; $p=0.009$: $d_H$ $0.05390$, residual $0.07275$, syndrome $0.07975$, random $0.09054$. Recovered headroom $R(d_H)$ at 20% $=0.939$ and $0.866$. Random bands are 95% intervals over 200 seeds; other CIs are in the CSV.

## Caption draft

Cross-family matched-budget routing on the radial / lifted-product code `radial_90_8` ($n=90$, $k=8$; distance unknown). Each panel is an independent 20,000-shot test set at circuit-level Z-memory noise $p=0.008$ (a) and $p=0.009$ (b). Shots are ranked by a score computed from the OSD-0 fast path only, then the top $f_{\mathrm{esc}}$ are escalated to truncated one-free-variable search with $K=1000$; $f=0$ is OSD-0 and $f=1$ is always $K=1000$. $K=1000$ is a genuine truncation ($k_{\mathrm{free}}=3289$). Decoder disagreement $d_H$ outperforms BP residual $r_{\mathrm{BP}}$, syndrome weight $w_s$, and random routing at the primary 20% budget, recovering $93.9\%$ and $86.6\%$ of the OSD-0 to full-1FV gap. Full 1FV (dash-dotted) differs from always $K=1000$ (dashed) by one shot at each $p$ and is visually coincident. Random bands are 95% intervals over 200 seeds. AUROC for OSD-0 failure: $d_H$ $0.951$ / $0.940$, residual $0.832$ / $0.809$.

## Is this enough for a main-text figure?

Yes. The two-panel LER-versus-budget plot is the same claim geometry as the BB Adaptive V1 routing figure, on a structurally distinct code, with the extra residual control the cross-family question requires. It is compact enough for a two-column layout. Suggested insertion: immediately after the BB matched-budget / recovered-headroom discussion, as a short cross-family validation paragraph plus this figure — not as a replacement for the BB panels. Keep the Hamming-HGP negative control out of the main text. Do not add this figure to the manuscript in this step.
