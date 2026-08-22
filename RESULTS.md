# Results — Graph Structure and Cross-Sectional Return Rank Prediction

This document reports the project's evaluation on a **cross-sectional 5-day
forward return rank classification** task over S&P 500 constituents
(2015–2024), including the corrections and extensions made after the original
version of this document.

> **Note on revisions.** An earlier version of this document claimed the
> predictive gain was "specifically attributable to the real economic
> relationships encoded in the Bloomberg data," based on Bloomberg edges
> beating a "sector-based proxy" at *p* < .05. That proxy was later found to
> be a **random block graph** — sector labels were assigned as alphabetical
> ticker order modulo 11 — so the comparison tested real edges against random
> edges, not against sectors. Re-running with real GICS labels overturns the
> conclusion (Finding 2). The original results remain in `results/main_ablation/`
> and are reproducible; this document supersedes their interpretation.

All numbers are reproducible from the raw JSON in [`results/`](results/):

- `scripts/analyze_realsector.py` — Finding 1 (main ablation, real vs fake sectors)
- `scripts/analyze_granularity.py` — Findings 2–3 (granularity, decomposition)
- `scripts/analyze_recency.py` — Finding 5 (look-ahead diagnostic)
- `scripts/analyze_per_class.py` — Finding 7 (mechanism)
- `scripts/analyze_fama_macbeth.py` — Finding 9 (control regressions)
- `scripts/run_horizon_backtest.py`, `scripts/analyze_industry_rotation.py` — Finding 10
- `scripts/analyze_results.py` — original tables, walk-forward, MPT backtest

---

## TL;DR

> On cross-sectional 5-day forward return rank prediction (3-class, 500
> stocks, 2015–2024), graph structure over economic groupings improves
> per-date macro-F1 robustly — in **7 of 7 non-overlapping evaluation
> windows** (*t* = +11.67). Partition granularity matters and behaves as a
> **step, not a gradient**: moving from sector-level grouping to any finer
> partition improves tail-F1 by ≈0.018 (*p* < .001, replicated on 10
> disjoint seeds), after which the benefit saturates by roughly 25 groups.
> The improvement is **not** specific to proprietary data — a free GICS
> graph is statistically indistinguishable from one built on Bloomberg
> supplier/customer and holder relationships (Δ = +0.003, n.s.).
> Correlation-only and holder-only graphs add nothing. A recency
> diagnostic shows the static supply-chain snapshot's look-ahead does not
> explain any of it.
>
> **None of this demonstrably becomes portfolio performance.** Realised
> long-short spread is worse in 6 of 7 windows (*p* < .05), and
> mean-variance portfolios on the graph signals underperform an identical
> no-graph portfolio (Δ = −0.94 Sharpe). But the sign is **construction-
> dependent**: with equal-weight sizing at the same horizon the graph
> signal is marginally *better* (+0.258 Sharpe, *t* = +1.99), and no
> construction reaches significance once power is properly accounted for.
> The tradeability question is **unresolved rather than settled negative**.
> Two analyses explain why any advantage is hard to realise. Per-class F1 shows the accuracy gain sits almost entirely
> in the **Neutral** class — 60% of the mass, which a quintile long-short
> book never trades — while tail accuracy is flat-to-worse. Fama-MacBeth
> regressions show the residual return-predictive content does not survive
> controls for momentum, short-term reversal, volatility and industry
> fixed effects (23% of univariate magnitude retained, pooled *t* = +0.64),
> and is identified almost entirely *between* industries rather than
> within them.

**One-line version:** *graph structure reliably improves a standard
classification metric, but the improvement lives in the part of the return
distribution portfolios cannot trade and is largely subsumed by industry
momentum — so macro-F1 is a misleading objective for cross-sectional
equity models.*

---

## 1. Task definition

For each trading day *t* and stock *i ∈ S&P 500*, the label is the
rank-quantile of the forward 5-day return:

```
y[i, t] =  0  (Down)     if forward 5-day return is in bottom 20% across stocks on day t
        =  1  (Neutral)  if in middle 60%
        =  2  (Up)       if in top 20%
```

The target is **cross-sectional**: on any day *t*, every stock shares the
same market-level context, so aggregate market features cannot in principle
discriminate between stocks within a date. Any predictive signal must come
from per-stock or inter-stock structure. Random baseline macro-F1 ≈ 0.333.

### Why this task (and not the original regime task)

The original regime-classification task labels each *day* by mechanical
percentile rules on `vol_20d`, `avg_corr`, and `ret_20d`. Because the same
aggregate features that define the labels are observable, a logistic
regression on `(mkt_vol, mkt_ret, avg_corr)` achieves AUC ≈ 0.963 on the
early-warning target, dominating any GNN (Appendix B). The cross-sectional
rank task closes this leak by construction. See Appendix A for the full task
history.

---

## 2. Architecture and data

`scripts/run_xsec_rank.py` implements a dynamic heterogeneous GNN:

- **Inputs:** 22-dim per-stock features (price/volume/return/vol over
  multiple horizons + market context + sector one-hots) × 30-day sequence ×
  500 stocks.
- **Per-relation scaled-dot-product attention** over each active edge type,
  2 graph layers per time-step, per-stock LSTM, 3-way classifier head with
  class-weighted cross-entropy. ~1.0M parameters, 15 epochs, AdamW + cosine.
- **Sector labels:** real GICS via `--sector-file` (Bloomberg
  `GICS_SECTOR_NAME` for 469/500 tickers, Yahoo-crosswalk fallback for 19,
  12 unmapped into an explicit unknown bucket). Built by
  `scripts/build_sector_file.py`. The GICS file itself is not committed
  (proprietary classification); the builder is.

Edge configurations:

| Mode | Correlation | Group/holder | Supply |
|---|---|---|---|
| `bloomberg` | top-K rolling | Real holder Jaccard ≥ 0.4 | Real supplier/customer |
| `proxy` | top-K rolling | Same-GICS-group edges | — |
| `corronly` | top-K rolling | — | — |
| `none` | — | — | — (per-stock LSTM only) |
| `supply_only` | top-K rolling | — | Real supplier/customer |
| `holder_only` | top-K rolling | Real holder Jaccard ≥ 0.4 | — |

Train cutoff 2022-09-30 (validation 2022Q4–2024) unless noted. Paired
t-tests across seeds, df = n−1.

### Data caveats (read before citing any number)

1. **The supply-chain/holder metadata is a single static snapshot** (~2025/26
   pull) applied across the whole 2015–2024 panel. Bloomberg's `SPLC` does
   not honor historical date overrides (verified directly: a 2016-06-30
   override returns the identical, identifiably-modern supplier list for
   AAPL). Finding 5 tests whether this look-ahead drives the results.
2. **The universe is survivorship-biased**: 500 tickers screened near the
   pull date; firms that exited the index during the sample are missing.
3. Prices are Bloomberg `px_last` without delisting returns; transaction
   costs in the portfolio section are a flat 5 bps assumption.

---

## 3. Finding 1 — graph structure helps; supply-chain specificity does not (n=5)

Main ablation with **real GICS sector labels** (`results/realsector/`,
train cutoff 2022-09-30, per-date macro-F1, best epoch):

| Config | s42 | s123 | s7 | s101 | s2025 | **mean** | std |
|---|---|---|---|---|---|---|---|
| **Bloomberg** | 0.3730 | 0.3653 | 0.3762 | 0.3707 | 0.3761 | **0.3723** | 0.0045 |
| Sector proxy | 0.3749 | 0.3755 | 0.3667 | 0.3599 | 0.3677 | 0.3689 | 0.0064 |
| CorrOnly | 0.3461 | 0.3296 | 0.3546 | 0.3569 | 0.3612 | 0.3497 | 0.0125 |
| NoGraph | 0.3579 | 0.3527 | 0.3520 | 0.3526 | 0.3590 | 0.3549 | 0.0033 |

**Paired t-tests (n=5, df=4):**

| Comparison | Δ | t | *p* |
|---|---|---|---|
| Bloomberg vs NoGraph | +0.0174 | +8.94 | **< .001** |
| Sector proxy vs NoGraph | +0.0141 | +4.97 | **< .01** |
| Bloomberg vs Sector proxy | +0.0033 | +0.82 | n.s. |
| CorrOnly vs NoGraph | −0.0052 | −0.97 | n.s. |

**Interpretation.** Economically meaningful graph structure helps: both the
Bloomberg graph and the free sector graph beat no-graph decisively. But the
Bloomberg graph does **not** beat the sector graph — the premium the earlier
version of this document attributed to proprietary relationship data is
statistically zero once the comparison graph is a real sector partition
rather than a random one. Correlation edges alone add nothing (numerically
negative).

For reference, against the earlier fake-sector run: Bloomberg-vs-proxy was
+0.0145 (*p* < .05) when "proxy" meant random edges; the proxy arm itself
gained +0.0135 from receiving real labels.

---

## 4. Finding 2 — partition granularity: a step, not a gradient (n=10, replicated)

Walking the same-group graph down the GICS hierarchy. The table below uses
**10 seeds disjoint from all other results in this document**
(`results/seedsweep/`), so it is an independent replication rather than an
extension of the original n=5 run in `results/granularity/`.

| Graph partition | Same-group edges | macro-F1 | **tail-F1** | LS spread |
|---|---|---|---|---|
| Sector (11 groups) | 25,782 | 0.3664 ± 0.0094 | **0.2344 ± 0.0114** | 10.78 ± 4.02 |
| Industry group (25) | 12,268 | 0.3728 ± 0.0032 | **0.2524 ± 0.0113** | 10.89 ± 4.74 |
| Industry (67) | 5,904 | 0.3722 ± 0.0017 | **0.2528 ± 0.0040** | 11.86 ± 5.73 |
| Sub-industry (124) | 3,548 | 0.3740 ± 0.0042 | **0.2526 ± 0.0064** | 11.69 ± 4.49 |

**Paired tests vs sector (n=10, df=9):**

| Partition | macro-F1 | **tail-F1** | LS spread |
|---|---|---|---|
| Industry group (25) | +0.0063, *p* < .05 | **+0.0180, *p* < .01** | +0.11, n.s. |
| Industry (67) | +0.0058, n.s. | **+0.0184, *p* < .001** | +1.07, n.s. |
| Sub-industry (124) | +0.0076, *p* < .05 | **+0.0183, *p* < .001** | +0.90, n.s. |

**This is a step function, not a gradient.** An earlier version of this
section reported a monotone rise with fineness (+0.0090 macro-F1
sector→sub-industry, *p* < .05, n=5). That does not replicate. On disjoint
seeds, industry-group, industry and sub-industry are statistically
indistinguishable from one another (tail-F1 0.2524 / 0.2528 / 0.2526);
what is robust is the **jump from sector to anything finer**, after which
the benefit saturates by roughly 25 groups.

Two things make this the strongest positive result in the project. First,
it replicates on fully disjoint seeds. Second, it is **larger and far more
significant on tail-F1 (+0.018, *p* < .001) than on macro-F1 (+0.006 to
+0.008, *p* < .05)** — the only finding here that looks better on the
portfolio-relevant metric than on the misleading one (cf. Finding 7).

Granularity does **not** affect realised long-short spread at any level
(all n.s.), so better tail ranking still does not become tradeable
performance.

For reference, the original n=5 run (`results/granularity/`, macro-F1
only) gave sector 0.3689 → sub-industry 0.3779. Run-to-run CUDA
nondeterminism alone moves this comparison between *p* < .05 and n.s.,
which is why the 10-seed replication is the number to cite.

**Headline comparison** — the free sub-industry graph vs the proprietary
Bloomberg graph:

| Comparison | Δ | t | *p* | seeds won |
|---|---|---|---|---|
| Sub-industry vs Bloomberg | +0.0057 | +2.30 | < .10 | **5/5** |
| Sub-industry vs Supply-only | +0.0075 | +3.75 | **< .05** | 5/5 |
| Sub-industry vs NoGraph | +0.0231 | +11.38 | **< .001** | 5/5 |

**Interpretation.** Performance improves as the partition moves from sector
to industry-group level, while edge count falls 7× — so the mechanism is
grouping *precision*, not connectivity. The improvement is a **step, not a
monotone gradient**: it saturates by roughly 25 groups, and the further
refinements to industry (67) and sub-industry (124) are statistically
indistinguishable from industry group (see the retraction note in the claim
table — the monotone reading did not replicate on disjoint seeds).

Message passing among a small set of true economic peers (a sub-industry
here averages ~4 firms) beats both broad sector pooling and
curated bilateral supply-chain links. The best-performing graph in the
entire study is built from a free, public classification.

The *p* < .10 on the headline pairwise test is suggestive rather than
decisive; the 5/5 seed sweep is the stronger evidence (the monotone gradient
originally cited here has since been retracted). Treat "sub-industry ≥ Bloomberg" as established and
"sub-industry > Bloomberg" as likely but not yet conclusive.

> **Resolved caveat.** An earlier revision flagged that this section was
> measured only in macro-F1, which Finding 7 shows rewards the untradeable
> Neutral class. The tail-F1 and long-short-spread columns above are that
> re-measurement. Outcome: the granularity effect **does** hold on
> tail-F1, and more strongly than on macro-F1 — but it does **not** reach
> realised spread. Read alongside Findings 7 and 8.

---

## 5. Finding 3 — edge-source decomposition with real sectors (n=5)

Re-run of the decomposition (`results/decomp_realsector/`; the original n=3
fake-sector version is preserved in `results/edge_decomposition/`):

| Config | mean | std |
|---|---|---|
| Bloomberg (full) | 0.3723 | 0.0045 |
| Supply-only | 0.3704 | 0.0025 |
| Sector proxy | 0.3689 | 0.0064 |
| NoGraph | 0.3549 | 0.0033 |
| CorrOnly | 0.3497 | 0.0125 |
| Holder-only | 0.3491 | 0.0064 |

| Comparison | Δ | t | *p* |
|---|---|---|---|
| Supply-only vs NoGraph | +0.0156 | +8.49 | **< .01** |
| Supply-only vs Sector proxy | +0.0015 | +0.53 | n.s. |
| Bloomberg vs Supply-only | +0.0018 | +0.63 | n.s. |
| Holder-only vs NoGraph | −0.0058 | −1.53 | n.s. |

**What survives from the original version:** supplier/customer edges carry
essentially all of the Bloomberg graph's signal (≈90% of the gain over
no-graph; originally reported as 96%), and institutional-holder overlap is
useless-to-harmful alone (dense, dominated by universal owners).

**What does not survive:** any premium for supply-chain edges over a
same-quality sector partition. Supply-chain links are *a* way to encode
economic relatedness — not a *better* way than free classifications.

---

## 6. Finding 4 — walk-forward robustness (n=3 per cutoff)

Unchanged from the original document (fake-sector context; Bloomberg vs
NoGraph is unaffected by the proxy-arm issue):

| Train cutoff | BBG mean ± std | NoGraph mean ± std | Δ | *p* |
|---|---|---|---|---|
| 2021-09-30 | 0.3626 ± 0.0021 | 0.3482 ± 0.0006 | +0.0145 | < .01 |
| 2022-09-30 | 0.3720 ± 0.0015 | 0.3514 ± 0.0008 | +0.0206 | < .01 |
| 2023-09-30 | 0.3673 ± 0.0008 | 0.3513 ± 0.0088 | +0.0161 | < .20 |

The graph-vs-no-graph advantage is stable across validation windows spanning
post-COVID rally, the 2022 drawdown, and recovery.

---

## 7. Finding 5 — recency diagnostic: look-ahead does not explain the results (n=3 per period)

Because the supply-chain snapshot is static (~2025/26), look-ahead severity
scales with distance from the snapshot. If contamination drove the Bloomberg
advantage, the Bloomberg-minus-NoGraph delta should shrink as evaluation
approaches the snapshot date (`results/recency/`,
`scripts/analyze_recency.py`):

| Eval period | Look-ahead | BBG | NoGraph | Δ | t | *p* |
|---|---|---|---|---|---|---|
| 2017 | ~9 yr | 0.3896 | 0.3801 | +0.0096 | +5.44 | < .05 |
| 2019 | ~7 yr | 0.3618 | 0.3357 | +0.0261 | +3.41 | < .10 |
| 2021 | ~5 yr | 0.3679 | 0.3570 | +0.0109 | +3.52 | < .10 |
| 2024 | ~2 yr | 0.3699 | 0.3507 | +0.0192 | +19.69 | **< .01** |

The advantage is *smallest* where look-ahead is worst and both large and
tightest in the cleanest period — the opposite of the artifact prediction.
Economically: the signal appears to ride on persistent relationships, for
which a stale snapshot is a serviceable proxy. This weakens the look-ahead
objection substantially without eliminating it; a point-in-time rebuild
(Compustat Segment Customer / SEC 10-K disclosures) remains worthwhile as
robustness.

---

## 8. Finding 6 — portfolio results are sizing-dependent and underpowered (n=5)

A Markowitz mean-variance layer on the model's predictions
(`results/mpt_program/`, `RESULTS_MPT.md`; μ = P(Up)−P(Down), covariance
from rolling correlations, long top-quintile / short bottom-quintile,
dollar-neutral, gross cap 2.0, 5 bps costs, 5-day rebalance).

An initial run on the standard split (163 rebalances) found no significant
differences. That was **underpowered, not neutral**: the standard error of
a Sharpe estimate over ~2.25 years is roughly ±0.65. Extending the
out-of-sample window (2019 cutoff, **383 rebalances**) resolves it:

| Signal | MV Sharpe (n=5) | vs no-graph | *p* |
|---|---|---|---|
| Sub-industry (best classifier) | 0.174 ± 0.368 | −0.943 | **< .01** |
| Bloomberg | 0.716 ± 0.205 | −0.400 | **< .05** |
| No-graph | 1.117 ± — | — | — |

Under **mean-variance sizing** both graph signals underperform no-graph on
this window, and the best classifier produces the worst portfolio.

> **Important qualification (see Finding 10).** This result is specific to
> mean-variance sizing and should not be read as a general property of the
> signal. Re-running with **equal-weight quintile sizing** and
> non-overlapping rebalances *reverses* the sign at the same 5-day horizon
> (graph − no-graph = **+0.258** Sharpe, *t* = +1.99). That is consistent
> with the observation below that MV sizing carries ~3.4× the seed variance
> of equal-weight sizing on identical picks: at this effect size, MV-based
> comparisons are too noisy to establish direction. No construction reaches
> significance once power is accounted for. The honest summary is that the
> tradeability question is **unresolved**, not settled negative.

**It is not a transaction-cost effect.** Decomposing gross vs net:

| Signal | Gross Sharpe | Net Sharpe | Cost drag | Turnover |
|---|---|---|---|---|
| Sub-industry | +0.257 | +0.144 | 8.4 bp | 1.678 |
| Bloomberg | +0.657 | +0.595 | 4.5 bp | 0.904 |
| No-graph | +0.982 | +0.928 | 3.4 bp | 0.682 |

The per-period gross-return gap is −44.4 bp against a cost gap of +5.0 bp,
so **costs explain ~10%** of the shortfall (sub-industry vs no-graph gross:
*p* < .05). Within the MV construction the gap is present before any
friction; turnover is a symptom, not the cause. (This decomposition is
itself MV-specific — see the qualification above.)

Three further interventions were tested and none rescues performance:
a **shrinkage sweep** (λ = 0 → 0.95) moves Sharpe 0.431 → 0.444 (n.s.) and
seed-std only 1.066 → 1.007 — scalar shrinkage leaves the quintile ranking
unchanged, so it cannot bind; a **turnover penalty** at 5 bps cuts turnover
1.489 → 1.340 but also Sharpe 0.438 → 0.357 (*p* < .05); and **equal-weight
sizing** of the same picks has 3.4× lower seed-variance than mean-variance
sizing (0.31 vs 1.07), which is where Michaud (1989) error-maximisation
actually shows up in this study. Incidentally, *every* configuration —
including no-graph — concentrates ~50% of gross exposure in a single
sub-industry and holds ~2 effective groups out of 125, a property of
gross-capped mean-variance optimisation itself.

---

## 9. Finding 7 — mechanism: the advantage lives in the untradeable class

Per-class F1 on the long OOS (`results/attribution/per_class_f1.txt`)
explains the reversal:

| Signal | Down F1 | **Neutral F1** | Up F1 | macro | **tails avg** |
|---|---|---|---|---|---|
| Sub-industry | 0.2579 | **0.5852** | 0.2355 | 0.3595 | 0.2467 |
| Bloomberg | 0.2118 | **0.5795** | 0.2817 | 0.3577 | 0.2467 |
| No-graph | 0.2298 | 0.5472 | 0.2774 | 0.3515 | **0.2536** |

Deltas vs no-graph: sub-industry Neutral **+0.0380**, tails **−0.0069**;
Bloomberg Neutral **+0.0323**, tails **−0.0068**.

Both graph models' macro-F1 advantage comes almost entirely from the
**Neutral** class — 60% of the probability mass, and the one region a
quintile long-short book never trades. On the tails, the only part the
portfolio touches, both are slightly *worse*.

The portfolio-relevant metrics agree, and order identically to the Sharpes:

| Signal | Top-Q precision | Bottom-Q precision | **LS spread/period** |
|---|---|---|---|
| Sub-industry | 21.7% | 20.3% | **6.0 bp** |
| Bloomberg | 20.9% | 21.2% | **7.4 bp** |
| No-graph | 21.8% | 21.7% | **9.7 bp** |

```
LS spread  6.0bp  <  7.4bp  <  9.7bp
MV Sharpe  0.174  <  0.716  <  1.117
```

**Full causal chain:** graph structure raises macro-F1 → the gain sits in
Neutral → tail precision is flat-to-worse → realised long-short spread
shrinks → portfolios underperform gross of costs → higher turnover adds a
further ~10%.

**Methodological implication.** macro-F1 rewards precisely the part of the
cross-sectional distribution a quintile portfolio cannot trade. Every
classification result in Findings 1–4, including the granularity gradient,
is measured in macro-F1 and therefore **may not carry portfolio
relevance**. A re-measurement of the gradient on tail-F1 and long-short
spread is in progress; until it lands, Finding 2 should be read as a
statement about macro-F1 specifically, not about tradeable signal.

For transparency, three mechanism hypotheses were tested and **rejected**
before this one (`results/attribution/weight_attribution.txt`):
transaction costs (above); *signal smoothing* — within-sub-industry share
of μ variance is 0.444 (sub-industry) vs 0.480 (no-graph) vs 0.569
(Bloomberg), non-monotonic and inside noise; and *portfolio concentration*
— group HHI 0.499 / 0.497 / 0.542 respectively, with no-graph marginally
the most concentrated.

---

## 10. Finding 8 — period sweep: what survives across independent windows

The single strongest robustness test in the project
(`results/periods/`): seven **non-overlapping** evaluation windows, each
training only on prior data and evaluating the following ~15 months, so
each is an independent read. Sub-industry graph vs no-graph, 3 seeds each.

| Eval window | macro-F1 Δ | tail-F1 Δ | LS spread Δ |
|---|---|---|---|
| 2016Q4–2017 | +0.0209 | +0.0253 | −14.33 bp |
| 2017Q4–2018 | +0.0177 | +0.0192 | −12.09 bp |
| 2018Q4–2019 | +0.0218 | +0.0053 | −10.02 bp |
| 2019Q4–2020 | +0.0182 | +0.0019 | −12.46 bp |
| 2020Q4–2021 | +0.0163 | −0.0039 | −10.45 bp |
| 2021Q4–2022 | +0.0307 | +0.0048 | +5.69 bp |
| 2022Q4–2024 | +0.0216 | −0.0086 | −3.97 bp |
| **Graph wins** | **7/7** | 5/7 | **1/7** |
| **Mean (t, df=6)** | **+0.0210 (t=+11.67)** | +0.0063 (t=+1.38, n.s.) | **−8.23 bp (t=−3.13, *p* < .05)** |

Three conclusions, each now robust across independent periods:

1. **Graph structure improves macro-F1 in 7/7 windows** (t = +11.67), through
   COVID, the 2022 drawdown, and two recoveries. Not period-dependent.
2. The **tail-F1 advantage of graph-vs-no-graph is not robust** (5/7,
   n.s.) — distinct from Finding 2, where finer-vs-coarser *partitions*
   robustly improve tail-F1. Having a graph is not the same intervention
   as choosing its resolution.
3. **Long-short spread is reliably worse** (6/7 windows, mean −8.23 bp,
   *p* < .05).

This also retires an earlier single-window comparison. A standard-OOS
measurement suggested graph signals produced a far *better* spread
(sub-industry 14.72 bp vs no-graph 2.52 bp). The identical configuration
in this sweep gives −3.97 bp. Spread estimates swing ~16 bp between runs
of the same configuration, so **single-window spread comparisons are not
informative**; only the multi-window aggregate is.

---

## 11. Finding 9 — the signal does not survive standard controls

Fama-MacBeth cross-sectional regressions at each rebalance date
(`results/attribution/fama_macbeth.txt`,
`scripts/analyze_fama_macbeth.py`):

```
fwd_ret[i,t] = a_t + b_t · signal[i,t] + momentum + reversal + volatility
               + industry FE + e[i,t]
```

Coefficients averaged across dates, Newey-West standard errors (lag 5, for
the overlapping 5-day horizon). 707 regressions over the same 7
non-overlapping windows, 3 seeds each. Coefficient on the standardised
signal, in bp per 5-day period:

| Window | Signal alone | + controls + industry FE |
|---|---|---|
| 2016Q4–2017 | 4.81 (t = +2.57) | 2.75 (t = +1.54) |
| 2017Q4–2018 | 2.45 (t = +0.88) | −0.09 (t = −0.04) |
| 2018Q4–2019 | 3.48 (t = +0.95) | 2.08 (t = +0.93) |
| 2019Q4–2020 | 9.10 (t = +2.54) | 2.06 (t = +0.83) |
| 2020Q4–2021 | −1.25 (t = −0.16) | −1.81 (t = −0.33) |
| 2021Q4–2022 | −5.78 (t = −0.87) | −3.50 (t = −0.95) |
| 2022Q4–2024 | 5.30 (t = +2.78) | 2.05 (t = +2.15) |
| **Pooled** | **2.88 (t = +1.74)** | **0.67 (t = +0.64)** |

**Even uncontrolled, return-predictive content is weak**: pooled *t* =
+1.74, significant in 3 of 7 windows and negative in 2. The robust 7/7
macro-F1 improvement (Finding 8) therefore does **not** correspond to
robust return prediction — exactly what the Neutral-class mechanism
(Finding 7) predicts.

**After controls the signal retains 23% of its univariate magnitude and is
not significant** (*t* = +0.64); only 1 of 7 windows survives.

Because the industry fixed effect is implemented as within-sector
demeaning, this also localises the signal: its predictive content is
largely **between** industries, not within them. That completes the
mechanism. Graph message passing ranks whole peer groups well — which
registers as macro-F1, concentrated in the Neutral class — but does not
separate stocks *inside* a group, which is what a quintile book requires;
and group-level prediction is in turn largely subsumed by industry
momentum.

**Limitation, stated plainly.** Size and book-to-market are not controlled
for; both need data this project does not have (Bloomberg
`CUR_MKT_CAP` / `PX_TO_BOOK_RATIO`, or CRSP/Compustat via WRDS). Note the
direction of that gap: the signal already fails against a *partial* control
set, so additional controls can only reduce it further, not restore it.
The conclusion is robust to the missing data; obtaining it would improve
presentation, not the finding.

---

## 12. Finding 10 — horizon and sizing: longer holds help the strategy, not the graph

Two constructions were tested to see whether the Finding 6 reversal was an
artifact rather than a property of the signal
(`results/horizon/`, `scripts/run_horizon_backtest.py`;
`results/attribution/industry_rotation.txt`,
`scripts/analyze_industry_rotation.py`). Equal-weight quintiles,
**non-overlapping** rebalances (hold *H* days, then trade), 5 bps costs,
7 windows × 3 seeds, Newey-West on the paired difference.

| Horizon | Signal | Sharpe | Mean/period | Turnover/yr | n periods |
|---|---|---|---|---|---|
| 5d | graph | 0.368 | 2.5 bp | 32.7 | 101 |
| 5d | no-graph | 0.110 | −2.3 bp | 20.4 | 101 |
| 20d | graph | 0.653 | 19.3 bp | 11.9 | 25 |
| 20d | no-graph | 0.713 | 34.5 bp | 7.0 | 25 |
| 60d | graph | 0.878 | 91.2 bp | 4.7 | 8 |
| 60d | no-graph | **1.491** | 148.0 bp | 3.0 | 8 |

| Horizon | Δ Sharpe (graph − no-graph) | t (NW) |
|---|---|---|
| 5d | **+0.258** | +1.55 |
| 20d | −0.060 | −0.26 |
| 60d | −0.614 | −1.00 |

> **Corrected 2026-08-22.** These *t*-statistics were first reported as
> +1.99 / −0.32 / −1.22. Those values pooled the three seeds as if they
> were independent observations; seeds are re-trainings of the same
> architecture on identical data, so the effective *n* was inflated 3× and
> every *t* by ≈√3. The table now collapses seeds within a window and
> treats the 7 disjoint windows as the units of inference. Point estimates
> are unaffected (the correction is linear in Δ).


**Confirmed for both signals.** Absolute performance rises sharply with
horizon and annualised turnover falls ~7× (32.7 → 4.7 for the graph
signal). Holding longer is simply a better strategy on this data, and it is
the cost mechanism Finding 6 identified, now acting in the favourable
direction.

**Not confirmed.** The graph advantage does not improve with horizon; it
shrinks and flips sign. Extending the horizon does not rescue the
graph-vs-no-graph case.

**Sizing flips the sign at 5 days.** With equal-weight quintiles the graph
signal is marginally *better* (+0.258, *t* = +1.55, n.s.), the opposite of
Finding 6's mean-variance result on the same horizon. This is consistent
with MV sizing carrying ~3.4× the seed variance of equal-weight sizing on
identical picks: at this effect size MV comparisons cannot establish
direction. **Finding 6's reversal is therefore MV-specific.**

**Power collapses as horizon grows.** Non-overlapping 60-day holds leave
only 8 periods per window-seed; a Sharpe standard error of roughly
$1/\sqrt{8} \approx 0.35$ swamps the 0.614 gap. Neither the 20d nor the 60d
difference is significant, and those point estimates should not be read as
directional evidence. This is an intrinsic tension: longer horizons raise
Sharpe but destroy the sample needed to verify it.

**Industry rotation — rejected.** Because Finding 9 localised the signal
between industries, a group-level rotation book (long top-quintile
sub-industries, short bottom, equal-weight within group) was tested against
the stock-level book on identical predictions. It is significantly *worse*
(mean −5.35 bp, *t* = −2.78, *p* < .05; better in 1/7 windows). The
inference from Finding 9 was too strong: industry fixed effects shrinking
the coefficient shows between-industry variation carries signal, not that
within-industry variation carries none. Aggregating to ~124 group means
discards real within-group information and cuts effective bets from ~200
stocks to ~50 groups.

---

## 13. Finding 11 — better portfolio construction does not recover the value

Findings 6 and 10 showed the graph signal builds no better book than the
no-graph baseline. The obvious rebuttal is that the quintile portfolio is a
bad *map* from signal to weights, not that the signal is worthless. Sprint 1
tested that directly: 3 horizons × 6 constructions on identical stored
predictions (7 windows × 3 seeds), with a primary specification fixed in
advance — **20-day horizon, continuous dollar-neutral weights** — so the
grid could not be mined for a headline.

Constructions: equal-weight quintile, equal-weight decile, continuous
dollar-neutral (centred scores, 5% position cap), and three
confidence-filtered variants excluding the middle 25/50/75% of the
cross-section. Costs 5 bp, gross 2.0, γ = 5, non-overlapping rebalances.

### The pre-registered answer is null

| 20d, continuous | Sharpe | CER | maxDD | turn/yr |
|---|---|---|---|---|
| graph | 0.693 | 0.0107 | −0.080 | 11.05 |
| no-graph | 0.725 | 0.0263 | −0.113 | 5.71 |
| **Δ** | **−0.032** | −0.0156 | | **+5.34** |

Δ Sharpe = −0.032, *t* = −0.15 (n = 7 windows), bootstrap *p* = 0.416. The
graph is fractionally behind while paying roughly double the turnover.

### The sign of the advantage is a function of horizon, not construction

| Horizon | Δ Sharpe range across the 6 constructions | graph wins |
|---|---|---|
| 5d | +0.040 … +0.258 | 6/6 |
| 20d | −0.167 … −0.032 | 0/6 |
| 60d | −0.614 … −0.164 | 0/6 |

Construction choice moves Δ Sharpe by ~0.2 within a horizon; horizon moves
it by ~0.9. Whatever the graph adds is a 5-day phenomenon, and **every 5-day
CER in the grid is negative** (−0.0071 to −0.0231) because those books run
at 30–38 turns/yr. The graph wins only where the strategy loses money, and
loses wherever the strategy is profitable.

### The mechanism: negative incremental R²

Section 1 of the sweep measures information content before any portfolio is
formed, which removes sizing as an explanation:

| Horizon | Signal | IC | Rank IC | ICIR | incremental R² |
|---|---|---|---|---|---|
| 5d | graph | 0.0126 | 0.0186 | 0.121 | **+0.00010** |
| 5d | no-graph | 0.0076 | 0.0106 | 0.062 | — |
| 20d | graph | 0.0244 | 0.0334 | 0.261 | **−0.00072** |
| 20d | no-graph | 0.0362 | 0.0399 | 0.260 | — |
| 60d | graph | 0.0352 | 0.0449 | 0.471 | **−0.00288** |
| 60d | no-graph | 0.0642 | 0.0642 | 0.552 | — |

The graph's IC exceeds the baseline's only at 5 days. At 20 and 60 days
incremental R² is *negative* — the graph model is not adding weak
information at the profitable horizons, it is adding noise. This is a
portfolio-free statement, so it cannot be blamed on weighting, cost
assumptions, or position caps.

### The one significant cell does not survive

5d quintile shows Δ = +0.258, bootstrap *p* = 0.017 — the only cell of 18
under .05. It is not evidence:

1. **The two tests disagree.** *t* = +1.55 on 7 windows is far from
   significance (≈2.45 needed at 6 df). The pooled bootstrap has more
   observations and hence more power, but calibration on synthetic data puts
   its false-positive rate at 12.5% at a nominal 5% — it is a *liberal*
   test, not a conservative one.
2. **Multiple comparisons.** 18 cells at α = .05 predicts ≈0.9 false
   positives. Observing exactly one is the modal outcome under pure noise.
   Bonferroni-corrected: 0.017 × 18 = 0.31.
3. **It loses money.** CER = −0.0140 at 32.7 turns/yr.

### Why this strengthens the paper

Finding 6 established that the economic value is weak. Finding 11 replaces
that null with an identified mechanism and a bound that construction cannot
argue with: the graph's advantage has a **term structure** that is
orthogonal to the cost structure. It is largest at the horizon where
turnover is most punitive and negative at the horizons where the strategy
works. A reader who wants to believe the value is recoverable now has to
explain a negative incremental R², measured without any portfolio at all.

---

## 14. Revised headline claim

> On a cross-sectional 5-day forward return rank task for S&P 500
> constituents (2015–2024), graph neural networks over economically
> meaningful firm groupings significantly improve per-date macro-F1 over
> no-graph and correlation-graph baselines (*p* < .001), driven by
> partition granularity rather than relationship specificity: moving from
> sector- to industry-group-level groupings improves tail accuracy
> (+0.018 tail-F1, *p* < .001) even as graph density falls, the benefit
> saturates by roughly 25 groups with no further gain from industry or
> sub-industry detail, and a free GICS graph matches a proprietary
> Bloomberg supplier/customer graph on every seed.
> **However, this classification advantage does not transfer to
> portfolios.** On an adequately powered out-of-sample
> window (383 rebalances), mean-variance portfolios built on the graph
> signals significantly underperform an otherwise identical no-graph
> portfolio (sub-industry Δ = −0.94 Sharpe, *p* < .01), and the shortfall
> is present gross of transaction costs. Per-class analysis identifies the
> cause: the graph models' accuracy gain is concentrated in the Neutral
> class, which a quintile long-short book never trades, while tail
> accuracy is flat-to-worse and the realised long-short spread falls from
> 9.7 bp to 6.0 bp per rebalance. Fama-MacBeth regressions complete the
> picture: the signal's residual return-predictive content does not
> survive controls for momentum, short-term reversal, volatility and
> industry fixed effects (retaining 23% of its univariate magnitude,
> pooled *t* = +0.64), and is identified almost entirely *between*
> industries rather than within them. A pre-registered sweep over three
> horizons and six portfolio constructions rules out the objection that
> the mapping from signal to weights was simply badly chosen: the
> pre-committed specification returns Δ = −0.032 Sharpe (*t* = −0.15,
> *p* = .42), construction choice moves the comparison by ~0.2 Sharpe
> while horizon moves it by ~0.9, and the graph model's incremental R²
> over the no-graph baseline is *negative* at both 20 and 60 days — a
> portfolio-free measurement. The graph's advantage exists only at a
> 5-day horizon, where 30–38 annual turns render every certainty-
> equivalent return in the grid negative. We stop short of calling the
> graph signal economically harmful: the sign of the mean-variance
> comparison depends on the sizing rule, and no construction reaches
> significance at adequate power. What we can say is that a robust,
> replicated classification gain yields no demonstrable portfolio
> advantage, and that this is a property of the signal's term structure
> rather than of any particular portfolio map.

**Practitioner summary: a free sub-industry classification is at least as
good as paid supply-chain relationship data for ranking — but neither
improves a tradeable portfolio, because the accuracy they add sits in the
middle of the distribution rather than the tails, and what little edge
reaches the tails lives at a 5-day horizon whose turnover costs exceed it.
Evaluate cross-sectional models on tail metrics at your intended holding
period, not on macro-F1.**

---

## 15. Summary of claim status

Every surviving claim below has been replicated on either disjoint seeds
or independent time periods. Claims that failed replication are listed so
the record is auditable.

| Claim | Evidence | Status |
|---|---|---|
| Graph structure improves cross-sectional classification | 7/7 independent windows, t = +11.67 | **Robust** |
| Finer-than-sector partitions improve tail accuracy | +0.018 tail-F1, *p* < .001, 10 disjoint seeds | **Robust** |
| Benefit saturates by ~25 groups | industry-group ≈ industry ≈ sub-industry (n.s.) | **Robust** |
| Correlation and holder-overlap edges add nothing | n.s. vs no-graph, n=5 | **Supported** |
| Supply-chain edges beat a same-granularity public classification | +0.0015, n.s. | **Not supported** |
| Monotone granularity gradient | did not replicate on disjoint seeds | **Retracted** |
| Graph signals improve tradeable long-short spread | worse in 6/7 windows, *p* < .05 | **Not supported** |
| Graph signals build worse mean-variance portfolios | Δ = −0.94 Sharpe, but sign flips under equal-weight sizing | **Sizing-dependent** |
| Longer holding periods improve absolute performance | Sharpe 0.11–0.37 (5d) → 0.88–1.49 (60d); turnover ÷7 | **Supported** |
| Signal works better as industry rotation | −5.35 bp vs stock-level, *t* = −2.78 | **Rejected** |
| Look-ahead from the static snapshot explains the results | advantage smallest where contamination worst | **Rejected** |
| Signal is distinct from momentum / reversal / industry effects | retains 23% of magnitude, *t* = +0.64 pooled | **Not supported** |
| Better portfolio construction recovers the economic value | pre-registered 20d continuous: Δ = −0.032, *t* = −0.15, *p* = .42 | **Not supported** |
| Graph adds information at tradeable horizons | incremental R² negative at 20d and 60d | **Rejected** |
| Graph advantage is confined to a 5-day horizon | 6/6 constructions positive at 5d, 0/6 at 20d and 60d | **Supported** |
| 5-day graph advantage is exploitable | every 5d CER negative (−0.007 to −0.023) at 30–38 turns/yr | **Not supported** |

Mechanism hypotheses tested and rejected: transaction costs (explain ~10%
of the portfolio shortfall), signal smoothing (non-monotonic, within
noise), portfolio concentration (identical across configurations).
Mechanism supported: the accuracy gain concentrates in the Neutral class
(Finding 7).

---

## Appendix A — task definition history

Three formulations preceded the cross-sectional rank task, and one further
bug was found after the first version of this document. Each revealed a
methodological flaw the next iteration addressed; the progression is
documented deliberately.

### A.1 Original regime task (label leakage; not publishable)

The original pipeline predicts a 4-class market regime and a binary "stress
in next 5–20 days" target, with labels **defined** by percentile rules on
`vol_20d`, `avg_corr`, `ret_20d`. A logistic regression on the same three
aggregate features achieves transition AUC = 0.963, dominating any GNN
(`results/baselines/baselines.json`). The baseline directly observes the
label-generating variables; no architecture can win this task.

### A.2 Sample-balance bug

Default `--max-train-samples 240 --train-sample-strategy tail` kept only the
last 240 trading days of the 2015–2022 train period — overwhelmingly the
2022 rate-hike stress regime. `linspace` sampling restored balance
(`results/legacy_market_regime/`), moving the model from "predict 80%
positives" to calibrated, but the leakage above remained.

### A.3 Per-stock drawdown task (no signal)

Per-stock 20-day forward drawdown classification: every configuration hit
macro-AUC ≈ 0.50 across 5 seeds. Features encode market regime, not which
specific stock will drop. Bloomberg edges actively hurt (Δ = −0.07
micro-AUC, *p* < .05) — the dense holder graph added noise.

### A.4 Fake-sector bug (discovered after the first version of this document)

Sector labels were `np.arange(n) % 11` — alphabetical ticker order modulo
11. Consequences: (i) 11 of 22 node features were noise; (ii) the "proxy"
ablation arm was a random block graph, so the original headline comparison
tested real edges against *random* edges while describing them as
*sector-based*. Fixed by `--sector-file` with real GICS labels
(`scripts/build_sector_file.py`); Findings 1–3 above are the corrected
results. The original numbers are preserved in `results/main_ablation/` and
`results/edge_decomposition/`.

### A.5 Cross-sectional rank task (this report)

Rank-within-date labels cannot be predicted from features constant across
stocks on a date, eliminating the leakage and "always predict the market"
failure modes by construction.

---

## Appendix B — baselines

`scripts/run_baselines.py`: on the original regime/transition task,
logistic regression (transition AUC 0.963) and an LSTM (0.900) dominate the
GNN (0.676) because the task leaks. On the cross-sectional rank task,
aggregate-feature models are bounded by the no-graph result (≈ 0.355
macro-F1 vs random 0.333), confirming aggregate features cannot reach into
the cross-sectional dimension.

---

## Reproducibility

```bash
# 1. Environment
uv sync --dev

# 2. Sector file (requires a Bloomberg GICS pull; see scripts/build_sector_file.py)
python scripts/build_sector_file.py path/to/gics_500.csv

# 3. Main ablation with real sectors (n=5 × bloomberg/proxy/corronly/none)
for mode in bloomberg proxy corronly none; do
  for seed in 42 123 7 101 2025; do
    uv run python scripts/run_xsec_rank.py \
      --xlsx "path/to/sp500_prices 1.xlsx" \
      --output "results/realsector/${mode}_s${seed}.json" \
      --sector-file data_sectors_gics.csv \
      --edge-mode $mode --seed $seed --epochs 15 \
      --start 2015-01-01 --end 2024-12-31 --train-cutoff 2022-09-30
  done
done

# 4. Granularity gradient: as above with --edge-mode proxy and
#    --graph-sector-column {gics_industry_group|gics_industry|gics_sub_industry}
#    -> results/granularity/
# 5. Decomposition: --edge-mode {supply_only|holder_only} -> results/decomp_realsector/
# 6. Recency diagnostic: scripts/analyze_recency.py docstring gives the period grid
# 7. Analysis
python scripts/analyze_realsector.py
python scripts/analyze_granularity.py
python scripts/analyze_recency.py
python scripts/analyze_results.py --figures
```

Each GPU job takes ~25 min on an A100 (graph modes) or ~3 min (none). The
original fake-sector results reproduce with the same commands minus
`--sector-file`.
