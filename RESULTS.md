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
- `scripts/analyze_results.py` — original tables, walk-forward, MPT backtest

---

## TL;DR

> On cross-sectional 5-day forward return rank prediction (3-class, 500
> stocks, 2015–2024), graph structure over meaningful economic groupings
> improves per-date macro-F1 over a no-graph baseline (*p* < .001, n=5
> seeds). However, the improvement is **not** specific to proprietary
> supply-chain relationships: a graph built from **free GICS sub-industry
> labels (0.378 ± 0.003) outperforms the Bloomberg supplier/customer graph
> (0.372 ± 0.005) on all 5 seeds** (*p* < .10). Across the GICS hierarchy,
> performance rises monotonically with partition fineness — sector (11
> groups) → sub-industry (124 groups) — while edge count *falls* 7×, so
> grouping precision, not connectivity, is the driver. Correlation-only
> graphs and institutional-holder graphs do not beat no-graph at all. A
> recency diagnostic shows the static supply-chain snapshot's look-ahead
> bias does not explain these results. **But none of this transfers to
> portfolios.** On an adequately powered out-of-sample window (383
> rebalances), mean-variance portfolios built on the graph signals
> significantly *underperform* an identical no-graph portfolio
> (sub-industry Δ = −0.94 Sharpe, *p* < .01), gross of transaction costs.
> Per-class analysis explains why: the graph models' accuracy gain sits
> almost entirely in the **Neutral** class — 60% of the mass, which a
> quintile long-short book never trades — while tail accuracy is
> flat-to-worse and the realised long-short spread falls from 9.7 bp to
> 6.0 bp per rebalance.

**One-line version:** *partition granularity, not relationship specificity,
drives graph-based cross-sectional return predictability — the best
partition is free, and none of it is tradeable, because macro-F1 rewards
the part of the distribution portfolios cannot touch.*

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

**Interpretation.** Performance rises monotonically as the partition gets
finer, while edge count falls 7× — so the mechanism is grouping *precision*,
not connectivity. Message passing among a small set of true economic peers
(a sub-industry here averages ~4 firms) beats both broad sector pooling and
curated bilateral supply-chain links. The best-performing graph in the
entire study is built from a free, public classification.

The *p* < .10 on the headline pairwise test is suggestive rather than
decisive; the 5/5 seed sweep and the monotone gradient are the stronger
evidence. Treat "sub-industry ≥ Bloomberg" as established and
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

## 8. Finding 6 — graph signals build *significantly worse* portfolios (n=5)

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

**Both graph signals significantly underperform the no-graph signal**, and
the best classifier produces the worst portfolio. An earlier n=3 claim of
significant Bloomberg underperformance, later retracted when it failed to
replicate at n=5 on the short window, is reinstated on the powered test.

**It is not a transaction-cost effect.** Decomposing gross vs net:

| Signal | Gross Sharpe | Net Sharpe | Cost drag | Turnover |
|---|---|---|---|---|
| Sub-industry | +0.257 | +0.144 | 8.4 bp | 1.678 |
| Bloomberg | +0.657 | +0.595 | 4.5 bp | 0.904 |
| No-graph | +0.982 | +0.928 | 3.4 bp | 0.682 |

The per-period gross-return gap is −44.4 bp against a cost gap of +5.0 bp,
so **costs explain ~10%** of the shortfall (sub-industry vs no-graph gross:
*p* < .05). The graph signals build worse portfolios before any friction;
turnover is a symptom, not the cause.

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

## 11. Revised headline claim

> On a cross-sectional 5-day forward return rank task for S&P 500
> constituents (2015–2024), graph neural networks over economically
> meaningful firm groupings significantly improve per-date macro-F1 over
> no-graph and correlation-graph baselines (*p* < .001), driven by
> partition granularity rather than relationship specificity: performance
> rises monotonically from sector- to sub-industry-level groupings even as
> graph density falls 7×, and a free GICS sub-industry graph matches or
> exceeds a proprietary Bloomberg supplier/customer graph on every seed.
> **However, this classification advantage does not transfer to
> portfolios — it reverses.** On an adequately powered out-of-sample
> window (383 rebalances), mean-variance portfolios built on the graph
> signals significantly underperform an otherwise identical no-graph
> portfolio (sub-industry Δ = −0.94 Sharpe, *p* < .01), and the shortfall
> is present gross of transaction costs. Per-class analysis identifies the
> cause: the graph models' accuracy gain is concentrated in the Neutral
> class, which a quintile long-short book never trades, while tail
> accuracy is flat-to-worse and the realised long-short spread falls from
> 9.7 bp to 6.0 bp per rebalance.

**Practitioner summary: a free sub-industry classification is at least as
good as paid supply-chain relationship data for ranking — but neither
improves a quintile portfolio, because the accuracy they add sits in the
middle of the distribution rather than the tails. Evaluate cross-sectional
models on tail metrics, not macro-F1.**

---

## 12. Summary of claim status

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
| Graph signals improve tradeable long-short spread | worse in 6/7 windows, *p* < .05 | **Robustly false** |
| Look-ahead from the static snapshot explains the results | advantage smallest where contamination worst | **Rejected** |

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
