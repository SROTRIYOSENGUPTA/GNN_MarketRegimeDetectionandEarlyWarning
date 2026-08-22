# Changelog

All notable changes to this project are logged here, newest first. This file
exists to separate work completed as of the original ECE538 course
submission from anything added afterward.

The `v1.0-ece538-submission` git tag marks the exact commit as submitted for
the course. Compare any later state against it with:

```bash
git diff v1.0-ece538-submission..main
```

## [Unreleased]

### Added — Sprint 1: portfolio-mapping sweep (Finding 11)

- `scripts/run_sprint1_portfolio_engine.py` — pre-registered sweep over 3
  horizons x 6 portfolio constructions on stored predictions, with
  portfolio-free information metrics (IC, rank IC, ICIR, incremental R^2)
  and a moving-block bootstrap for Sharpe differences. Primary
  specification (20d, continuous dollar-neutral) fixed in advance so the
  18-cell grid cannot be mined.
- `results/sprint1/` — console output and JSON for the Amarel run.
- `figures/paper/fig5_termstructure.pdf` — the term-structure result:
  horizon governs the graph-vs-no-graph sign while construction barely
  moves it, and incremental R^2 goes negative at tradeable horizons.
- Result: the pre-registered cell is null (Delta Sharpe = -0.032,
  t = -0.15, bootstrap p = 0.42). This closes the "the portfolio map was
  just badly chosen" objection to Findings 6 and 10, and replaces a bare
  null with an identified mechanism.

### Corrected — findings that changed after re-testing

- **Seed-replicate inflation in portfolio inference — corrected.** Every
  portfolio test pooled window x seed as if seeds were independent
  observations. Seeds are re-trainings of one architecture on identical
  data, so effective n was inflated 3x, every t-statistic by ~sqrt(3), and
  the pooled block bootstrap saw each calendar return three times. Now
  collapsed within window, leaving the 7 disjoint windows as the units of
  inference. Point estimates unaffected (the correction is linear in the
  difference). Finding 10's horizon t-statistics were published as
  +1.99 / -0.32 / -1.22 and are actually **+1.55 / -0.26 / -1.00**.
- **Monotone granularity claim leaked back into prose.** The retraction was
  recorded in the claim table but the RESULTS.md headline and the
  granularity interpretation still asserted a monotone gradient. Both now
  describe the effect as a step saturating at ~25 groups, matching the
  claim table.

- **Fake sector labels.** `run_xsec_rank.py` assigned sectors as
  `np.arange(n) % 11` (alphabetical ticker order modulo 11). Two
  consequences: 11 of 22 node features were noise, and the `proxy` ablation
  arm was a *random block graph* rather than a sector graph. RESULTS.md's
  headline — that the gain was "specifically attributable to real economic
  relationships" — rested on beating random edges. Against a real sector
  partition the premium is not significant (t=0.82). Fixed via
  `--sector-file` with real GICS labels.
- **Monotone granularity gradient — retracted.** Did not replicate on 10
  disjoint seeds. The robust result is a *step*: sector-level grouping is
  too coarse, anything finer is better (+0.018 tail-F1, p<.001), and the
  benefit saturates by ~25 groups.
- **Portfolio nulls — superseded.** An n=5 "no significant difference" was
  underpowered, not neutral. On 383 rebalances the graph signals
  significantly *underperform* no-graph (Δ=−0.94 Sharpe, p<.01).

### Added — portfolio layer (Markowitz MPT)

- `market_regime_gnn/portfolio/mean_variance.py`: covariance estimation,
  convex signal shrinkage (Michaud 1989), mean-variance optimiser with
  gross-exposure / dollar-neutral / direction constraints and a
  transaction-cost-aware turnover penalty.
- `scripts/run_mpt_backtest.py`: walk-forward backtest over six portfolio
  configurations reporting Sharpe, Sortino, drawdown and turnover net of
  costs.
- `--save-predictions` on `run_xsec_rank.py` to persist per-(date, ticker)
  probabilities and realised forward returns.

### Added — robustness and mechanism analysis

- **Recency diagnostic** (`scripts/analyze_recency.py`): the static
  supply-chain snapshot's look-ahead does not explain the results — the
  advantage is *smallest* where contamination is worst.
- **Period sweep** (`results/periods/`): 7 non-overlapping evaluation
  windows. Graph structure improves macro-F1 in 7/7 (t=+11.67); long-short
  spread is worse in 6/7 (p<.05).
- **Seed sweep** (`results/seedsweep/`): 10 disjoint seeds × 4 GICS levels.
- **Per-class analysis** (`scripts/analyze_per_class.py`): the mechanism.
  The graph advantage sits almost entirely in the *Neutral* class — 60% of
  the mass, which a quintile portfolio never trades — while tail accuracy
  is flat-to-worse.
- **Weight attribution** (`scripts/analyze_weight_attribution.py`): tested
  and rejected the signal-smoothing and portfolio-concentration hypotheses.

### Added — data and tooling

- `scripts/fetch_sectors.py`, `scripts/build_sector_file.py`: GICS sector
  file construction (Bloomberg primary, yfinance crosswalk fallback).
  The GICS data itself is gitignored — proprietary to S&P/MSCI.
- Declared previously-missing dependencies: `cvxpy`, `scikit-learn`,
  `matplotlib` (the latter two were already imported by existing scripts).
- Tests: `tests/test_mean_variance.py`, `tests/test_mpt_backtest_smoke.py`.

### Changed

- `RESULTS.md` rewritten around the corrected findings, with a claim-status
  table distinguishing robust / supported / retracted / robustly-false, and
  a revision note explaining what changed and why.

## [v1.0-ece538-submission] — course submission baseline

Everything through this point was completed as the ECE538 project
(Srotriyo Sengupta and Yifan Zhang):

- Dynamic heterogeneous GNN for market regime detection and stress
  early-warning (`market_regime_gnn/`, legacy source under
  `GNNsMarketRegimeDetection&Early-Warning/`).
- 30-stock Yahoo Finance benchmark (Setting A) and 500-stock Bloomberg
  workbook pilot (Setting B) — see `Report.md`.
- Cross-sectional 5-day forward-return-rank ablation — see `RESULTS.md`.
- Diagnosis and writeup of label leakage in the original regime/transition
  task, and of a training-sample-balance bug (`results/legacy_market_regime/`).
