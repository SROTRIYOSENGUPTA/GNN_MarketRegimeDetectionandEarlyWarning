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

### Added
- Markowitz mean-variance portfolio construction layer
  (`market_regime_gnn/portfolio/mean_variance.py`) on top of the
  cross-sectional rank model's predictions — covariance estimation, convex
  shrinkage of the return signal, and a mean-variance optimizer (closed-form
  unconstrained, `cvxpy`-backed for gross-exposure / dollar-neutral /
  long-short direction constraints).
- `--save-predictions` flag on `scripts/run_xsec_rank.py` to persist
  per-(date, ticker) softmax probabilities and realized forward returns,
  needed for backtesting (previously discarded after computing aggregate
  metrics).
- `scripts/run_mpt_backtest.py`: walk-forward portfolio backtest evaluating
  six configurations (GNN-derived vs. sample covariance, Bloomberg vs.
  no-graph return signal, mean-variance vs. equal-weight sizing, and an
  equal-weight-universe benchmark), reporting Sharpe, Sortino, max drawdown,
  and turnover net of transaction costs.
- `scripts/analyze_results.py`: new `mpt_backtest()` table/figure
  (`figures/v2/fig4_mpt_backtest.png`, `RESULTS_MPT.md`), reusing the
  existing `paired_t`/`stats` helpers.
- Added missing `cvxpy`, `scikit-learn`, and `matplotlib` to
  `pyproject.toml` — the latter two were already imported by existing
  scripts (`run_xsec_rank.py`, `analyze_results.py --figures`) but were
  never declared as dependencies.
- Tests: `tests/test_mean_variance.py`, `tests/test_mpt_backtest_smoke.py`.

### Why
Classification metrics (macro-F1) don't say whether the Bloomberg-edge
advantage documented in RESULTS.md is economically meaningful once you're
allocating capital rather than scoring predictions. This adds that
evaluation layer. See the mean-variance module's docstring for the
Michaud (1989) "error maximization" rationale behind the shrinkage knob.

## [v1.0-ece538-submission] — course submission baseline

Everything through this point was completed as the ECE538 project
(Srotriyo Sengupta and Yifan Zhang):

- Dynamic heterogeneous GNN for market regime detection and stress
  early-warning (`market_regime_gnn/`, legacy source under
  `GNNsMarketRegimeDetection&Early-Warning/`).
- 30-stock Yahoo Finance benchmark (Setting A) and 500-stock Bloomberg
  workbook pilot (Setting B) — see `Report.md`.
- Cross-sectional 5-day forward-return-rank ablation isolating real
  Bloomberg supplier/customer/holder edges from proxy and synthetic edges
  — see `RESULTS.md`.
- Diagnosis and writeup of label leakage in the original regime/transition
  task, and of a training-sample-balance bug (`results/legacy_market_regime/`).
