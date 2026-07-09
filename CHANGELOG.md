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

Nothing yet — this section fills in as new work lands.

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
