# How to upload these to GitHub

This folder contains all the new code, results, figures, and write-ups
generated in this work session. Upload them manually into your repo at
https://github.com/SROTRIYOSENGUPTA/GNN_MarketRegimeDetectionandEarlyWarning .

## Folder mapping

Copy contents into your local clone of the repo as follows:

| In this folder | → In the repo |
|---|---|
| `RESULTS.md` | `RESULTS.md` (repo root) |
| `scripts/*` | `scripts/` |
| `results/*` | `results/` (create if missing) |
| `figures/v2/*` | `figures/v2/` (create if missing) |

## Suggested README addition

Optionally, add a section to `README.md` linking the new results (write
it in your own words; here's a starting point you can edit):

```markdown
## New experimental results (May 2026)

A cross-sectional 5-day forward return rank classification task evaluates
the heterogeneous-graph approach with statistical rigor. Across 5 random
seeds and 3 walk-forward train cutoffs, a dynamic heterogeneous GNN using
real Bloomberg supplier/customer/holder edges significantly outperforms
non-Bloomberg ablations. See [RESULTS.md](./RESULTS.md) for full evaluation.

- Pipelines: `scripts/run_xsec_rank.py`, `scripts/run_per_stock.py`, `scripts/run_baselines.py`
- Raw outputs: `results/`
- Figures: `figures/v2/`
- Reproduce all tables/figures: `python scripts/analyze_results.py --figures`
```

## Suggested commit message (write in your own voice)

Something like:
```
Add cross-sectional rank GNN: BBG-edge significance results

* New cross-sectional 5-day forward return rank classification task
  (closes the label-leakage flaw of the original regime task)
* Heterogeneous GNN with real Bloomberg supplier/customer/holder edges
  achieves macro-F1 0.370 ± 0.004 across n=5 seeds
* Significant lift over proxy edges (p<.05), correlation-only (p<.01),
  and per-stock LSTM (p<.001)
* Supply-only edges reproduce 96% of the gain; holder-only is harmful alone
* Effect direction stable across 3 walk-forward train cutoffs
```

## File inventory

- `RESULTS.md` — main paper-grade writeup with statistical tests
- `scripts/run_xsec_rank.py` — cross-sectional rank pipeline (the headline experiment)
- `scripts/run_per_stock.py` — per-stock drawdown pipeline (documented negative result)
- `scripts/run_baselines.py` — logistic + LSTM baselines
- `scripts/label_diagnostic.py` — labeling threshold + named-event coverage tool
- `scripts/analyze_results.py` — reproduces every table and figure
- `results/main_ablation/` — 20 JSONs: 4 configs × 5 seeds
- `results/edge_decomposition/` — 6 JSONs: holder_only + supply_only × 3 seeds
- `results/walk_forward/` — 12 JSONs: BBG/NoGraph × 3 seeds × 2 alternative cutoffs
- `results/baselines/baselines.json` — logistic + LSTM on legacy regime task
- `results/legacy_market_regime/` — older regime sweep for context
- `figures/v2/fig1_main_ablation.png`
- `figures/v2/fig2_edge_decomposition.png`
- `figures/v2/fig3_walk_forward.png`
