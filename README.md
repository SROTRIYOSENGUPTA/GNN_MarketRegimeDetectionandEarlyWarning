# Graph Neural Networks for Cross-Sectional Equity Prediction

A financial graph learning project studying whether graph neural networks
over inter-firm relationships improve cross-sectional equity prediction —
and whether any improvement is economically tradeable.

The project began as market regime detection and early warning (the ECE538
course scope, still documented below and in [Report.md](./Report.md)). It
now centres on a cross-sectional 5-day forward return rank task, because
the original regime labels were found to leak (see
[RESULTS.md](./RESULTS.md) Appendix A).

![Final project poster](figures/final_project_poster.svg)

## Current findings

Full evidence, statistical tests and a claim-status table are in
**[RESULTS.md](./RESULTS.md)**. Headline:

| Claim | Evidence | Status |
|---|---|---|
| Graph structure improves cross-sectional classification | 7/7 independent windows, t = +11.67 | **Robust** |
| Finer-than-sector partitions improve tail accuracy | +0.018 tail-F1, *p* < .001, 10 disjoint seeds | **Robust** |
| Benefit saturates by ~25 groups | industry-group ≈ industry ≈ sub-industry | **Robust** |
| Proprietary supply-chain data beats free GICS labels | Δ = +0.003, n.s. | **Not supported** |
| Monotone granularity gradient | did not replicate on disjoint seeds | **Retracted** |
| Graph signals improve tradeable long-short spread | worse in 6/7 windows, *p* < .05 | **Robustly false** |

**The central result is a dissociation.** Graph structure reliably improves
per-date macro-F1, but Markowitz mean-variance portfolios built on those
same signals significantly *underperform* an identical no-graph portfolio
(Δ = −0.94 Sharpe, *p* < .01), and the shortfall is present gross of
transaction costs. Per-class analysis explains why: the accuracy gain sits
almost entirely in the **Neutral** class — 60% of the distribution, which a
quintile long-short book never trades — while tail accuracy is
flat-to-worse.

> Practical implication: evaluate cross-sectional models on tail metrics and
> realised long-short spread, not macro-F1. And exhaust free industry
> classifications before licensing relationship data.

Two quantities are large relative to the effects measured, and are reported
throughout rather than hidden: run-to-run CUDA nondeterminism (which alone
moves the granularity comparison between *p* < .05 and n.s.) and
evaluation-window choice (single-window spread estimates swing ~16 bp
between runs of an identical configuration). Conclusions are therefore drawn
from multi-window aggregates and disjoint-seed replications only.

### Reproducing

```bash
uv sync --dev
python scripts/analyze_results.py --figures   # original tables/figures
python scripts/analyze_realsector.py          # real-sector ablation
python scripts/analyze_granularity.py         # granularity + decomposition
python scripts/analyze_recency.py             # look-ahead diagnostic
python scripts/analyze_per_class.py results/... # mechanism
```

- Pipelines: `scripts/run_xsec_rank.py`, `scripts/run_mpt_backtest.py`,
  `scripts/run_per_stock.py`, `scripts/run_baselines.py`
- Raw experiment outputs: `results/` — see `periods/`, `seedsweep/`,
  `realsector/`, `attribution/`, `mpt_program/`
- Figures: `figures/v2/`

## Project Status

- **Original scope (ECE538 course project):** the regime-detection material
  described below and in [Report.md](./Report.md), frozen at git tag
  [`v1.0-ece538-submission`](../../releases/tag/v1.0-ece538-submission).
- **Post-submission extensions and corrections:** tracked in
  [CHANGELOG.md](./CHANGELOG.md). Compare against the submitted baseline
  with `git diff v1.0-ece538-submission..main`.

> **Note on revisions.** Several claims in the original write-up were
> corrected after further testing — most importantly, an earlier headline
> attributing the predictive gain "specifically" to proprietary Bloomberg
> relationships. That comparison used a sector-proxy baseline which, through
> a labelling bug, was a *random* graph; against a real sector partition the
> premium is not significant. The original results remain in
> `results/main_ablation/` and are reproducible; RESULTS.md documents what
> changed and why.

## Markowitz portfolio layer

`market_regime_gnn/portfolio/mean_variance.py` implements mean-variance
construction on top of the model's predictions: covariance estimation,
convex signal shrinkage (Michaud 1989), and an optimiser supporting
gross-exposure caps, dollar-neutrality, long/short direction masks and a
transaction-cost-aware turnover penalty. `scripts/run_mpt_backtest.py`
runs the walk-forward backtest. This layer is what establishes the
dissociation above — it is the economic-significance test, not a trading
system.


## What The Main Project Does

The main idea is to turn the stock market into a dynamic multi-relational graph:

- nodes are stocks
- edges encode relationships between stocks
- graphs evolve daily
- a temporal model consumes a rolling window of graph snapshots

Instead of only predicting returns, the main pipeline predicts:

- `regime_label`: a 4-class description of current market conditions
- `transition_label`: a binary early-warning target that becomes positive when `Stress` appears in the next `5-20` trading days

This makes the project closer to a market monitoring and systemic-risk early-warning system than a standard alpha model.

## Main Project: Dynamic Regime GNN

Directory: [`GNNsMarketRegimeDetection&Early-Warning`](./GNNsMarketRegimeDetection%26Early-Warning)

### Pipeline

The real-data entry point is [`GNNsMarketRegimeDetection&Early-Warning/run_real_data.py`](./GNNsMarketRegimeDetection%26Early-Warning/run_real_data.py).

It does the following:

1. Downloads a curated sample of S&P 500 stocks, plus `SPY` and `^VIX`, from Yahoo Finance.
2. Builds a `37`-dimensional daily feature vector per stock.
3. Computes rule-based market regime labels from SPY returns, realized volatility, and average cross-sectional correlation.
4. Builds daily heterogeneous graphs with three relation types:
   - `correlation`
   - `etf_cohold`
   - `supply_chain`
5. Slices the data into rolling sequences of `T=30` graph snapshots.
6. Trains a dynamic GNN with two heads:
   - a 4-way regime classifier
   - a binary transition / early-warning head

### Model architecture

The main model is defined in [`GNNsMarketRegimeDetection&Early-Warning/models/dynamic_regime_gnn.py`](./GNNsMarketRegimeDetection%26Early-Warning/models/dynamic_regime_gnn.py).

Its structure is:

1. `NodeFeatureEncoder`
   Projects raw stock features from `37 -> 128`.
2. `SpatialRGCN`
   Applies relation-aware message passing over the three edge types.
3. Graph pooling
   Aggregates node embeddings into one graph-level embedding per day.
4. Temporal encoder
   Uses either:
   - `LSTM` by default
   - `Transformer` as an alternative
5. Dual prediction heads
   - `RegimeClassifierHead`
   - `TransitionLogitHead`

In short:

`30 daily heterogeneous graphs -> spatial GNN -> temporal encoder -> current regime + future stress warning`

### Label generation logic

The statistical labeling engine lives in [`GNNsMarketRegimeDetection&Early-Warning/data/label_generator.py`](./GNNsMarketRegimeDetection%26Early-Warning/data/label_generator.py).

It derives four market states using expanding-window thresholds to reduce look-ahead bias:

- `Stress`
  High volatility and high average cross-sectional correlation.
- `Crash`
  Large negative recent return plus elevated volatility.
- `Bull`
  Positive recent return plus relatively low volatility.
- `Liquidity`
  Residual bucket for everything else.

The early-warning target is:

- `transition_label = 1` if a `Stress` regime appears anywhere in the next `5-20` trading days

### Graph construction

Graph building logic lives in [`GNNsMarketRegimeDetection&Early-Warning/data/hetero_dataset.py`](./GNNsMarketRegimeDetection%26Early-Warning/data/hetero_dataset.py).

Each daily graph has one node type, `stock`, and three edge types:

- `correlation`
  Built from rolling return correlations using top-K positive and bottom-K negative neighbors per node.
- `etf_cohold`
  Intended to represent ETF co-holding overlap. If real holdings are not provided, the current prototype falls back to a sector/sub-industry proxy.
- `supply_chain`
  Intended to represent production-network links. If no external adjacency is provided, the current prototype falls back to a sparse synthetic adjacency.

## Repository Structure

```text
.
├── market_regime_gnn/
│   ├── config.py
│   ├── run_real_data.py
│   ├── train.py
│   ├── data/
│   ├── models/
│   └── portfolio/            # Markowitz mean-variance layer
│       └── mean_variance.py
├── scripts/
│   ├── run_xsec_rank.py      # cross-sectional rank pipeline (main)
│   ├── run_mpt_backtest.py   # portfolio backtest
│   ├── build_sector_file.py  # GICS sector file construction
│   ├── fetch_sectors.py
│   ├── analyze_results.py    # original tables/figures
│   ├── analyze_realsector.py # real-sector ablation
│   ├── analyze_granularity.py
│   ├── analyze_recency.py    # look-ahead diagnostic
│   ├── analyze_per_class.py  # mechanism
│   └── analyze_weight_attribution.py
├── results/                  # raw experiment JSON, one dir per study
│   ├── periods/              # 7-window robustness sweep
│   ├── seedsweep/            # 10 disjoint seeds x 4 GICS levels
│   ├── realsector/           # main ablation, real sectors
│   ├── attribution/          # per-class + weight attribution
│   └── mpt_program/          # portfolio backtests
├── RESULTS.md                # findings, tests, claim-status table
├── RESULTS_MPT.md            # portfolio-layer detail
├── CHANGELOG.md
├── pyproject.toml
├── uv.lock
└── GNNsMarketRegimeDetection&Early-Warning/   # legacy course-project layout
    ├── config.py
    ├── run_real_data.py
    ├── train.py
    ├── data/
    │   ├── hetero_dataset.py
    │   └── label_generator.py
    ├── models/
    │   └── dynamic_regime_gnn.py
    └── tests/
```

## Environment Setup

The repository already includes [`pyproject.toml`](./pyproject.toml) and [`uv.lock`](./uv.lock).

Required runtime stack:

- Python `>=3.11,<3.12`
- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `yfinance`

Recommended setup:

```bash
uv sync --dev
```

This creates a local `.venv`, installs runtime and test dependencies from the lockfile, and installs the local project in editable mode so package imports and console scripts work from outside the repo root.

If you prefer not to activate the environment manually, run commands through `uv run`.

## Python API Notes

The installable Python package is exposed as `market_regime_gnn`. The legacy source layout remains under `GNNsMarketRegimeDetection&Early-Warning/` for script compatibility.

The regime-detection project has an import-friendly wrapper package:

```python
from market_regime_gnn import RegimeConfig
from market_regime_gnn.data.label_generator import generate_market_labels
from market_regime_gnn.models.dynamic_regime_gnn import DynamicRegimeGNN
```

The wrapper keeps the legacy source layout working for in-repo scripts, but the installable package now ships a bundled `market_regime_gnn._legacy` implementation so editable installs and built wheels behave the same way.

After `uv sync --dev`, these imports work from any current working directory as long as you use the project environment's Python:

```bash
.venv/bin/python -c "import market_regime_gnn"
```

## Build And Install

Build an sdist and wheel:

```bash
uv build
```

The generated wheel includes the bundled `market_regime_gnn._legacy` package, so isolated installs do not need the repository checkout path to resolve the main regime-detection prototype.

## Data Access

The main 500-stock experiments require a Bloomberg-sourced workbook
(`sp500_prices 1.xlsx`) that is **not included in this repository** because
it is proprietary licensed data through a Hedge Fund's license. Specifically:

- The file is licensed market data sourced via Bloomberg Terminal and the
  Bloomberg Data License program.
- It contains daily price/volume time series for ~500 S&P 500 constituents
  from 2015–2024, plus reference metadata (Top Suppliers, Top Customers,
  Top 20 institutional Holders) per ticker.
- Redistribution to third parties is prohibited under the Bloomberg
  terms of use.

You can potentially recreate an equivalent workbook if you have equivalent access
with two sheets containing the following columns:

| Column | Description |
| --- | --- |
| `date` | Trading day (datetime) |
| `ticker` | Bloomberg ticker (e.g. `AAPL UW`) |
| `px_last` | Daily close price |
| `px_volume` | Daily volume |
| `Ticker` | (metadata row only) Same Bloomberg ticker |
| `Top Suppliers` | Comma-separated supplier tickers (e.g. `FLEX US Equity, ...`) |
| `Top Customers` | Comma-separated customer tickers |
| `Top 20 Holders` | Comma-separated institutional holder names |

Place the file at the path expected by the entry-point scripts:

```bash
# default location used by scripts/
./sp500_prices\ 1.xlsx

## How To Run

Because the main project directory contains `&`, quote that path in shell commands.

### Run the main regime-detection pipeline

```bash
uv run python "GNNsMarketRegimeDetection&Early-Warning/run_real_data.py"
```

Equivalent module entry point:

```bash
uv run python -m market_regime_gnn.run_real_data
```

See available CLI options without starting a run:

```bash
uv run python -m market_regime_gnn.run_real_data --help
```

Installed console script:

```bash
uv run market-regime-real-data --help
```

What it does:

- downloads Yahoo Finance data
- generates regime labels
- builds temporal heterogeneous graphs
- trains the dynamic regime GNN
- prints validation metrics and prediction statistics
- uses `--train-cutoff` as the last training day and starts validation on the next calendar day
- currently uses a fixed curated 30-stock sample instead of a user-configurable universe size

For the real-data CLI, `--train-cutoff` must lie inside the inclusive `[--start, --end]` range. Validation begins on the next calendar day after the cutoff, so if `--train-cutoff` equals `--end`, the validation split is intentionally empty.
If the training split becomes empty after the model's warm-up and label horizon rules are applied, the script fails fast with a clear error telling you to widen the training window.
The `--device` option accepts `cpu`, `cuda`, `cuda:0`, or `mps`. Unsupported or unavailable accelerators fail fast before the run proceeds, and the quick sanity-check forward pass uses the same device as training.
The CLI validates integer hyperparameters before downloading data: `--epochs`, `--batch-size`, and `--grad-accum-steps` must be positive; correlation-neighbour counts must be non-negative; and `--seq-len` and `--rolling-zscore-window` must be positive.
When a CLI argument is invalid, the script returns a normal argparse `usage: ... error: ...` message instead of a Python traceback. Runtime/data failures that happen after argument parsing still surface as normal runtime errors rather than being mislabeled as usage mistakes.

## Testing And Smoke Checks

Recommended full test command:

```bash
uv run pytest -q
```

Validated locally:

```bash
uv run pytest -q
uv build
uv run python "GNNsMarketRegimeDetection&Early-Warning/data/hetero_dataset.py"
uv run python "GNNsMarketRegimeDetection&Early-Warning/models/dynamic_regime_gnn.py"
uv run python "GNNsMarketRegimeDetection&Early-Warning/train.py"
```

Notes:

- `uv run pytest -q` covers lightweight main-project smoke tests, CLI validation, and import-regression tests.
- The project now includes setuptools build metadata and console scripts, so `uv sync --dev` and `uv build` produce importable packages instead of relying only on the repo root being on `sys.path`.
- The `market_regime_gnn` package now bundles the main prototype's legacy implementation, which avoids wheel-install regressions caused by resolving modules from the checkout path.
- Dataset boundary tests now guard against over-trimming early valid samples in the regime-detection dataset.
- The new `market_regime_gnn` wrapper is covered by root-level tests so main-project labeling logic can be imported without relying on ad hoc `sys.path` edits.
- The real-data entry point now provides a real `--help` path instead of immediately starting downloads and training.
- Script-mode import fallbacks are now guarded so package imports raise real dependency errors instead of silently dropping into `sys.path` hacks.
- The real-data scripts depend on Yahoo Finance availability and network access.

## Current Limitations

- The main regime branch still uses proxy relations for `etf_cohold` and `supply_chain` unless external datasets are provided.
- The market-regime branch has smoke checks, but its dedicated in-project `tests/` package is still sparse.
- Real-data runs pull directly from Yahoo Finance, so reproducibility depends on upstream data availability and any ticker-history revisions.

## Recent Fixes

The following issues were addressed while updating this repository:

- README was aligned with the current repo state: `pyproject.toml`, `uv.lock`, `uv`-based setup, and validated commands are now documented accurately.
- Deprecated Pandas `fillna(method=...)` calls in the real-data and labeling code were replaced with `.ffill()` / `.bfill()` equivalents for forward compatibility.
- Dataset warm-up boundary logic was corrected so the regime-detection dataset keeps the earliest valid supervised samples instead of silently dropping them.
- The main regime-detection prototype now has a wrapper package and relative-import-friendly internals, so it can be imported programmatically despite the legacy directory name.
- The wrapper package no longer depends on the repository checkout path at runtime, so isolated wheel installs can import `market_regime_gnn` successfully.
- Shared default `RegimeConfig()` constructor arguments were replaced with per-call `None` sentinels, avoiding accidental cross-instance config reuse.
- Import fallback branches now distinguish direct script execution from package imports, preventing misleading fallback behavior when a real dependency import fails.

## Sector labels (`--sector-file`)

The cross-sectional pipeline takes real GICS labels via `--sector-file`, and
`--graph-sector-column` selects which level of the hierarchy builds the
graph (`sector`, `gics_industry_group`, `gics_industry`,
`gics_sub_industry`):

```bash
python scripts/build_sector_file.py path/to/gics_500.csv   # -> data_sectors_gics.csv

python scripts/run_xsec_rank.py \
  --xlsx "path/to/sp500_prices 1.xlsx" \
  --sector-file data_sectors_gics.csv \
  --graph-sector-column gics_sub_industry \
  --edge-mode proxy --seed 42 --output out.json
```

**Without `--sector-file` the pipeline falls back to a placeholder
assignment** (alphabetical order modulo 11) and warns at runtime. That
placeholder is what produced the retracted headline described above — the
sector one-hot features become noise and `--edge-mode proxy` degenerates
into a random block graph. Always pass real labels.

The GICS data itself is gitignored: the classifications are proprietary to
S&P Dow Jones Indices / MSCI, so only the builder script is committed.

## One-Sentence Summary

A financial GNN research workspace showing that graph structure over firm
peer groups reliably improves cross-sectional return-rank classification —
and that the improvement does not become portfolio performance, because it
concentrates in the part of the return distribution a long-short book never
trades.
