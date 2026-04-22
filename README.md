# GNN Market Regime Detection and Early Warning

This repository contains two closely related research prototypes built around graph neural networks for financial markets:

1. `GNNsMarketRegimeDetection&Early-Warning/`
   Primary project in this repo. It models the equity market as a sequence of heterogeneous graphs and predicts:
   - the current market regime (`Bull`, `Crash`, `Liquidity`, `Stress`)
   - whether a systemic stress regime is likely to appear in the next `5-20` trading days
2. `GNNProject/thgnn/`
   A separate but related prototype focused on forecasting future stock-to-stock correlations with a temporal heterogeneous GNN (THGNN).

If you only want to understand what this repo is "about", start with `GNNsMarketRegimeDetection&Early-Warning/`. The repo name, labeling logic, and end-to-end training script all point to that directory as the main deliverable. The `thgnn` directory is best treated as a reference implementation or adjacent experiment rather than the headline project.

## What Problem This Repo Is Solving

The main idea is to turn the stock market into a dynamic multi-relational graph:

- nodes are stocks
- edges describe relationships between stocks
- graphs evolve daily
- a temporal model consumes a rolling window of graph snapshots

Instead of only predicting prices, the main pipeline predicts market state and a forward-looking warning signal:

- `regime_label`: a 4-way classification of current market conditions
- `transition_label`: a binary early-warning label that becomes positive when a `Stress` regime appears within the next `5-20` trading days

This makes the project closer to a market monitoring / systemic risk detection system than a traditional alpha model.

## Main Project: Dynamic Regime GNN

Directory: [`GNNsMarketRegimeDetection&Early-Warning`](./GNNsMarketRegimeDetection%26Early-Warning)

### Core pipeline

The main pipeline implemented in [`GNNsMarketRegimeDetection&Early-Warning/run_real_data.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNsMarketRegimeDetection&Early-Warning/run_real_data.py) is:

1. Download a curated sample of S&P 500 stocks, plus `SPY` and `^VIX`, from Yahoo Finance.
2. Build a `37`-dimensional daily feature vector per stock.
3. Compute rule-based market regime labels from SPY returns, realized volatility, and average cross-sectional correlation.
4. Build daily heterogeneous graphs with three relation types:
   - `correlation`
   - `etf_cohold`
   - `supply_chain`
5. Slice the data into rolling sequences of `T=30` graph snapshots.
6. Train a dynamic GNN with two prediction heads:
   - regime classification head
   - transition / early-warning head

### Model architecture

The main model is defined in [`models/dynamic_regime_gnn.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNsMarketRegimeDetection&Early-Warning/models/dynamic_regime_gnn.py):

1. `NodeFeatureEncoder`
   Projects raw per-stock features from `37 -> 128`.
2. `SpatialRGCN`
   Applies relation-aware message passing over the three edge types.
3. `Graph pooling`
   Aggregates node embeddings into one graph-level embedding per day.
4. `Temporal encoder`
   Uses either:
   - `LSTM` by default
   - `Transformer` as an alternative
5. Dual heads
   - `RegimeClassifierHead`: 4-class logits
   - `TransitionLogitHead`: binary stress-warning logit

In short:

`30 daily heterogeneous graphs -> spatial GNN -> temporal encoder -> current regime + future stress warning`

### Label generation logic

The statistical labeling engine lives in [`data/label_generator.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNsMarketRegimeDetection&Early-Warning/data/label_generator.py).

It derives four market states using expanding-window thresholds to reduce look-ahead bias:

- `Stress`
  High volatility and high average cross-sectional correlation
- `Crash`
  Large negative recent return plus elevated volatility
- `Bull`
  Positive recent return plus relatively low volatility
- `Liquidity`
  Residual bucket for everything else

The early-warning target is:

- `transition_label = 1` if a `Stress` regime appears anywhere in the next `5-20` trading days

This is important because the repo is not using manually curated crisis dates; it is generating supervision from market statistics.

### Graph construction

Graph building logic lives in [`data/hetero_dataset.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNsMarketRegimeDetection&Early-Warning/data/hetero_dataset.py).

Each daily graph has one node type, `stock`, and three edge types:

- `correlation`
  Built from rolling return correlations using top-K positive and bottom-K negative neighbors per node.
- `etf_cohold`
  Intended to represent ETF co-holding overlap. In the current prototype, when real holdings are not provided, this falls back to a sector/sub-industry similarity proxy.
- `supply_chain`
  Intended to represent production-network links. In the current prototype, when no external adjacency is provided, this falls back to a sparse synthetic adjacency.

This means the repo already encodes the right modeling idea, but two of the three relation channels are still proxy-based unless you plug in real external data.

### Training setup

Training logic lives in [`train.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNsMarketRegimeDetection&Early-Warning/train.py).

Key defaults from [`config.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNsMarketRegimeDetection&Early-Warning/config.py):

- `seq_len = 30`
- `num_features = 37`
- `num_relations = 3`
- `temporal_type = "lstm"`
- `num_regime_classes = 4`
- loss = regime focal cross-entropy + transition BCE-with-logits
- optimizer = `AdamW`
- scheduler = linear warmup + cosine decay

Validation metrics computed by the trainer include:

- regime accuracy
- per-class regime accuracy
- macro-F1
- transition accuracy
- transition precision / recall
- transition ROC-AUC

## Secondary Project: THGNN Correlation Forecasting

Directory: [`GNNProject/thgnn`](./GNNProject/thgnn)

This directory is a separate project focused on forecasting future stock correlation structure rather than regime classification.

### What it does

The THGNN branch:

- builds graphs from rolling pairwise correlations
- encodes per-stock temporal features with a Transformer-based temporal encoder
- applies a relation-aware GAT-style relational encoder
- predicts future Fisher-z correlation residuals and reconstructed correlations

The main files are:

- [`GNNProject/thgnn/models/thgnn.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNProject/thgnn/models/thgnn.py)
- [`GNNProject/thgnn/data/dataset.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNProject/thgnn/data/dataset.py)
- [`GNNProject/thgnn/train.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNProject/thgnn/train.py)
- [`GNNProject/thgnn/run_real_data.py`](/Users/yifanzhang/Library/CloudStorage/OneDrive-Personal/Courses/ECE538/GNN_MarketRegimeDetectionandEarlyWarning/GNNProject/thgnn/run_real_data.py)

### Why it is still relevant here

Although it is not the same task, it clearly feeds the same broader theme:

- temporal structure in stock features
- graph-based cross-sectional modeling
- market structure / dependency learning

The `thgnn` directory is therefore useful as:

- a baseline
- a precursor implementation
- a source of reusable engineering patterns

## Repository Structure

```text
.
├── GNNsMarketRegimeDetection&Early-Warning/
│   ├── config.py
│   ├── run_real_data.py
│   ├── train.py
│   ├── data/
│   │   ├── hetero_dataset.py
│   │   └── label_generator.py
│   ├── models/
│   │   └── dynamic_regime_gnn.py
│   └── tests/
├── GNNProject/
│   └── thgnn/
│       ├── config.py
│       ├── run_real_data.py
│       ├── train.py
│       ├── data/
│       ├── losses/
│       ├── models/
│       └── tests/
└── README.md
```

## Dependencies

There is no pinned environment file in the repo right now, but the code clearly depends on:

- Python `3.10+`
- `torch`
- `torch-geometric`
- `numpy`
- `pandas`
- `yfinance`

Optional but useful:

- `pytest` if you want to adapt the existing test scripts into a standard test workflow

A minimal install path is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy pandas yfinance
pip install torch-geometric
```

Note:

- `torch-geometric` is required by both projects.
- In the current local environment, `torch_geometric` was missing, so the GNN smoke tests did not run successfully until that dependency is installed.

## How To Run

Because the directory name contains `&`, quote the path in shell commands.

### Run the main regime-detection pipeline

```bash
cd "GNNsMarketRegimeDetection&Early-Warning"
python run_real_data.py
```

What this does:

- downloads Yahoo Finance data
- generates regime labels
- builds temporal heterogeneous graphs
- trains the dynamic regime GNN
- prints validation metrics and prediction statistics

### Run the THGNN real-data experiment

```bash
cd "GNNProject/thgnn"
python run_real_data.py
```

### Smoke tests / sanity checks

THGNN branch:

```bash
python GNNProject/thgnn/tests/test_smoke.py
python GNNProject/thgnn/tests/test_e2e_step2.py
python GNNProject/thgnn/tests/test_e2e_step3.py
```

Main regime branch:

```bash
python "GNNsMarketRegimeDetection&Early-Warning/data/hetero_dataset.py"
python "GNNsMarketRegimeDetection&Early-Warning/models/dynamic_regime_gnn.py"
```

## Current Validation Status

What I verified while inspecting the repo:

- the repo is clean and currently has no top-level README before this one
- the main regime-detection branch is the most aligned with the repository name and problem statement
- the THGNN branch has more explicit test scripts
- the market-regime branch currently has almost no dedicated test coverage under `tests/`

What I attempted locally:

- `python GNNProject/thgnn/tests/test_smoke.py`
- `python "GNNsMarketRegimeDetection&Early-Warning/data/hetero_dataset.py"`
- `python "GNNsMarketRegimeDetection&Early-Warning/models/dynamic_regime_gnn.py"`

All three failed for the same reason in the current environment:

- `ModuleNotFoundError: No module named 'torch_geometric'`

So the code layout is coherent, but the environment is not yet fully provisioned for execution.

## Strengths of the Repo

- Clear research direction: systemic market monitoring via dynamic graphs.
- Reasonable decomposition: data, labels, model, training, and run scripts are separated cleanly.
- Good inline documentation inside the Python files.
- The main project combines cross-sectional structure and temporal dynamics in a sensible way.
- The THGNN branch includes useful smoke and integration tests that help explain the intended tensor flow.

## Current Limitations

- No `requirements.txt`, `environment.yml`, or `pyproject.toml`.
- No root README existed before this file.
- The main regime project depends on proxy graph relations for ETF co-holdings and supply-chain links unless external data is provided.
- Real-data scripts fetch from Yahoo Finance directly, so runs are sensitive to network access and upstream data availability.
- The primary market-regime branch has limited automated test coverage compared with the THGNN branch.
- The repo contains generated `__pycache__` files in versioned directories, which usually should not be committed.

## Recommended Next Steps

If you want to turn this from a research prototype into a more usable project, the highest-value next steps are:

1. Add a real dependency file (`requirements.txt` or `pyproject.toml`).
2. Add a root-level quickstart that assumes a clean machine.
3. Add smoke tests for the market-regime branch similar to the THGNN tests.
4. Replace proxy `etf_cohold` and `supply_chain` relations with real external datasets.
5. Save checkpoints, config snapshots, and evaluation outputs to disk instead of only printing to stdout.
6. Add notebooks or reports showing qualitative regime periods and warning examples.

## One-Sentence Summary

This repo is a financial GNN research workspace whose main contribution is a dynamic heterogeneous graph model for classifying market regimes and issuing early warnings for systemic stress, with a second THGNN branch for stock-correlation forecasting.
