# ECE538 Project Report

## Dynamic Regime GNN for Market Regime Detection and Early Warning

## Abstract

This project studies how to model the stock market as a dynamic heterogeneous graph for the task of market regime detection and stress early warning. Instead of treating stocks as independent time series, the project represents each trading day as a graph whose nodes are stocks and whose edges encode cross-asset relationships such as return correlation, ETF co-holding proxy, and supply-chain proxy. A temporal model is then applied to a rolling sequence of graph snapshots in order to predict both the current market regime and whether a market stress episode is likely to occur in the near future.

The main output of the system is a dual prediction target: a four-class market regime label (`Bull`, `Crash`, `Liquidity`, `Stress`) and a binary transition label indicating whether `Stress` appears within the next 5-20 trading days. A GPU experiment on an NVIDIA H200 node used a validation window containing real `Stress` and positive early-warning events. On this split, the best sparse-correlation run reached validation accuracy `0.6385`, regime macro-F1 `0.4513`, transition ROC-AUC `0.7922`, transition precision `0.5276`, and transition recall `0.4718`. A denser correlation graph under the same protocol performed worse, suggesting that additional correlation edges can add noise rather than signal in this small universe. In addition to implementing the core modeling pipeline, this project improved the engineering quality of the repository by correcting split logic, stabilizing packaging and imports, hardening command-line behavior, and expanding regression-test coverage. The resulting codebase is a more reproducible and maintainable research prototype suitable for course submission and future extension.

## 1. Introduction

Financial markets are highly interconnected. Stocks co-move because of shared macroeconomic exposure, sector structure, institutional ownership, and production-network relationships. This interconnected behavior becomes especially important during systemic events, when correlation rises and diversification weakens. Standard time-series models often do not represent these relationships explicitly.

This project approaches the problem from a graph-learning perspective. The core idea is that the market should be modeled as an evolving relational system rather than as isolated price sequences. Each day is represented as a graph over stocks, and a sequence of such daily graphs is used to predict the current market condition and whether a stress regime is likely to appear soon. This framing is useful because it combines two complementary goals:

- understanding the present state of the market
- detecting early warning signals of future systemic stress

The final system is therefore more similar to a market-monitoring and risk-warning model than to a standard return-prediction model.

### 1.1 Project Contributions

This submission makes three concrete contributions:

- It formulates market regime analysis as a multi-relational dynamic graph-learning problem rather than as a collection of independent stock time series.
- It implements a full end-to-end Dynamic Regime GNN pipeline, from data download and rule-based label construction to graph generation, temporal modeling, and dual-task prediction.
- It improves the repository as software by fixing data-split logic, stabilizing packaging, hardening the CLI, and expanding automated regression coverage.

## 2. Problem Statement

The project solves a supervised learning problem with two outputs.

### 2.1 Regime Classification

For each day, the model predicts:

- `Bull`
- `Crash`
- `Liquidity`
- `Stress`

This four-class label is intended to summarize the current macro market environment.

### 2.2 Stress Transition Prediction

For the same day, the model also predicts a binary label:

- `transition_label = 1` if a `Stress` regime appears in the next 5-20 trading days
- `transition_label = 0` otherwise

This second target is important because it turns the project into an early-warning system rather than only a contemporaneous classifier.

## 3. Data and Feature Engineering

### 3.1 Data Source

The real-data pipeline downloads market data from Yahoo Finance. The current implementation uses:

- a curated 30-stock universe
- `SPY` as the broad equity-market proxy
- `^VIX` as a volatility proxy

The curated stock universe spans 10 GICS sectors. This design keeps the project computationally manageable while preserving cross-sector diversity.

### 3.2 Feature Representation

Each stock is represented by a 37-dimensional daily feature vector. The feature set includes:

- price and volume features
- momentum and reversal features
- technical indicators
- market and factor-style signals
- realized volatility features
- sector and sub-industry identifiers
- stock-to-market correlation features
- cross-sectional market statistics

The purpose of the feature design is to combine local stock information with broader market context. This is important because market regime changes are not purely firm-specific events.

## 4. Label Construction

The project does not rely on an externally labeled regime dataset. Instead, labels are constructed algorithmically from observable market statistics.

### 4.1 Rolling Market Metrics

The labeling engine computes:

- 20-day cumulative SPY return
- 20-day realized volatility
- average cross-sectional stock correlation

Thresholds are computed with expanding windows to reduce look-ahead bias.

### 4.2 Regime Definitions

The four regimes are defined through a priority-ordered rule system:

1. `Stress`
   High volatility and high average cross-sectional correlation.
2. `Crash`
   Strong negative recent return together with elevated volatility.
3. `Bull`
   Positive recent return with relatively lower volatility.
4. `Liquidity`
   Residual state for mixed or moderate conditions.

This construction is transparent and reproducible, although it is still based on design choices rather than external ground-truth annotations.

### 4.3 Early-Warning Label

The transition target is defined by checking whether `Stress` appears in the forward window from day `t+5` through day `t+20`. If so, the current day is assigned `transition_label = 1`. This creates a forward-looking classification target that is better aligned with real risk monitoring.

## 5. Graph Construction

The stock market is represented as a heterogeneous graph with one node type, `stock`, and three edge types.

### 5.1 Nodes

Each node corresponds to a stock on a given day and contains the 37-dimensional feature vector.

### 5.2 Edge Types

The graph contains three relation classes:

1. `correlation`
   Constructed from rolling stock-return correlations using top-K positive and bottom-K negative neighbors per node.
2. `etf_cohold`
   Intended to approximate common institutional or ETF ownership. In the current prototype, it falls back to a sector/sub-industry proxy when true holdings data is unavailable.
3. `supply_chain`
   Intended to model production-network exposure. In the current prototype, it falls back to a sparse synthetic adjacency when external supply-chain data is unavailable.

Among these relations, rolling correlation is the strongest directly data-grounded link. The ETF and supply-chain relations are currently useful approximations but should still be interpreted as proxies.

### 5.3 Temporal Sequences

The model operates on rolling windows of 30 daily graph snapshots. This allows it to capture both:

- cross-sectional structure within a single day
- temporal evolution of market structure across days

## 6. Model Architecture

The main model is the Dynamic Regime GNN defined in the main project branch.

### 6.1 Stage 1: Node Feature Encoder

The raw 37-dimensional stock features are projected into a hidden embedding space by an MLP-based encoder. This creates a learned node representation suitable for graph message passing.

### 6.2 Stage 2: Spatial Encoder

The spatial encoder is an R-GCN-based module that performs relation-aware message passing over the three edge types. This stage allows the model to combine a stock's own feature state with relational information from neighboring stocks under different financial relationship types.

### 6.3 Stage 3: Graph Pooling and Temporal Modeling

After spatial message passing, node embeddings are pooled into one graph-level embedding per day. These daily graph embeddings are then fed into a temporal model. The default setting uses an LSTM, although a Transformer-style temporal encoder is also supported by configuration.

### 6.4 Stage 4: Dual Prediction Heads

The temporal context vector is used by two output heads:

- a regime classifier for the 4-class regime prediction
- a transition head for the binary early-warning prediction

This dual-head design is a natural fit for the project because both tasks depend on shared market-structure information, while still requiring different decision boundaries.

### 6.5 Default Model Instantiation

The default implementation uses a `37 -> 128` node-feature encoder, a 2-layer R-GCN with hidden size `128`, mean pooling to a `128`-dimensional graph embedding, and a 2-layer unidirectional LSTM with hidden size `256`. The regime head uses a hidden layer of size `128`, while the transition head uses a hidden layer of size `64`. In the feasibility run reported later, this configuration produced a model with `1,141,783` trainable parameters.

## 7. Training Objective

The model is trained with a dual-task loss:

- regime classification loss
- transition prediction loss

The total loss is the weighted sum

`L_total = 1.0 * L_regime + 1.0 * L_transition`

in the current implementation.

### 7.1 Regime Loss

The regime head uses focal cross-entropy with label smoothing. Focal loss is appropriate here because the market regime classes are imbalanced, with relatively stable market states generally occurring more frequently than rare stress states.

### 7.2 Transition Loss

The transition head uses binary cross-entropy with logits. This is appropriate because the early-warning target is a binary event indicator.

### 7.3 Optimization

The training code uses:

- AdamW optimization
- gradient accumulation
- gradient clipping
- linear warmup followed by cosine-style decay

These choices make the optimization pipeline more stable and better aligned with modern deep-learning practice.

## 8. Software Implementation and Engineering Work

In addition to the modeling pipeline, a substantial part of this project involved improving the engineering quality of the repository.

### 8.1 Packaging and Import Fixes

The repository originally behaved mostly like a collection of scripts. To make it reproducible and installable, the following changes were completed:

- packaging metadata was added and validated in `pyproject.toml`
- a stable `market_regime_gnn` wrapper package was introduced
- the package now bundles a vendored `market_regime_gnn._legacy` implementation so editable installs and built wheels behave consistently

This matters because a course project should be runnable outside the exact original directory layout.

### 8.2 Data Split and Boundary Fixes

Several correctness issues were fixed:

- training and validation now split correctly around `--train-cutoff`
- validation starts on the next calendar day after the cutoff
- earliest valid supervised samples are preserved instead of being silently dropped
- empty training splits now fail fast with informative messages
- empty validation suffixes are handled intentionally rather than as silent errors

These changes reduce the risk of leakage, silent sample loss, and confusing experimental behavior.

### 8.3 CLI and Runtime Hardening

The real-data CLI was improved so that:

- `--help` works without launching downloads or training
- invalid runtime parameters are rejected before expensive work begins
- device selection is validated for `cpu`, `cuda`, `cuda:N`, and `mps`
- invalid argument errors produce standard argparse messages
- runtime/data failures remain runtime failures instead of being mislabeled as argument errors

### 8.4 Regression Testing

The repository now includes regression tests for:

- import behavior
- CLI help paths
- CLI validation behavior
- device resolution
- split-date construction
- runtime-argument validation
- curated-universe assumptions
- dataset-boundary behavior
- lightweight label-generation smoke checks

These tests improve confidence that future code edits will not silently break the project.

## 9. Reproducibility and Validation

The project now supports a reproducible environment setup using `uv`.

### 9.1 Environment Setup

Recommended setup:

```bash
uv sync --dev
```

This installs dependencies, creates the local environment, and enables package-style entry points.

### 9.2 Validation Commands

The project has been validated with:

```bash
uv run pytest -q
uv build
```

Observed validation status at the time of writing:

- `uv run pytest -q` -> `47 passed`
- `uv build` -> source distribution and wheel built successfully

These results do not constitute a financial-performance benchmark, but they do show that the codebase is installable, testable, and structurally consistent. This distinction is important: software validation establishes reproducibility of the implementation, while the next section evaluates the behavior of the model itself on real data.

## 10. Experimental Results

The following experiments were run on a GPU node rather than as CPU smoke tests. The main goal was to evaluate the early-warning head on a validation window that actually contains `Stress` days and positive `transition_label` examples.

### 10.1 Experiment Setup

The data payload was fetched once for `2018-01-01` through `2025-12-31` and cached locally. Supervised samples were then restricted to the split ranges below. The dataset constructor keeps the full 5-20 day transition horizon inside each split, so target dates near a split boundary are filtered out rather than allowed to leak information across train and validation periods.

| Item | Value |
| --- | --- |
| Train range | `2018-01-01` to `2021-12-31` |
| Validation range | `2022-01-01` to `2024-12-31` |
| Device | NVIDIA H200, `cuda` |
| PyTorch | `2.11.0+cu130` |
| Training epochs | `2` |
| Batch size | `1` |
| Gradient accumulation | `1` |
| Learning rate | `5e-4` with `10` warmup steps |
| Sequence length | `30` daily graph snapshots |
| Main correlation graph | `corr_top_k = 5`, `corr_bot_k = 3` |
| Comparison correlation graph | `corr_top_k = 10`, `corr_bot_k = 5` |
| Valid stocks | `30` |
| Train / validation samples | `900 / 733` |
| Trainable parameters | `1,141,783` |

The main run was executed with the same package entry points and internal training code as the CLI. The resulting experiment artifacts were saved to:

```text
artifacts/gpu_h200_main_split2018to2024_cut2021_k5_3_seq30_e2.json
artifacts/gpu_h200_densecorr_split2018to2024_cut2021_k10_5_seq30_e2.json
```

### 10.2 Label Distribution

The chosen split deliberately includes rare-event labels in both training and validation. This makes the validation metrics more informative than the earlier CPU feasibility window, which contained no validation `Stress` or transition positives.

| Split | Samples | Bull | Crash | Liquidity | Stress | `transition=1` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 900 | 538 | 18 | 227 | 117 | 179 |
| Validation | 733 | 354 | 15 | 255 | 109 | 142 |

The validation set is still imbalanced, but it is now suitable for testing the early-warning objective: `109` validation targets are `Stress`, and `142` validation targets have a future stress event within the 5-20 day warning window.

### 10.3 Main GPU Result

The best checkpoint by validation macro-F1 was epoch 2 of the sparse-correlation run.

| Metric | Value |
| --- | ---: |
| Validation loss | 1.6267 |
| Regime accuracy | 0.6385 |
| Regime macro-F1 | 0.4513 |
| Transition accuracy | 0.8158 |
| Transition precision | 0.5276 |
| Transition recall | 0.4718 |
| Transition ROC-AUC | 0.7922 |
| Training time for 2 epochs | 311.1 s |

The per-epoch training summary was:

```text
Epoch   1/2 | train 1.2249 (reg 0.6419, trans 0.5830) | val 1.6533 (acc 0.617, F1 0.425, AUC 0.828) | lr 2.53e-04 | 156.7s
Epoch   2/2 | train 0.6678 (reg 0.3928, trans 0.2751) | val 1.6267 (acc 0.638, F1 0.451, AUC 0.792) | lr 1.00e-06 | 154.4s
```

Per-class validation accuracy for the best checkpoint was:

| Regime | Accuracy |
| --- | ---: |
| `Bull` | 0.8023 |
| `Crash` | 0.0000 |
| `Liquidity` | 0.5294 |
| `Stress` | 0.4495 |

The validation prediction counts were:

| Regime | Predicted | True |
| --- | ---: | ---: |
| `Bull` | 387 | 354 |
| `Crash` | 0 | 15 |
| `Liquidity` | 267 | 255 |
| `Stress` | 79 | 109 |

For the early-warning head, the model predicted `127` positive warnings against `142` true positives. The average predicted stress-transition probability was `0.1753`, with standard deviation `0.3603` and range `[0.0004, 0.9999]`.

### 10.4 Graph Sparsity Comparison

A second GPU run used a denser correlation graph with `corr_top_k = 10` and `corr_bot_k = 5`, keeping the same data split, model architecture, optimizer, seed, and number of epochs.

| Configuration | Val Acc | Val Macro-F1 | Transition Precision | Transition Recall | Transition ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sparse correlation, `5/3` | 0.6385 | 0.4513 | 0.5276 | 0.4718 | 0.7922 |
| Dense correlation, `10/5` | 0.5975 | 0.4068 | 0.4276 | 0.4366 | 0.7608 |

The denser graph predicted more stress-warning positives (`145` versus `127`) but had lower precision, lower recall, lower regime macro-F1, and lower transition ROC-AUC. In this small 30-stock universe, adding more correlation edges appears to introduce noisy neighbors faster than it adds useful information.

### 10.5 Interpretation

The GPU results change the empirical story of the project. The pipeline is no longer only shown to run on real data; it is also evaluated on a validation period with meaningful stress events. The early-warning head produces non-trivial scores and reaches transition ROC-AUC near `0.79`, which is a useful sign that the dynamic graph representation contains forward-looking stress information.

At the same time, the regime-classification task remains difficult. `Bull` and `Liquidity` are detected more reliably than the rare regimes, while `Crash` is never predicted in this run. This is expected given the small number of `Crash` examples (`18` in training and `15` in validation), but it means the model is not yet a reliable four-regime classifier.

The two-epoch trajectory also shows a task tradeoff. Regime macro-F1 improves from epoch 1 to epoch 2, while transition ROC-AUC decreases from `0.828` to `0.792`. A stronger training protocol should therefore select checkpoints with an explicit multi-task criterion rather than relying only on the final epoch.

### 10.6 Remaining Experimental Work

These GPU runs are a stronger empirical result than the earlier CPU feasibility test, but they are still not a full benchmark study. The most important missing pieces are:

- multiple random seeds with mean and standard deviation reporting
- a held-out test period separate from validation
- non-graph baselines using market-level and pooled stock-level features
- threshold selection for transition precision, recall, and F1 on validation data
- PR-AUC for the imbalanced early-warning task
- broader ablations over sequence length, temporal encoder type, and proxy relation types

A practical next-stage protocol would train the main model and baselines across at least three seeds, choose checkpoints on validation macro-F1 with transition PR-AUC as a secondary criterion, and evaluate the selected checkpoints once on a held-out test window. A rolling-origin evaluation would also be valuable because it would show whether the model remains stable across different macro regimes rather than only on the `2022-2024` validation period.

## 11. Current Project Status

This project should be understood as a research-engineering prototype with a working end-to-end pipeline. The strongest current outcomes are:

- a coherent graph-based formulation of market regime detection
- an explicit early-warning target for future stress events
- a working real-data pipeline using temporal heterogeneous graphs
- GPU validation results on a split with real `Stress` and positive early-warning labels
- a small graph-sparsity comparison showing that denser correlation graphs are not automatically better
- a substantially improved engineering foundation for reproducibility and maintainability

What the project does not yet provide is a full benchmark-style empirical study with a held-out test set, multiple random seeds, extensive baseline comparisons, and broader ablation tables. That is the most important missing component from a pure research-evaluation perspective.

## 12. Limitations

Several limitations remain.

### 12.1 Data Limitations

- Real-data runs depend on Yahoo Finance availability and possible upstream revisions.
- The stock universe is currently a fixed curated 30-stock sample rather than a fully configurable universe.
- ETF co-holding and supply-chain relations are still proxy-based unless external datasets are supplied.

### 12.2 Modeling Limitations

- The regime labels are rule-based, so they reflect the assumptions built into the labeling engine.
- The current implementation focuses more on correctness and reproducibility than on final benchmarked predictive performance.
- The relation design is financially motivated, but not all relations are equally grounded in direct data.

### 12.3 Evaluation Limitations

- The automated tests focus on software correctness and smoke-level behavior.
- The GPU experiments are single-seed pilot runs rather than a full multi-seed benchmark.
- The report includes one graph-sparsity comparison but does not include a full quantitative comparison against standard baselines.

## 13. Future Work

The most valuable next steps are:

1. Execute the full experimental protocol described in Section 10.6 using fixed train/validation/test windows and multiple seeds.
2. Add benchmark baselines such as non-graph temporal classifiers or simpler factor-based models.
3. Replace proxy relation sources with real ETF-holdings and supply-chain datasets.
4. Make the stock universe configurable while preserving a stable default smoke-test setup.
5. Turn the current JSON experiment artifacts into a more systematic experiment-tracking workflow.
6. Expand tests around the full training and evaluation loop of the regime branch.

## 14. Conclusion

This project demonstrates how a dynamic heterogeneous graph neural network can be used to model market structure for regime detection and early warning. The central contribution is the Dynamic Regime GNN pipeline, which combines stock-level features, multi-relational graph construction, temporal modeling, and dual-task prediction. From a software-engineering perspective, the repository has also been improved significantly through packaging fixes, split-logic corrections, CLI hardening, and broader regression testing.

For a course project, this submission offers both a meaningful modeling idea and a working implementation. Its primary strength is the coherent integration of graph structure and temporal context for systemic market analysis, now supported by GPU experiments on a validation window with real stress events. Its primary remaining weakness is the lack of a deeper quantitative benchmark study. Even so, the project provides a strong base for continued experimentation and further research.
