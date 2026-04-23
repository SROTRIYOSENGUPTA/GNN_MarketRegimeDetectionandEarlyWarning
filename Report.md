# ECE538 Project Report

## Dynamic Regime GNN for Market Regime Detection and Early Warning

## Abstract

This project studies how to model the stock market as a dynamic heterogeneous graph for the task of market regime detection and stress early warning. Instead of treating stocks as independent time series, the project represents each trading day as a graph whose nodes are stocks and whose edges encode cross-asset relationships such as return correlation, ETF co-holding proxy, and supply-chain proxy. A temporal model is then applied to a rolling sequence of graph snapshots in order to predict both the current market regime and whether a market stress episode is likely to occur in the near future.

The main output of the system is a dual prediction target: a four-class market regime label (`Bull`, `Crash`, `Liquidity`, `Stress`) and a binary transition label indicating whether `Stress` appears within the next 5-20 trading days. A preliminary real-data feasibility run on a curated 30-stock universe reached validation accuracy `0.8942` and macro-F1 `0.8388` for regime classification, showing that the end-to-end pipeline is operational on real market data. However, the same validation split contained no positive `Stress` or transition events, so the early-warning task is not yet meaningfully benchmarked. In addition to implementing the core modeling pipeline, this project improved the engineering quality of the repository by correcting split logic, stabilizing packaging and imports, hardening command-line behavior, and expanding regression-test coverage. The resulting codebase is a more reproducible and maintainable research prototype suitable for course submission and future extension.

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

- `uv run pytest -q` -> `44 passed`
- `uv build` -> source distribution and wheel built successfully

These results do not constitute a financial-performance benchmark, but they do show that the codebase is installable, testable, and structurally consistent. This distinction is important: software validation establishes reproducibility of the implementation, while the next section evaluates the behavior of the model itself on real data.

## 10. Experimental Results

The following results come from a real-data feasibility run of the Dynamic Regime GNN. They should be interpreted as preliminary evidence that the end-to-end pipeline trains successfully on real market data, not as a final benchmark study.

### 10.1 Experiment Setup

Table 1 summarizes the main protocol choices for the reported run.

| Item | Value |
| --- | --- |
| Date range | `2022-01-01` to `2024-06-30` |
| Training cutoff | `2023-12-31` |
| Training epochs | `1` |
| Device | `cpu` |
| Batch size | `1` |
| Gradient accumulation | `1` |
| Sequence length | `30` daily graph snapshots |
| Correlation graph settings | `corr_top_k = 5`, `corr_bot_k = 3` |
| Valid stocks | `30` |
| Trading days | `625` |
| Train / validation samples | `393 / 104` |
| Trainable parameters | `1,141,783` |

The command used for this run was:

```bash
uv run python -m market_regime_gnn.run_real_data \
  --start 2022-01-01 \
  --end 2024-06-30 \
  --train-cutoff 2023-12-31 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --corr-top-k 5 \
  --corr-bot-k 3 \
  --device cpu
```

### 10.2 Dataset Label Distribution

Across the full 625-day dataset, the rule-based labels were distributed as follows:

| Label | Count | Share |
| --- | ---: | ---: |
| `Bull` | 306 | 49.0% |
| `Crash` | 7 | 1.1% |
| `Liquidity` | 279 | 44.6% |
| `Stress` | 33 | 5.3% |
| `transition_label = 1` | 86 | 13.8% |
| `transition_label = 0` | 539 | 86.2% |

This distribution confirms that the problem is class-imbalanced, especially for the rare `Crash` and `Stress` states.

### 10.3 Validation Results

After one epoch, the model produced the following validation metrics:

| Metric | Value |
| --- | ---: |
| Validation loss | 0.22498 |
| Regime accuracy | 0.8942 |
| Regime macro-F1 | 0.8388 |
| Stress ROC-AUC | `nan` |
| Transition precision | 0.0000 |
| Transition recall | 0.0000 |
| Training time | 208.6 s |

The epoch summary reported by the training loop was:

```text
Epoch   1/1 | train 1.0524 (reg 0.7866, trans 0.2658) | val 0.2250 (acc 0.894, F1 0.839, AUC nan) | lr 1.00e-06 | 208.6s
```

Per-class validation accuracy was:

| Regime | Accuracy |
| --- | ---: |
| `Bull` | 0.9059 |
| `Crash` | N/A |
| `Liquidity` | 0.8421 |
| `Stress` | N/A |

The validation prediction counts were:

| Regime | Predicted | True |
| --- | ---: | ---: |
| `Bull` | 80 | 85 |
| `Crash` | 0 | 0 |
| `Liquidity` | 24 | 19 |
| `Stress` | 0 | 0 |

For the early-warning head, the average predicted stress-transition probability on the validation split was `0.0047`, with standard deviation `0.0264` and range `[0.0000, 0.2069]`.

### 10.4 Interpretation

These results show that the Dynamic Regime GNN can train end-to-end on real data and can separate the dominant validation regimes reasonably well in this short run, especially between `Bull` and `Liquidity`. However, this experiment is not sufficient to validate the early-warning objective. The validation split for this particular window contains no positive `Stress` examples and no positive transition events, which makes `Stress` ROC-AUC undefined and leaves transition precision/recall uninformative.

Therefore, the current experiment should be read as a feasibility result rather than as a conclusive empirical evaluation. A stronger study would require longer or better-balanced out-of-sample windows, repeated runs, and comparisons against non-graph baselines.

### 10.5 Key Takeaways

- The regime-classification branch is already capable of learning useful decision boundaries on real data for the dominant market states in the chosen validation window.
- The early-warning branch remains insufficiently evaluated because the validation split does not contain the rare events that the branch is meant to detect.
- The current codebase is therefore strongest as a validated research prototype: the pipeline runs correctly, but the empirical study is still incomplete.

### 10.6 Proposed Full Experimental Protocol

To move from a feasibility result to a course-quality empirical study, the project should adopt a fixed experimental protocol rather than a single short run. The key requirement is that both validation and test windows must contain non-trivial numbers of `Stress` days and positive transition events. Without this condition, the early-warning task cannot be evaluated meaningfully.

The recommended primary split is:

| Split | Date Range | Purpose |
| --- | --- | --- |
| Train | `2012-01-01` to `2018-12-31` | Fit model parameters |
| Validation | `2019-01-01` to `2021-12-31` | Hyperparameter selection and threshold tuning |
| Test | `2022-01-01` to `2024-06-30` | Final out-of-sample reporting |

Before freezing this split, the label generator should be run once to verify that the validation and test periods both contain at least one `Stress` episode and a meaningful number of positive `transition_label = 1` samples. If one split is too sparse, the window boundaries should be adjusted before training begins, not after seeing model performance.

The recommended training-and-selection procedure is:

- train each configuration for a full schedule such as `20-50` epochs instead of a 1-epoch smoke test
- use `3` random seeds, for example `42`, `52`, and `62`
- select checkpoints on the validation set using regime macro-F1 as the primary score
- use transition PR-AUC as a secondary tie-breaker because the early-warning task is imbalanced
- evaluate the chosen checkpoint exactly once on the held-out test set
- report mean and standard deviation across seeds

The core metrics should be reported separately for the two tasks.

For regime classification:

- overall accuracy
- macro-F1
- balanced accuracy
- per-class precision, recall, and F1
- confusion matrix

For early warning:

- ROC-AUC
- PR-AUC
- precision, recall, and F1 at a threshold selected on validation data
- number of predicted positive warnings
- event-level lead time before the first day of each realized `Stress` episode

The report should also include a small but defensible baseline suite:

- majority-class regime predictor with always-negative transition prediction
- non-graph temporal baseline using only market-level features such as `SPY`, `VIX`, realized volatility, and average correlation
- stock-feature temporal baseline that pools stock features over names but removes graph edges
- correlation-only Dynamic Regime GNN, using the same temporal head but removing proxy relation types

In addition to baselines, the main model should be stress-tested with a focused ablation study:

- remove `etf_cohold` edges
- remove `supply_chain` edges
- compare `LSTM` versus `Transformer` temporal encoders
- compare sequence lengths such as `10`, `20`, and `30`
- compare graph sparsity settings by varying `corr_top_k` and `corr_bot_k`

Finally, the evaluation should include one robustness check beyond the single frozen split. A simple and appropriate choice is a rolling-origin evaluation with three folds, where each fold trains on all data up to a cutoff, validates on the following year, and tests on the year after that. This would show whether the model remains effective across different macro periods rather than only in one selected window.

Under this protocol, a practical course-project run matrix would be:

- 1 main model configuration x 3 seeds
- 4 baselines x 3 seeds
- 5 ablations x 3 seeds

This totals `30` full training runs, which is large enough to support a meaningful empirical section while still remaining manageable on a modest compute budget if the curated 30-stock universe is retained.

## 11. Current Project Status

This project should be understood as a research-engineering prototype with a working end-to-end pipeline. The strongest current outcomes are:

- a coherent graph-based formulation of market regime detection
- an explicit early-warning target for future stress events
- a working real-data pipeline using temporal heterogeneous graphs
- a substantially improved engineering foundation for reproducibility and maintainability

What the project does not yet provide is a full benchmark-style empirical study with fixed frozen datasets, extensive baseline comparisons, and ablation tables. That is the most important missing component from a pure research-evaluation perspective.

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
- Long real-data experiments are not yet deeply automated.
- The report does not include a full quantitative comparison against standard baselines.

## 13. Future Work

The most valuable next steps are:

1. Execute the full experimental protocol described in Section 10.6 using fixed train/validation/test windows and multiple seeds.
2. Add benchmark baselines such as non-graph temporal classifiers or simpler factor-based models.
3. Replace proxy relation sources with real ETF-holdings and supply-chain datasets.
4. Make the stock universe configurable while preserving a stable default smoke-test setup.
5. Add experiment tracking, saved configurations, and artifact outputs for real-data runs.
6. Expand tests around the full training and evaluation loop of the regime branch.

## 14. Conclusion

This project demonstrates how a dynamic heterogeneous graph neural network can be used to model market structure for regime detection and early warning. The central contribution is the Dynamic Regime GNN pipeline, which combines stock-level features, multi-relational graph construction, temporal modeling, and dual-task prediction. From a software-engineering perspective, the repository has also been improved significantly through packaging fixes, split-logic corrections, CLI hardening, and broader regression testing.

For a course project, this submission offers both a meaningful modeling idea and a working implementation. Its primary strength is the coherent integration of graph structure and temporal context for systemic market analysis. Its primary remaining weakness is the lack of a deeper quantitative benchmark study. Even so, the project provides a strong base for continued experimentation and further research.
