"""
Market Regime Detection — Configuration
=========================================
Central dataclass encoding every architectural hyperparameter for the
Multi-Relational Dynamic GNN for Market Regime Detection & Early Warning.

Architecture Overview:
    1. Per-snapshot Spatial Encoder  (R-GCN / Relation-specific GAT)
       — processes one day's heterogeneous graph → per-node embeddings
    2. Graph Pooling  (global_mean_pool)
       — collapses N node embeddings → 1 graph-level macro embedding
    3. Temporal Aggregator  (LSTM or Transformer)
       — consumes T=30 daily graph-level embeddings → context vector
    4. Dual Prediction Heads
       a) 4-class regime classifier  (Bull / Crash / Liquidity / Stress)
       b) Binary transition logit    (stress onset in 5–20 days)

Edge types  (3 relation classes):
    0 = correlation     (rolling return correlation — contagion channel)
    1 = etf_cohold      (ETF co-holding overlap — fund-flow channel)
    2 = supply_chain    (production-network proximity — supply-chain shocks)
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RegimeConfig:
    """Central configuration for the Multi-Relational Dynamic GNN."""

    # ── Universe ─────────────────────────────────────────────────────────
    num_stocks: int = 500               # S&P 500 constituents (max)
    num_features: int = 37              # per-stock daily feature dim (same as THGNN)

    # ── Edge Types (3 distinct relation classes) ─────────────────────────
    num_relations: int = 3              # correlation / etf_cohold / supply_chain
    relation_names: Tuple[str, ...] = ("correlation", "etf_cohold", "supply_chain")

    # ── Graph Construction (per relation) ────────────────────────────────
    # Correlation edges: top-k / bottom-k neighbours per node
    corr_top_k: int = 30               # top-30 most correlated
    corr_bot_k: int = 20               # bottom-20 most anti-correlated
    corr_window: int = 30              # 30-day rolling correlation window

    # ETF co-holding edges: connect if ≥ threshold fraction of shared ETFs
    etf_cohold_threshold: float = 0.3  # min Jaccard overlap to form edge

    # Supply-chain edges: from BEA Input-Output or Compustat segments
    supply_chain_hops: int = 1         # direct (1-hop) supplier/customer links

    # Per-edge attribute dimension (edge weight + optional features)
    edge_attr_dim: int = 4             # [weight, |weight|, same_sector, same_subind]

    # ── Node Feature Encoder (per-stock MLP before GNN) ──────────────────
    node_input_dim: int = 37           # raw feature dimension
    node_hidden_dim: int = 128         # projected node dim for GNN input

    # ── Spatial Encoder (R-GCN) ──────────────────────────────────────────
    rgcn_hidden_dim: int = 128         # hidden dimension per R-GCN layer
    rgcn_out_dim: int = 128            # output node embedding dimension
    rgcn_layers: int = 2               # number of R-GCN message-passing layers
    rgcn_num_bases: int = 3            # basis decomposition (None = full per-relation)
    rgcn_dropout: float = 0.2          # dropout within R-GCN layers
    rgcn_activation: str = "elu"       # activation function

    # ── Graph Pooling ────────────────────────────────────────────────────
    pool_method: str = "mean"          # "mean", "max", or "attention"
    graph_embed_dim: int = 128         # dimension of pooled graph-level embedding

    # ── Temporal Aggregator (LSTM) ───────────────────────────────────────
    seq_len: int = 30                  # T = 30 daily graph snapshots
    temporal_type: str = "lstm"        # "lstm" or "transformer"
    lstm_hidden_dim: int = 256         # LSTM hidden state dimension
    lstm_layers: int = 2               # number of stacked LSTM layers
    lstm_dropout: float = 0.2          # inter-layer LSTM dropout
    lstm_bidirectional: bool = False   # unidirectional (causal)

    # Transformer temporal encoder (alternative to LSTM)
    temporal_n_heads: int = 4          # attention heads for temporal transformer
    temporal_ff_dim: int = 512         # FFN inner dim in temporal transformer
    temporal_layers: int = 2           # number of temporal transformer layers
    temporal_dropout: float = 0.2

    # ── Temporal output dimension (auto-derived) ─────────────────────────
    # LSTM: lstm_hidden_dim (last hidden state)
    # Transformer: graph_embed_dim (last time-step embedding)

    # ── Prediction Heads ─────────────────────────────────────────────────
    # Head 1: 4-class regime classifier
    num_regime_classes: int = 4        # Bull=0, Crash=1, Liquidity=2, Stress=3
    regime_names: Tuple[str, ...] = ("Bull", "Crash", "Liquidity", "Stress")
    regime_head_hidden: int = 128      # MLP hidden dim for regime head
    regime_head_dropout: float = 0.3   # dropout in regime classifier

    # Head 2: Binary transition logit (early warning)
    # P(stress onset within 5–20 days)
    transition_horizon_min: int = 5    # earliest stress onset (days)
    transition_horizon_max: int = 20   # latest stress onset (days)
    transition_head_hidden: int = 64   # MLP hidden dim for transition head
    transition_head_dropout: float = 0.3

    # ── Loss ─────────────────────────────────────────────────────────────
    regime_loss_weight: float = 1.0    # weight for regime classification CE loss
    transition_loss_weight: float = 1.0  # weight for transition BCE loss
    focal_gamma: float = 2.0          # focal loss gamma (class imbalance)
    label_smoothing: float = 0.05     # label smoothing for regime CE

    # ── Normalisation ────────────────────────────────────────────────────
    rolling_zscore_window: int = 60    # 60-day rolling z-score for features

    # ── Training ─────────────────────────────────────────────────────────
    batch_size: int = 4                # number of sequences per batch
    grad_accum_steps: int = 4          # effective batch = 4 × 4 = 16
    epochs: int = 100
    lr: float = 1e-3                   # AdamW learning rate
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 1e-4
    lr_min: float = 1e-6               # cosine schedule floor
    grad_clip_norm: float = 1.0        # max gradient norm
    warmup_steps: int = 200            # linear warmup before cosine

    # ── Data Split ───────────────────────────────────────────────────────
    train_start: str = "2006-01-01"
    train_end: str = "2020-12-31"
    val_start: str = "2021-01-01"
    val_end: str = "2023-12-31"

    # ── Misc ─────────────────────────────────────────────────────────────
    seed: int = 42
    weight_init: str = "xavier_uniform"

    @property
    def temporal_out_dim(self) -> int:
        """Output dimension from the temporal aggregator."""
        if self.temporal_type == "lstm":
            return self.lstm_hidden_dim
        else:  # transformer
            return self.graph_embed_dim
