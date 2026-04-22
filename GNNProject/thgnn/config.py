"""
THGNN Configuration
===================
All hyperparameters from the paper:
  Fanshawe, Masih & Cameron (2026)
  "Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network"
  arXiv: 2601.04602
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class THGNNConfig:
    """Central configuration mirroring every architectural choice in the paper."""

    # ── Universe ──────────────────────────────────────────────────────────
    num_stocks: int = 500                    # S&P 500 constituents (N ≤ 500)

    # ── Feature space ─────────────────────────────────────────────────────
    num_features: int = 37                   # F = 37  (Section 4.2.1)
    # Feature groups (for reference / future per-group processing):
    #   Price & volume:          PRC, VOL                             (2)
    #   Technicals:              mom5, mom20, mom60, rev5, RSI14,
    #                            ATR14                                (6)
    #   Firm chars:              mktcap, bm                           (2)
    #   Factor betas:            beta_mkt, beta_smb, beta_hml         (3)
    #   Macro & risk:            mkt_rf, smb, hml, rf, umd,
    #                            DCOILWTICO, DGS10, DTWEXBGS,
    #                            VIX, garch_vol                       (10)
    #   Returns:                 excess_ret, raw_ret, spy_ret         (3)
    #   Sector codes:            gsector, gsubind                     (2)
    #   Corr & vol context:      corr_mkt_10, corr_mkt_21,
    #                            corr_mkt_63, corr_sector_21,
    #                            corr_subind_21, rvol_sector_20,
    #                            rvol_subind_20, rvol_mkt_10,
    #                            cross_disp                           (9)
    #                                                         Total: 37

    # ── Temporal Encoder (Transformer) ────────────────────────────────────
    seq_len: int = 30                        # L = 30 trading days lookback
    d_model: int = 128                       # model width after linear proj
    n_heads: int = 8                         # multi-head attention heads
    d_k: int = 16                            # d_model / n_heads = 16
    n_encoder_layers: int = 4                # 4 pre-norm transformer layers
    dim_feedforward: int = 512               # FFN inner dim (standard 4×d_model)
    transformer_dropout: float = 0.2         # dropout in transformer layers
    node_embed_dim: int = 512                # MLP output → GAT input dim
    # flatten dim = seq_len * d_model = 30 * 128 = 3840

    # ── Graph Construction ────────────────────────────────────────────────
    top_k_corr: int = 50                     # top-50 most positive neighbours
    bot_k_corr: int = 50                     # bottom-50 most negative neighbours
    rand_mid_k: int = 75                     # 75 random mid-range [0.2, 0.8] pctile
    num_edge_attr: int = 5                   # ρ_base, |ρ_base|, sign, same_sector,
                                             #   same_subind
    num_edge_types: int = 3                  # neg=0, mid=1, pos=2
    edge_state_dim: int = 64                 # persistent edge state dimension

    # ── Relational Encoder (GAT) ──────────────────────────────────────────
    gat_layers: int = 3                      # L_g = 3 graph attention layers
    gat_heads: int = 4                       # 4 attention heads per GAT layer
    gat_hidden_dim: int = 512                # node state dim throughout GAT
    gat_dropout: float = 0.2

    # ── Prediction Head ───────────────────────────────────────────────────
    num_expert_heads: int = 3                # neg / mid / pos expert MLPs
    expert_hidden_dim: int = 256             # hidden width inside each expert MLP

    # ── Loss ──────────────────────────────────────────────────────────────
    huber_delta: float = 1.0                 # Smooth-L1 / Huber δ
    hist_bins_per_type: int = 6              # 6 bins per edge-type histogram
    hist_bins_global: int = 15               # 15 bins for global histogram
    hist_sigma: float = 0.1                  # Gaussian soft-bin bandwidth
    hist_scale: float = 7.0                  # s = 7 multiplier for L_hist

    # ── Normalisation ─────────────────────────────────────────────────────
    rolling_zscore_window: int = 60          # 60-day rolling z-score for features
    rolling_corr_window: int = 30            # 30-day rolling corr baseline

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 3                      # physical batch size
    grad_accum_steps: int = 6                # effective batch = 3 × 6 = 18
    epochs: int = 75
    lr: float = 3e-4                         # AdamW learning rate
    betas: Tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 2e-4
    lr_min: float = 1e-6                     # cosine schedule floor
    grad_clip_norm: float = 1.0              # max gradient norm

    # ── Data split ────────────────────────────────────────────────────────
    train_start: str = "2006-01-01"
    train_end: str = "2018-12-31"
    val_start: str = "2019-03-26"            # 60 trading days into 2019
    val_end: str = "2024-12-05"

    # ── Forecast horizon ──────────────────────────────────────────────────
    forecast_horizon: int = 10               # 10-day ahead correlation target

    # ── Misc ──────────────────────────────────────────────────────────────
    min_trading_days: int = 30               # need 30 of last 33 days for inclusion
    max_gap_days: int = 33                   # tolerance window for missing days
    weight_init: str = "xavier_uniform"      # Xavier-uniform + zero bias
    seed: int = 42
