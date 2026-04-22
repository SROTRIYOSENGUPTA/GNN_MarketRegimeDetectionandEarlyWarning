"""
Dynamic Regime GNN — Real Market Data Training Run
=====================================================
Fetches S&P 500 stock data from Yahoo Finance, constructs the full
37-feature pipeline, generates regime labels, builds heterogeneous graph
edges, and trains the DynamicRegimeGNN model.

Uses a smaller universe (30 stocks) and fewer epochs (5) for a
feasibility test on real data. Scale up by adjusting parameters.

Pipeline:
─────────
    1. Fetch 30 S&P 500 stocks + SPY + ^VIX from Yahoo Finance (2020–2024)
    2. Construct 37-feature matrix per stock (same as THGNN paper)
    3. Generate regime labels (Bull/Crash/Liquidity/Stress) + transition labels
    4. Build heterogeneous edges:
         - Correlation: rolling 30-day ρ > 0.5
         - ETF co-holding proxy: same GICS sector → connected
         - Supply chain proxy: sparse random adjacency
    5. Instantiate RegimeDataset → DataLoader
    6. Train DynamicRegimeGNN for 5 epochs
    7. Print final metrics (loss, regime accuracy, macro-F1, stress ROC-AUC)

Usage:
    python run_real_data.py
"""

from __future__ import annotations

import math
import time
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import torch

from config import RegimeConfig
from data.hetero_dataset import RegimeDataset, build_regime_dataloader
from data.label_generator import (
    generate_market_labels,
    compute_avg_cross_correlation,
    REGIME_NAMES,
)
from models.dynamic_regime_gnn import DynamicRegimeGNN
from train import Trainer


# ═══════════════════════════════════════════════════════════════════════════
# S&P 500 curated sample — 30 stocks across 11 GICS sectors
# ═══════════════════════════════════════════════════════════════════════════
SP500_SAMPLE = {
    "Energy":                  ["XOM", "CVX", "COP"],
    "Materials":               ["LIN", "APD", "SHW"],
    "Industrials":             ["HON", "CAT", "GE"],
    "Consumer Discretionary":  ["AMZN", "HD", "MCD"],
    "Consumer Staples":        ["PG", "KO", "PEP"],
    "Health Care":             ["JNJ", "UNH", "PFE"],
    "Financials":              ["JPM", "BAC", "GS"],
    "Information Technology":  ["AAPL", "MSFT", "NVDA"],
    "Communication Services":  ["DIS", "NFLX", "CMCSA"],
    "Utilities":               ["NEE", "DUK", "SO"],
}

SECTOR_CODES = {s: i for i, s in enumerate(SP500_SAMPLE.keys())}


# ═══════════════════════════════════════════════════════════════════════════
# Helper: rolling correlation, beta, RSI, ATR
# ═══════════════════════════════════════════════════════════════════════════
def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=max(window // 2, 5)).corr(b)


def _rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = stock_ret.rolling(window, min_periods=30).cov(mkt_ret)
    var = mkt_ret.rolling(window, min_periods=30).var()
    return (cov / var.clip(lower=1e-10)).fillna(0)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-10)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close.clip(lower=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Fetch real market data from Yahoo Finance
# ═══════════════════════════════════════════════════════════════════════════
def fetch_real_data(
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    verbose: bool = True,
):
    """
    Fetch 30 S&P 500 stocks + SPY + ^VIX and build the 37-feature matrix.

    Returns
    -------
    features     : dict[int → np.ndarray (T, 37)]
    dates        : list[str]
    sector_map   : dict[int → int]
    subind_map   : dict[int → int]
    returns_dict : dict[int → np.ndarray (T,)]
    spy_df       : pd.DataFrame with "Close" column (for label generation)
    returns_df   : pd.DataFrame of stock returns (for cross-correlation)
    ticker_list  : list[str] — valid tickers in order
    ticker_sector: dict[str → int]
    """
    import yfinance as yf

    # Build ticker list + maps
    tickers = []
    ticker_sector = {}
    ticker_subind = {}
    for sector_name, syms in SP500_SAMPLE.items():
        sec_code = SECTOR_CODES[sector_name]
        for pos, sym in enumerate(syms):
            tickers.append(sym)
            ticker_sector[sym] = sec_code
            ticker_subind[sym] = sec_code * 3 + pos  # proxy sub-industry

    all_symbols = tickers + ["SPY", "^VIX"]

    if verbose:
        print(f"  Fetching {len(all_symbols)} symbols from Yahoo Finance...")
        print(f"  Date range: {start} to {end}")

    raw = yf.download(all_symbols, start=start, end=end, group_by="ticker",
                       auto_adjust=True, progress=verbose, threads=True)

    # ── Extract SPY and VIX ───────────────────────────────────────────────
    spy_data = raw["SPY"].copy()
    vix_data = raw["^VIX"].copy()

    spy_close = spy_data["Close"].dropna()
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.squeeze()

    spy_ret = spy_close.pct_change().fillna(0)
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.squeeze()

    spy_vol_10 = spy_ret.rolling(10).std().fillna(0)

    # VIX
    vix_close = vix_data["Close"].fillna(method="ffill").fillna(20.0)
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()

    # UMD proxy
    spy_mom12 = spy_close.pct_change(252).fillna(0)
    spy_mom1 = spy_close.pct_change(21).fillna(0)
    umd_proxy = spy_mom12 - spy_mom1

    trading_dates = spy_close.index.sort_values()
    dates_str = [d.strftime("%Y-%m-%d") for d in trading_dates]
    T = len(trading_dates)

    if verbose:
        print(f"  Trading days: {T}")

    # ── Per-stock data extraction ─────────────────────────────────────────
    all_returns_df = pd.DataFrame(index=trading_dates)
    stock_close_dict = {}
    stock_high_dict = {}
    stock_low_dict = {}
    stock_vol_dict = {}

    valid_tickers = []
    for ticker in tickers:
        try:
            tk_data = raw[ticker]
            close = tk_data["Close"].reindex(trading_dates)
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            if close.dropna().shape[0] < T * 0.5:
                if verbose:
                    print(f"    Skipping {ticker}: insufficient data")
                continue
            close = close.fillna(method="ffill").fillna(method="bfill")
            stock_close_dict[ticker] = close

            high = tk_data["High"].reindex(trading_dates)
            low = tk_data["Low"].reindex(trading_dates)
            vol = tk_data["Volume"].reindex(trading_dates)
            if isinstance(high, pd.DataFrame):
                high = high.squeeze()
            if isinstance(low, pd.DataFrame):
                low = low.squeeze()
            if isinstance(vol, pd.DataFrame):
                vol = vol.squeeze()

            high = high.fillna(method="ffill").fillna(close)
            low = low.fillna(method="ffill").fillna(close)
            vol = vol.fillna(0)

            stock_high_dict[ticker] = high
            stock_low_dict[ticker] = low
            stock_vol_dict[ticker] = vol
            all_returns_df[ticker] = close.pct_change().fillna(0)
            valid_tickers.append(ticker)
        except Exception as e:
            if verbose:
                print(f"    Skipping {ticker}: {e}")

    if verbose:
        print(f"  Valid stocks: {len(valid_tickers)}")

    if len(valid_tickers) < 2:
        raise ValueError(
            "Need at least 2 valid stocks to build market-regime features and correlations."
        )

    # ── Sector / sub-industry average returns ─────────────────────────────
    sector_avg_ret = {}
    subind_avg_ret = {}
    for sec_code in range(len(SP500_SAMPLE)):
        sec_tickers = [t for t in valid_tickers if ticker_sector.get(t) == sec_code]
        if sec_tickers:
            sector_avg_ret[sec_code] = all_returns_df[sec_tickers].mean(axis=1)
        else:
            sector_avg_ret[sec_code] = pd.Series(0.0, index=trading_dates)

    for ticker in valid_tickers:
        si = ticker_subind[ticker]
        si_tickers = [t for t in valid_tickers if ticker_subind.get(t) == si]
        if si_tickers:
            subind_avg_ret[si] = all_returns_df[si_tickers].mean(axis=1)
        else:
            subind_avg_ret[si] = pd.Series(0.0, index=trading_dates)

    # Cross-sectional return dispersion
    cross_disp = all_returns_df[valid_tickers].std(axis=1).fillna(0)

    # ── Build 37-feature matrix per stock ─────────────────────────────────
    features = {}
    returns_dict = {}
    sid_sector = {}
    sid_subind = {}

    for sid, ticker in enumerate(valid_tickers):
        close = stock_close_dict[ticker]
        high = stock_high_dict[ticker]
        low = stock_low_dict[ticker]
        vol = stock_vol_dict[ticker]
        ret = all_returns_df[ticker]
        sec_code = ticker_sector[ticker]
        si_code = ticker_subind[ticker]

        f = pd.DataFrame(index=trading_dates)

        # 1-2: Price & Volume (log-scaled)
        f["PRC"] = np.log1p(close.clip(lower=0.01))
        f["VOL"] = np.log1p(vol.clip(lower=0))

        # 3-6: Momentum & Reversal
        f["mom5"] = close.pct_change(5).fillna(0)
        f["mom20"] = close.pct_change(20).fillna(0)
        f["mom60"] = close.pct_change(60).fillna(0)
        f["rev5"] = -close.pct_change(5).fillna(0)

        # 7-8: Technical indicators
        f["RSI14"] = _rsi(close) / 100.0
        f["ATR14"] = _atr(high, low, close)

        # 9-10: Firm characteristics
        f["mktcap"] = np.log1p(close * vol.rolling(20).mean().clip(lower=1))
        f["bm"] = 0.0

        # 11-13: Factor betas
        f["beta_mkt"] = _rolling_beta(ret, spy_ret, 60)
        f["beta_smb"] = 0.0
        f["beta_hml"] = 0.0

        # 14-22: Macro & risk features
        f["mkt_rf"] = spy_ret.values
        f["smb"] = 0.0
        f["hml"] = 0.0
        f["rf"] = 0.0
        f["umd"] = umd_proxy.values
        f["DCOILWTICO"] = 0.0
        f["DGS10"] = 0.0
        f["DTWEXBGS"] = 0.0
        f["VIX"] = vix_close.reindex(trading_dates).fillna(20.0).values / 100.0
        f["garch_vol"] = ret.rolling(20).std().fillna(0).values

        # 23-25: Returns
        f["excess_ret"] = (ret - spy_ret).values
        f["raw_ret"] = ret.values
        f["spy_ret"] = spy_ret.values

        # 26-27: Sector codes (normalised)
        f["gsector"] = sec_code / 10.0
        f["gsubind"] = si_code / 30.0

        # 28-30: Market correlation features
        f["corr_mkt_10"] = _rolling_corr(ret, spy_ret, 10).fillna(0)
        f["corr_mkt_21"] = _rolling_corr(ret, spy_ret, 21).fillna(0)
        f["corr_mkt_63"] = _rolling_corr(ret, spy_ret, 63).fillna(0)

        # 31-32: Sector/sub-industry correlations
        f["corr_sector_21"] = _rolling_corr(ret, sector_avg_ret[sec_code], 21).fillna(0)
        f["corr_subind_21"] = _rolling_corr(ret, subind_avg_ret[si_code], 21).fillna(0)

        # 33-35: Realised volatility features
        sec_rets = all_returns_df[[t for t in valid_tickers if ticker_sector.get(t) == sec_code]]
        si_rets = all_returns_df[[t for t in valid_tickers if ticker_subind.get(t) == si_code]]
        f["rvol_sector_20"] = sec_rets.mean(axis=1).rolling(20).std().fillna(0).values
        f["rvol_subind_20"] = si_rets.mean(axis=1).rolling(20).std().fillna(0).values
        f["rvol_mkt_10"] = spy_vol_10.values

        # 36: Cross-sectional dispersion
        f["cross_disp"] = cross_disp.values

        # Clean up
        f = f.fillna(0).astype(np.float32)
        assert f.shape[1] == 37, f"Expected 37 features, got {f.shape[1]} for {ticker}"

        features[sid] = f.values                              # (T, 37)
        returns_dict[sid] = ret.values.astype(np.float32)     # (T,)
        sid_sector[sid] = sec_code
        sid_subind[sid] = si_code

    # SPY DataFrame for label generation
    spy_df = pd.DataFrame({"Close": spy_close}, index=trading_dates)

    if verbose:
        feature_tensor = np.stack(list(features.values()), axis=0) if features else np.zeros((0, 0, 0))
        n_real = int((feature_tensor != 0).any(axis=(0, 1)).sum()) if feature_tensor.size > 0 else 0
        print(f"\n  Feature pipeline complete:")
        print(f"    Stocks      : {len(features)}")
        print(f"    Days        : {T}")
        print(f"    Features    : 37 ({n_real} non-zero)")
        print(f"    Date range  : {dates_str[0]} → {dates_str[-1]}")

    return (features, dates_str, sid_sector, sid_subind, returns_dict,
            spy_df, all_returns_df[valid_tickers], valid_tickers, ticker_sector)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 74)
    print("  Dynamic Regime GNN — Real Market Data Training")
    print("=" * 74)

    # ── Configuration ─────────────────────────────────────────────────────
    cfg = RegimeConfig()

    # Adjust for smaller real-data feasibility test
    cfg.epochs = 5
    cfg.batch_size = 2
    cfg.grad_accum_steps = 2           # effective batch = 4
    cfg.warmup_steps = 10              # shorter warmup for 5-epoch run
    cfg.lr = 5e-4
    cfg.corr_top_k = 10               # fewer edges for 30-stock universe
    cfg.corr_bot_k = 5
    cfg.seq_len = 30
    cfg.rolling_zscore_window = 60

    N_STOCKS = 30
    START = "2020-01-01"
    END = "2024-12-31"
    TRAIN_CUTOFF = "2023-12-31"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ══════════════════════════════════════════════════════════════════════
    # Step 1: Fetch real market data
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 1: Fetching real market data from Yahoo Finance")
    print(f"{'─' * 74}")

    (features, dates, sector_map, subind_map, returns_dict,
     spy_df, returns_df, valid_tickers, ticker_sector) = fetch_real_data(
        start=START, end=END, verbose=True,
    )

    n_stocks = len(features)
    T = len(dates)

    # ══════════════════════════════════════════════════════════════════════
    # Step 2: Generate regime labels
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 2: Generating regime & transition labels")
    print(f"{'─' * 74}")

    # Compute average cross-sectional correlation (30-day rolling)
    print("  Computing 30-day rolling cross-sectional correlation...")
    avg_corr = compute_avg_cross_correlation(returns_df, window=30)

    # Generate labels
    labels_df = generate_market_labels(spy_df, avg_corr, min_history=60)

    # Convert to numpy arrays aligned with dates
    dates_index = pd.to_datetime(dates)
    labels_reindexed = labels_df.reindex(dates_index)

    regime_labels = labels_reindexed["regime_label"].fillna(2).astype(np.int64).values
    transition_labels = labels_reindexed["transition_label"].fillna(0).astype(np.int64).values

    # Print distribution
    print(f"\n  Label Distribution ({T} days):")
    for rid, rname in REGIME_NAMES.items():
        count = (regime_labels == rid).sum()
        pct = 100 * count / T
        print(f"    Class {rid} ({rname:10s}) : {count:5d} days ({pct:5.1f}%)")

    trans_pos = transition_labels.sum()
    print(f"    Transition=1 (stress ahead) : {trans_pos:5d} days ({100*trans_pos/T:5.1f}%)")
    print(f"    Transition=0 (no stress)    : {T - trans_pos:5d} days ({100*(T-trans_pos)/T:5.1f}%)")

    # ══════════════════════════════════════════════════════════════════════
    # Step 3: Build train/val datasets
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 3: Building train/val datasets")
    print(f"{'─' * 74}")

    train_ds = RegimeDataset(
        features=features,
        dates=dates,
        sector_map=sector_map,
        subind_map=subind_map,
        returns=returns_dict,
        regime_labels=regime_labels,
        transition_labels=transition_labels,
        cfg=cfg,
        date_range=(START, TRAIN_CUTOFF),
    )

    val_ds = RegimeDataset(
        features=features,
        dates=dates,
        sector_map=sector_map,
        subind_map=subind_map,
        returns=returns_dict,
        regime_labels=regime_labels,
        transition_labels=transition_labels,
        cfg=cfg,
        date_range=("2024-01-01", END),
    )

    train_loader = build_regime_dataloader(train_ds, cfg, shuffle=True)
    val_loader = build_regime_dataloader(val_ds, cfg, shuffle=False)

    print(f"  Train samples : {len(train_ds)} days")
    print(f"  Val samples   : {len(val_ds)} days")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 4: Inspect a sample batch
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 4: Inspecting a sample batch")
    print(f"{'─' * 74}")

    sample_batch = next(iter(train_loader))
    B = len(sample_batch["snapshots"][0])
    T_seq = len(sample_batch["snapshots"])
    snap0 = sample_batch["snapshots"][0][0]

    print(f"  Batch structure:")
    print(f"    Timesteps (T)  : {T_seq}")
    print(f"    Batch size (B) : {B}")
    print(f"    Snapshot[0][0] stock.x : {tuple(snap0['stock'].x.shape)}")

    # Edge counts from first snapshot
    for et_name in [("stock", "correlation", "stock"),
                    ("stock", "etf_cohold", "stock"),
                    ("stock", "supply_chain", "stock")]:
        try:
            n_edges = snap0[et_name].edge_index.shape[1]
            print(f"    {et_name[1]:15s} edges : {n_edges}")
        except Exception:
            print(f"    {et_name[1]:15s} edges : N/A")

    print(f"    regime_label    : {sample_batch['regime_label']}")
    print(f"    transition_label: {sample_batch['transition_label']}")

    # ══════════════════════════════════════════════════════════════════════
    # Step 5: Initialise model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 5: Initialising DynamicRegimeGNN")
    print(f"{'─' * 74}")

    model = DynamicRegimeGNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())

    groups = model.parameter_groups()
    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_nodecay = sum(p.numel() for p in groups[1]["params"])

    print(f"  Architecture       : {cfg.temporal_type.upper()} temporal encoder")
    print(f"  Total parameters   : {n_params:,}")
    print(f"  With weight decay  : {n_decay:>10,}")
    print(f"  Without decay      : {n_nodecay:>10,} (bias + LayerNorm)")
    print(f"  R-GCN layers       : {cfg.rgcn_layers} × {cfg.rgcn_hidden_dim}d")
    print(f"  LSTM               : {cfg.lstm_layers} layers × {cfg.lstm_hidden_dim}d")
    print(f"  Regime head        : {cfg.num_regime_classes} classes")
    print(f"  Transition head    : binary (stress early warning)")

    # ══════════════════════════════════════════════════════════════════════
    # Step 6: Forward pass sanity check
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 6: Forward pass sanity check")
    print(f"{'─' * 74}")

    with torch.no_grad():
        regime_logits, trans_logit = model(sample_batch["snapshots"])

    print(f"  Regime logits  : {tuple(regime_logits.shape)}")
    print(f"  Transition logit: {tuple(trans_logit.shape)}")
    print(f"  Regime range   : [{regime_logits.min():.4f}, {regime_logits.max():.4f}]")
    print(f"  Trans range    : [{trans_logit.min():.4f}, {trans_logit.max():.4f}]")
    assert torch.isfinite(regime_logits).all(), "Non-finite regime logits!"
    assert torch.isfinite(trans_logit).all(), "Non-finite transition logit!"
    print(f"  ✓ All predictions finite")

    # ══════════════════════════════════════════════════════════════════════
    # Step 7: Training
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print(f"  Step 7: Training ({cfg.epochs} epochs)")
    print(f"{'─' * 74}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if len(val_ds) > 0 else None,
        cfg=cfg,
        device="cpu",
    )

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════════
    # Step 8: Training results
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 8: Training Results Summary")
    print(f"{'─' * 74}")

    print(f"  Total training time : {elapsed:.1f}s")
    print(f"  Optimizer steps     : {trainer.global_step}")
    print(f"  Final LR            : {trainer.optimizer.param_groups[0]['lr']:.2e}")

    print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Reg Loss':>10} | {'Trans Loss':>10}"
          f" | {'Val Loss':>10} | {'Val Acc':>8} | {'Val F1':>8} | {'Val AUC':>8}")
    print(f"  {'─' * 85}")

    for i in range(len(history["train"])):
        h = history["train"][i]
        line = (f"  {i+1:5d} | {h['loss_total']:10.5f} | {h['loss_regime']:10.5f} "
                f"| {h['loss_transition']:10.5f}")

        if i < len(history["val"]) and history["val"][i]:
            v = history["val"][i]
            line += (f" | {v['loss_total']:10.5f} | {v['regime_accuracy']:8.4f} "
                     f"| {v['regime_macro_f1']:8.4f} | {v['transition_roc_auc']:8.4f}")
        else:
            line += f" | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}"
        print(line)

    # Training dynamics
    l0 = history["train"][0]["loss_total"]
    lf = history["train"][-1]["loss_total"]
    print(f"\n  Train loss: {l0:.5f} → {lf:.5f}  "
          f"({'↓ decreased' if lf < l0 else '~ flat/increased'})")

    if history["val"] and history["val"][-1]:
        v_final = history["val"][-1]
        print(f"\n  Final Validation Metrics:")
        print(f"    Total loss         : {v_final['loss_total']:.5f}")
        print(f"    Regime accuracy    : {v_final['regime_accuracy']:.4f}")
        print(f"    Regime macro-F1    : {v_final['regime_macro_f1']:.4f}")
        print(f"    Stress ROC-AUC     : {v_final['transition_roc_auc']:.4f}")
        print(f"    Trans precision    : {v_final['transition_precision']:.4f}")
        print(f"    Trans recall       : {v_final['transition_recall']:.4f}")

        # Per-class accuracy
        pca = v_final["regime_per_class_acc"]
        print(f"\n  Per-Class Validation Accuracy:")
        for c in range(cfg.num_regime_classes):
            name = cfg.regime_names[c]
            acc = pca[c]
            if not np.isnan(acc):
                print(f"    {name:12s} : {acc:.4f}")
            else:
                print(f"    {name:12s} : N/A (no samples)")

    # ══════════════════════════════════════════════════════════════════════
    # Step 9: Final inference on validation set
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 74}")
    print("  Step 9: Inference statistics on validation data")
    print(f"{'─' * 74}")

    model.eval()
    all_regime_preds = []
    all_trans_probs = []
    all_regime_labels_val = []
    all_trans_labels_val = []

    with torch.no_grad():
        for batch in val_loader:
            regime_logits, trans_logit = model(batch["snapshots"])
            regime_pred = regime_logits.argmax(dim=-1)
            trans_prob = torch.sigmoid(trans_logit)

            all_regime_preds.append(regime_pred)
            all_trans_probs.append(trans_prob)
            all_regime_labels_val.append(batch["regime_label"])
            all_trans_labels_val.append(batch["transition_label"])

    if all_regime_preds:
        regime_preds = torch.cat(all_regime_preds)
        trans_probs = torch.cat(all_trans_probs)
        regime_true = torch.cat(all_regime_labels_val)
        trans_true = torch.cat(all_trans_labels_val)

        print(f"  Total predictions: {len(regime_preds)}")

        # Predicted regime distribution
        print(f"\n  Predicted Regime Distribution:")
        for c in range(cfg.num_regime_classes):
            count = (regime_preds == c).sum().item()
            true_count = (regime_true == c).sum().item()
            print(f"    {cfg.regime_names[c]:12s} : pred={count:4d}  true={true_count:4d}")

        # Transition probability statistics
        print(f"\n  Transition Probabilities:")
        print(f"    Mean P(stress)  : {trans_probs.mean():.4f}")
        print(f"    Std P(stress)   : {trans_probs.std():.4f}")
        print(f"    Range           : [{trans_probs.min():.4f}, {trans_probs.max():.4f}]")

        # Transition predictions
        trans_preds = (trans_probs >= 0.5).long()
        print(f"    Predicted pos   : {trans_preds.sum().item()}")
        print(f"    Actual pos      : {trans_true.sum().item()}")
    else:
        print("  No validation data available for inference.")

    print(f"\n{'=' * 74}")
    print(f"  REAL DATA TRAINING COMPLETE ✓")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
