"""
Real Market Data Pipeline for THGNN
=====================================
Fetches S&P 500 stock data from Yahoo Finance and constructs the 37-feature
matrix required by the THGNN model.

Feature groups (37 total, matching paper Section 4.2.1):
─────────────────────────────────────────────────────────
 1.  PRC         — adjusted close price (log-scaled)
 2.  VOL         — trading volume (log-scaled)
 3.  mom5        — 5-day momentum (return)
 4.  mom20       — 20-day momentum
 5.  mom60       — 60-day momentum
 6.  rev5        — 5-day reversal (negative short-term return)
 7.  RSI14       — 14-day RSI
 8.  ATR14       — 14-day Average True Range / close
 9.  mktcap      — proxy: log(price × avg_volume) [approx.]
10.  bm          — placeholder (book-to-market, set to 0)
11.  beta_mkt    — 60-day rolling beta to SPY
12.  beta_smb    — placeholder (set to 0, needs Fama-French)
13.  beta_hml    — placeholder (set to 0, needs Fama-French)
14.  mkt_rf      — SPY daily return (market excess)
15.  smb         — placeholder (0)
16.  hml         — placeholder (0)
17.  rf          — placeholder (0, would need FRED)
18.  umd         — SPY 12-month minus 1-month return (momentum factor proxy)
19.  DCOILWTICO  — placeholder (0, oil price)
20.  DGS10       — placeholder (0, 10-yr treasury)
21.  DTWEXBGS    — placeholder (0, USD index)
22.  VIX         — ^VIX close
23.  garch_vol   — 20-day realized volatility
24.  excess_ret  — daily return - SPY return
25.  raw_ret     — daily return
26.  spy_ret     — SPY daily return
27.  gsector     — GICS sector code (integer, 0-indexed)
28.  gsubind     — GICS sub-industry code (integer, 0-indexed)
29.  corr_mkt_10 — 10-day rolling correlation with SPY
30.  corr_mkt_21 — 21-day rolling correlation with SPY
31.  corr_mkt_63 — 63-day rolling correlation with SPY
32.  corr_sector_21 — 21-day rolling corr with sector average
33.  corr_subind_21 — 21-day rolling corr with sub-industry average
34.  rvol_sector_20 — sector realised vol (20-day)
35.  rvol_subind_20 — sub-industry realised vol (20-day)
36.  rvol_mkt_10    — market-wide realised vol (10-day)
37.  cross_disp     — cross-sectional return dispersion

Where data sources are unavailable (Fama-French, FRED), features are
set to 0 and will be z-scored to 0 — the model can still learn from the
~25+ real features.

Usage:
    from data.real_data import fetch_real_data
    features, dates, sector_map, subind_map, returns = fetch_real_data(
        n_stocks=50, start="2020-01-01", end="2024-12-31"
    )
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ─── S&P 500 tickers by GICS sector ─────────────────────────────────────────
# Curated subset: 5 stocks per sector × 11 sectors = 55 tickers
# This gives diversity without overwhelming Yahoo Finance rate limits.
SP500_SAMPLE: Dict[str, List[str]] = {
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "NEM"],
    "Industrials": ["HON", "UNP", "CAT", "GE", "RTX"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT"],
    "Health Care": ["JNJ", "UNH", "PFE", "ABT", "TMO"],
    "Financials": ["JPM", "BAC", "GS", "MS", "BLK"],
    "Information Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "META"],
    "Communication Services": ["DIS", "NFLX", "CMCSA", "VZ", "T"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
}

SECTOR_CODES = {s: i for i, s in enumerate(SP500_SAMPLE.keys())}


def _get_tickers(n_stocks: int = 55) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """
    Return (tickers, sector_map, subind_map).
    sector_map: ticker → sector_code (0..10)
    subind_map: ticker → sub-industry proxy (sector_code * 5 + position)
    """
    tickers, sector_map, subind_map = [], {}, {}
    count = 0
    for sector_name, syms in SP500_SAMPLE.items():
        sec_code = SECTOR_CODES[sector_name]
        for pos, sym in enumerate(syms):
            if count >= n_stocks:
                break
            tickers.append(sym)
            sector_map[sym] = sec_code
            subind_map[sym] = sec_code * 5 + pos  # proxy sub-industry
            count += 1
        if count >= n_stocks:
            break
    return tickers, sector_map, subind_map


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation between two series."""
    return a.rolling(window, min_periods=max(window // 2, 5)).corr(b)


def _rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = 60) -> pd.Series:
    """Rolling OLS beta: stock_ret ~ mkt_ret."""
    cov = stock_ret.rolling(window, min_periods=30).cov(mkt_ret)
    var = mkt_ret.rolling(window, min_periods=30).var()
    return (cov / var.clip(lower=1e-10)).fillna(0)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-10)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range normalised by close."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean() / close.clip(lower=1e-6)


def fetch_real_data(
    n_stocks: int = 50,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    verbose: bool = True,
) -> Tuple[
    Dict[int, np.ndarray],   # features: sid → (T, 37)
    List[str],                # dates
    Dict[int, int],           # sector_map: sid → code
    Dict[int, int],           # subind_map: sid → code
    Dict[int, np.ndarray],   # returns: sid → (T,)
]:
    """
    Fetch real market data and construct the 37-feature matrix.

    Parameters
    ----------
    n_stocks : Number of stocks to fetch (max 55 from curated list).
    start    : Start date (YYYY-MM-DD).
    end      : End date (YYYY-MM-DD).
    verbose  : Print progress.

    Returns
    -------
    features, dates, sector_map, subind_map, returns
    — formatted for THGNNDataset constructor.
    """
    import yfinance as yf

    tickers, ticker_sector, ticker_subind = _get_tickers(n_stocks)

    if verbose:
        print(f"Fetching {len(tickers)} stocks + SPY + ^VIX from Yahoo Finance...")
        print(f"Date range: {start} to {end}")

    # ── Download all data at once ────────────────────────────────────────
    all_symbols = tickers + ["SPY", "^VIX"]
    raw = yf.download(all_symbols, start=start, end=end, group_by="ticker",
                       auto_adjust=True, progress=verbose, threads=True)

    # Extract SPY and VIX
    if len(all_symbols) > 1:
        spy_data = raw["SPY"].copy()
        vix_data = raw["^VIX"].copy()
    else:
        spy_data = raw.copy()
        vix_data = pd.DataFrame()

    spy_close = spy_data["Close"].dropna()
    spy_ret = spy_close.pct_change().fillna(0)
    spy_vol_10 = spy_ret.rolling(10).std().fillna(0)

    # VIX close
    vix_close = vix_data["Close"].fillna(method="ffill").fillna(20.0) if len(vix_data) > 0 else pd.Series(20.0, index=spy_close.index)

    # UMD factor proxy: SPY 12-month minus 1-month return
    spy_mom12 = spy_close.pct_change(252).fillna(0)
    spy_mom1 = spy_close.pct_change(21).fillna(0)
    umd_proxy = spy_mom12 - spy_mom1

    # Trading dates = SPY's valid dates
    trading_dates = spy_close.index.sort_values()
    dates_str = [d.strftime("%Y-%m-%d") for d in trading_dates]
    T = len(trading_dates)

    if verbose:
        print(f"Trading days: {T}")

    # ── Build per-stock returns DataFrame for sector/subind averages ─────
    all_returns_df = pd.DataFrame(index=trading_dates)
    stock_close_dict = {}
    stock_high_dict = {}
    stock_low_dict = {}
    stock_vol_dict = {}

    valid_tickers = []
    for ticker in tickers:
        try:
            if len(all_symbols) > 1:
                tk_data = raw[ticker]
            else:
                tk_data = raw
            close = tk_data["Close"].reindex(trading_dates)
            if close.dropna().shape[0] < T * 0.5:
                if verbose:
                    print(f"  Skipping {ticker}: insufficient data ({close.dropna().shape[0]}/{T})")
                continue
            close = close.fillna(method="ffill").fillna(method="bfill")
            stock_close_dict[ticker] = close
            stock_high_dict[ticker] = tk_data["High"].reindex(trading_dates).fillna(method="ffill").fillna(close)
            stock_low_dict[ticker] = tk_data["Low"].reindex(trading_dates).fillna(method="ffill").fillna(close)
            stock_vol_dict[ticker] = tk_data["Volume"].reindex(trading_dates).fillna(0)
            all_returns_df[ticker] = close.pct_change().fillna(0)
            valid_tickers.append(ticker)
        except Exception as e:
            if verbose:
                print(f"  Skipping {ticker}: {e}")

    if verbose:
        print(f"Valid stocks: {len(valid_tickers)}")

    # ── Sector/sub-industry average returns ──────────────────────────────
    sector_avg_ret = {}
    subind_avg_ret = {}
    for sec_code in range(11):
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

    # ── Build feature matrices ───────────────────────────────────────────
    features: Dict[int, np.ndarray] = {}
    returns_dict: Dict[int, np.ndarray] = {}
    sid_sector: Dict[int, int] = {}
    sid_subind: Dict[int, int] = {}

    for sid, ticker in enumerate(valid_tickers):
        close = stock_close_dict[ticker]
        high = stock_high_dict[ticker]
        low = stock_low_dict[ticker]
        vol = stock_vol_dict[ticker]
        ret = all_returns_df[ticker]
        sec_code = ticker_sector[ticker]
        si_code = ticker_subind[ticker]

        # Feature columns (37):
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
        f["RSI14"] = _rsi(close) / 100.0  # normalize to [0,1]
        f["ATR14"] = _atr(high, low, close)

        # 9-10: Firm characteristics
        f["mktcap"] = np.log1p(close * vol.rolling(20).mean().clip(lower=1))
        f["bm"] = 0.0  # placeholder

        # 11-13: Factor betas
        f["beta_mkt"] = _rolling_beta(ret, spy_ret, 60)
        f["beta_smb"] = 0.0  # needs Fama-French
        f["beta_hml"] = 0.0  # needs Fama-French

        # 14-22: Macro & risk features
        f["mkt_rf"] = spy_ret.values
        f["smb"] = 0.0
        f["hml"] = 0.0
        f["rf"] = 0.0
        f["umd"] = umd_proxy.values
        f["DCOILWTICO"] = 0.0  # needs FRED
        f["DGS10"] = 0.0       # needs FRED
        f["DTWEXBGS"] = 0.0    # needs FRED
        f["VIX"] = vix_close.reindex(trading_dates).fillna(20.0).values / 100.0
        f["garch_vol"] = ret.rolling(20).std().fillna(0).values

        # 23-25: Returns
        f["excess_ret"] = (ret - spy_ret).values
        f["raw_ret"] = ret.values
        f["spy_ret"] = spy_ret.values

        # 26-27: Sector codes (normalised to ~0-1 range for z-scoring)
        f["gsector"] = sec_code / 10.0
        f["gsubind"] = si_code / 50.0

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

        # Fill any remaining NaN
        f = f.fillna(0).astype(np.float32)

        assert f.shape[1] == 37, f"Expected 37 features, got {f.shape[1]} for {ticker}"

        features[sid] = f.values                          # (T, 37)
        returns_dict[sid] = ret.values.astype(np.float32)  # (T,)
        sid_sector[sid] = sec_code
        sid_subind[sid] = si_code

    if verbose:
        print(f"\nData pipeline complete:")
        print(f"  Stocks     : {len(features)}")
        print(f"  Days       : {T}")
        print(f"  Features   : 37 ({sum(1 for c in f.columns if (f[c] != 0).any())} non-zero)")
        print(f"  Date range : {dates_str[0]} → {dates_str[-1]}")

    return features, dates_str, sid_sector, sid_subind, returns_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Quick test
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    features, dates, sector_map, subind_map, returns = fetch_real_data(
        n_stocks=20, start="2022-01-01", end="2024-06-30", verbose=True,
    )
    print(f"\nSample feature matrix shape: {features[0].shape}")
    print(f"Sample returns shape:        {returns[0].shape}")
    print(f"Sector map: {sector_map}")
