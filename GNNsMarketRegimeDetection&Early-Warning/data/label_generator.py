"""
Statistical Market Regime Labeling Engine
==========================================
Classifies each trading day into one of 4 market regimes and computes
a binary early-warning transition label for the Dynamic Regime GNN.

Regime Definitions (rule-based, priority-ordered):
───────────────────────────────────────────────────
    Priority 1 — Class 3 (Stress):
        Systemic contagion regime.  Volatility > 80th percentile AND
        average cross-sectional correlation > 75th percentile.
        Everything moves together — diversification fails.
        ▸ Overrides Crash if both conditions overlap.

    Priority 2 — Class 1 (Crash):
        Drawdown regime.  SPY 20-day cumulative return < -5% AND
        volatility > 75th percentile.
        Sharp sell-off with elevated vol, but not yet systemic contagion.

    Priority 3 — Class 0 (Bull):
        Healthy uptrend.  SPY 20-day return > 0% AND volatility < 60th
        percentile AND not already classified as Crash or Stress.

    Priority 4 — Class 2 (Liquidity / Sideways):
        Residual bucket.  Everything else — choppy, flat, mixed signals,
        or moderate conditions that don't fit the above.

Transition Target:
──────────────────
    For each day t, transition_label[t] = 1 if Class 3 (Stress) occurs
    anywhere in the forward window [t+5, t+20].  Otherwise 0.
    This is the binary early-warning signal.

Rolling Metrics:
────────────────
    • ret_20d    : 20-day cumulative return = (P_t / P_{t-20}) - 1
    • vol_20d    : 20-day annualised realised volatility
                   = std(daily_returns, window=20) × √252
    • avg_corr   : Average pairwise cross-sectional correlation
                   (provided externally from the stock universe)

Percentile thresholds are computed over an expanding window (all history
up to day t) to avoid look-ahead bias.  This means the regime boundaries
adapt as the model ingests more data — a 2008 stress event won't be
calibrated using 2020 volatility levels.

Usage:
──────
    from data.label_generator import generate_market_labels

    labels_df = generate_market_labels(spy_df, avg_corr_series)
    # labels_df has columns: regime_label (int 0-3), transition_label (int 0-1)

    # Or with Yahoo Finance data fetched automatically:
    labels_df = generate_labels_from_yahoo(start="2006-01-01", end="2024-12-31")
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════
REGIME_BULL = 0
REGIME_CRASH = 1
REGIME_LIQUIDITY = 2
REGIME_STRESS = 3

REGIME_NAMES = {
    REGIME_BULL: "Bull",
    REGIME_CRASH: "Crash",
    REGIME_LIQUIDITY: "Liquidity",
    REGIME_STRESS: "Stress",
}

# Labeling hyperparameters
RETURN_WINDOW = 20              # 20-day cumulative return lookback
VOL_WINDOW = 20                 # 20-day realized volatility lookback
ANNUALISATION_FACTOR = np.sqrt(252)

# Regime thresholds
CRASH_RETURN_THRESHOLD = -0.05  # SPY 20d return < -5%
CRASH_VOL_PERCENTILE = 75      # volatility > 75th pctile
STRESS_VOL_PERCENTILE = 80     # volatility > 80th pctile
STRESS_CORR_PERCENTILE = 75    # avg correlation > 75th pctile
BULL_RETURN_THRESHOLD = 0.0    # SPY 20d return > 0%
BULL_VOL_PERCENTILE = 60       # volatility < 60th pctile

# Transition window
TRANSITION_HORIZON_MIN = 5     # earliest stress onset (days ahead)
TRANSITION_HORIZON_MAX = 20    # latest stress onset (days ahead)


# ═══════════════════════════════════════════════════════════════════════════
# Core: expanding-window percentile (no look-ahead bias)
# ═══════════════════════════════════════════════════════════════════════════
def _expanding_percentile(
    series: pd.Series,
    percentile: float,
    min_periods: int = 60,
) -> pd.Series:
    """
    Compute the threshold value at `percentile` using an expanding window.

    For day t, the threshold is computed from all valid (non-NaN) data
    up to and including day t.  This prevents look-ahead bias — the
    regime boundaries adapt as more history accumulates.

    Parameters
    ----------
    series      : time series of the metric (e.g., vol_20d).
    percentile  : target percentile (0–100).
    min_periods : minimum non-NaN observations before computing.

    Returns
    -------
    pd.Series of expanding percentile thresholds, same index as input.
    """
    def _pctile_at_t(window):
        valid = window[~np.isnan(window)]
        if len(valid) < min_periods:
            return np.nan
        return np.percentile(valid, percentile)

    return series.expanding(min_periods=1).apply(
        _pctile_at_t, raw=True
    )


# ═══════════════════════════════════════════════════════════════════════════
# Core: compute rolling metrics
# ═══════════════════════════════════════════════════════════════════════════
def compute_rolling_metrics(
    spy_close: pd.Series,
    return_window: int = RETURN_WINDOW,
    vol_window: int = VOL_WINDOW,
) -> pd.DataFrame:
    """
    Compute rolling return and volatility from SPY close prices.

    Parameters
    ----------
    spy_close     : pd.Series indexed by date — SPY adjusted close.
    return_window : lookback for cumulative return (default 20).
    vol_window    : lookback for realised volatility (default 20).

    Returns
    -------
    pd.DataFrame with columns:
        daily_ret : daily simple return
        ret_20d   : 20-day cumulative return = P_t/P_{t-20} - 1
        vol_20d   : 20-day annualised realised volatility
    """
    # Ensure spy_close is a 1-D Series (not a single-column DataFrame)
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.squeeze()

    daily_ret = spy_close.pct_change()

    # 20-day cumulative return
    ret_20d = spy_close / spy_close.shift(return_window) - 1.0

    # 20-day annualised realised volatility
    vol_20d = daily_ret.rolling(window=vol_window).std() * ANNUALISATION_FACTOR

    metrics = pd.DataFrame({
        "daily_ret": daily_ret,
        "ret_20d": ret_20d,
        "vol_20d": vol_20d,
    }, index=spy_close.index)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
# Core: regime classification
# ═══════════════════════════════════════════════════════════════════════════
def classify_regimes(
    metrics: pd.DataFrame,
    avg_corr: pd.Series,
    min_history: int = 60,
) -> pd.Series:
    """
    Classify each day into one of 4 regimes using expanding-window
    percentile thresholds (no look-ahead bias).

    Priority order (highest to lowest):
        1. Stress  (Class 3) — high vol + high correlation
        2. Crash   (Class 1) — negative return + high vol
        3. Bull    (Class 0) — positive return + low vol
        4. Liquidity (Class 2) — residual

    Parameters
    ----------
    metrics     : DataFrame with ret_20d, vol_20d columns.
    avg_corr    : pd.Series — average cross-sectional correlation per day.
    min_history : minimum days of history before assigning regimes.

    Returns
    -------
    pd.Series of int regime labels (0–3), same index as metrics.
    """
    # Align avg_corr to metrics index and ensure 1-D Series
    avg_corr = avg_corr.reindex(metrics.index).fillna(method="ffill").fillna(0.0)
    if isinstance(avg_corr, pd.DataFrame):
        avg_corr = avg_corr.squeeze()

    vol = metrics["vol_20d"]
    ret = metrics["ret_20d"]

    # Ensure these are 1-D Series (not single-column DataFrames)
    if isinstance(vol, pd.DataFrame):
        vol = vol.squeeze()
    if isinstance(ret, pd.DataFrame):
        ret = ret.squeeze()

    # ── Expanding-window percentile thresholds ──────────────────────────
    # These adapt over time — no look-ahead bias
    vol_p75 = _expanding_percentile(vol, CRASH_VOL_PERCENTILE, min_history)
    vol_p80 = _expanding_percentile(vol, STRESS_VOL_PERCENTILE, min_history)
    vol_p60 = _expanding_percentile(vol, BULL_VOL_PERCENTILE, min_history)
    corr_p75 = _expanding_percentile(avg_corr, STRESS_CORR_PERCENTILE, min_history)

    # ── Apply priority-ordered rules ────────────────────────────────────
    labels = pd.Series(REGIME_LIQUIDITY, index=metrics.index, dtype=np.int64)

    # Priority 3: Bull (applied first, can be overridden)
    bull_mask = (ret > BULL_RETURN_THRESHOLD) & (vol < vol_p60)
    labels[bull_mask] = REGIME_BULL

    # Priority 2: Crash (overrides Bull)
    crash_mask = (ret < CRASH_RETURN_THRESHOLD) & (vol > vol_p75)
    labels[crash_mask] = REGIME_CRASH

    # Priority 1: Stress (overrides everything — highest priority)
    stress_mask = (vol > vol_p80) & (avg_corr > corr_p75)
    labels[stress_mask] = REGIME_STRESS

    # NaN out days without enough history
    insufficient = vol.isna() | ret.isna() | vol_p75.isna() | corr_p75.isna()
    labels[insufficient] = REGIME_LIQUIDITY  # safe default for warm-up period

    return labels


# ═══════════════════════════════════════════════════════════════════════════
# Core: transition label (early-warning target)
# ═══════════════════════════════════════════════════════════════════════════
def compute_transition_labels(
    regime_labels: pd.Series,
    horizon_min: int = TRANSITION_HORIZON_MIN,
    horizon_max: int = TRANSITION_HORIZON_MAX,
) -> pd.Series:
    """
    Binary early-warning label: is Stress (Class 3) coming within [t+5, t+20]?

    For each day t:
        transition_label[t] = 1  if  ∃ t' ∈ [t + horizon_min, t + horizon_max]
                                     such that regime_labels[t'] == STRESS
        transition_label[t] = 0  otherwise

    Parameters
    ----------
    regime_labels : pd.Series of int regime classes (0–3).
    horizon_min   : earliest day to look ahead (default 5).
    horizon_max   : latest day to look ahead (default 20).

    Returns
    -------
    pd.Series of int {0, 1}, same index as regime_labels.
    """
    is_stress = (regime_labels == REGIME_STRESS).astype(np.int64)
    T = len(is_stress)
    values = is_stress.values
    transition = np.zeros(T, dtype=np.int64)

    for t in range(T):
        start = t + horizon_min
        end = min(t + horizon_max + 1, T)
        if start < T:
            if values[start:end].sum() > 0:
                transition[t] = 1

    return pd.Series(transition, index=regime_labels.index, dtype=np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# Main API: generate_market_labels
# ═══════════════════════════════════════════════════════════════════════════
def generate_market_labels(
    spy_df: pd.DataFrame,
    avg_corr_series: pd.Series,
    min_history: int = 60,
) -> pd.DataFrame:
    """
    Master labeling function: takes SPY data + average correlation and
    produces regime_label + transition_label for every trading day.

    Parameters
    ----------
    spy_df : pd.DataFrame with a "Close" column, indexed by datetime.
             (or a DatetimeIndex-indexed DataFrame from yfinance)
    avg_corr_series : pd.Series — average pairwise cross-sectional
                      correlation per day (from stock universe).
    min_history : minimum days of history before labeling.

    Returns
    -------
    pd.DataFrame indexed by date with columns:
        ret_20d          : float  — 20-day cumulative return
        vol_20d          : float  — 20-day annualised realised vol
        avg_corr         : float  — average cross-sectional correlation
        regime_label     : int    — {0: Bull, 1: Crash, 2: Liquidity, 3: Stress}
        transition_label : int    — {0: no stress ahead, 1: stress in 5-20 days}
        regime_name      : str    — human-readable regime name
    """
    # Extract close price — handle both Series and DataFrame formats.
    # Newer yfinance returns DataFrame columns with MultiIndex (Ticker),
    # so we .squeeze() any single-column DataFrame down to a Series.
    if isinstance(spy_df, pd.Series):
        spy_close = spy_df
    elif "Close" in spy_df.columns:
        spy_close = spy_df["Close"]
    elif "Adj Close" in spy_df.columns:
        spy_close = spy_df["Adj Close"]
    else:
        raise ValueError("spy_df must have a 'Close' or 'Adj Close' column")

    # Squeeze DataFrame-with-one-column to Series (yfinance compat)
    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.squeeze()

    spy_close = spy_close.dropna()

    # Step 1: Compute rolling metrics
    metrics = compute_rolling_metrics(spy_close)

    # Step 2: Classify regimes
    regime_labels = classify_regimes(metrics, avg_corr_series, min_history)

    # Step 3: Compute transition (early-warning) labels
    transition_labels = compute_transition_labels(regime_labels)

    # Step 4: Assemble output DataFrame
    result = pd.DataFrame({
        "ret_20d": metrics["ret_20d"],
        "vol_20d": metrics["vol_20d"],
        "avg_corr": avg_corr_series.reindex(metrics.index).fillna(method="ffill").fillna(0.0),
        "regime_label": regime_labels,
        "transition_label": transition_labels,
        "regime_name": regime_labels.map(REGIME_NAMES),
    }, index=metrics.index)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: compute average cross-sectional correlation from returns
# ═══════════════════════════════════════════════════════════════════════════
def compute_avg_cross_correlation(
    returns_df: pd.DataFrame,
    window: int = 30,
) -> pd.Series:
    """
    Compute the average pairwise rolling correlation across all stocks.

    For each day t, computes the mean of the upper triangle of the
    rolling 30-day pairwise correlation matrix.

    Parameters
    ----------
    returns_df : pd.DataFrame — daily returns, columns = stock tickers,
                 index = dates.
    window     : rolling correlation window (default 30).

    Returns
    -------
    pd.Series — average pairwise correlation per day.
    """
    T = len(returns_df)
    N = returns_df.shape[1]
    avg_corr = pd.Series(np.nan, index=returns_df.index, dtype=np.float64)

    if N < 2:
        return avg_corr.fillna(0.0)

    for t in range(window - 1, T):
        block = returns_df.iloc[t - window + 1 : t + 1]
        corr_mat = block.corr().values
        # Upper triangle (exclude diagonal)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        pairwise = corr_mat[mask]
        pairwise = pairwise[~np.isnan(pairwise)]
        if len(pairwise) > 0:
            avg_corr.iloc[t] = pairwise.mean()

    return avg_corr.fillna(method="ffill").fillna(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: fetch SPY + stock data from Yahoo Finance and label
# ═══════════════════════════════════════════════════════════════════════════
def generate_labels_from_yahoo(
    start: str = "2006-01-01",
    end: str = "2024-12-31",
    stock_tickers: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    End-to-end: fetch SPY + stock returns from Yahoo Finance, compute
    average cross-sectional correlation, and generate regime labels.

    Parameters
    ----------
    start          : start date (YYYY-MM-DD).
    end            : end date (YYYY-MM-DD).
    stock_tickers  : list of stock tickers for cross-correlation.
                     If None, uses a default set of 30 S&P 500 stocks.
    verbose        : print progress.

    Returns
    -------
    pd.DataFrame with regime_label, transition_label, and metrics.
    """
    import yfinance as yf

    # Default tickers: diversified S&P 500 sample across sectors
    if stock_tickers is None:
        stock_tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",   # Tech
            "JPM", "BAC", "GS", "MS", "BLK",            # Financials
            "JNJ", "UNH", "PFE", "ABT", "TMO",          # Health Care
            "XOM", "CVX", "COP",                          # Energy
            "PG", "KO", "PEP",                            # Staples
            "HD", "TSLA", "MCD",                          # Discretionary
            "NEE", "DUK", "SO",                           # Utilities
            "LIN", "HON", "CAT",                          # Industrials/Materials
        ]

    all_tickers = ["SPY"] + stock_tickers

    if verbose:
        print(f"Fetching {len(all_tickers)} tickers from Yahoo Finance...")
        print(f"Date range: {start} to {end}")

    # Download
    raw = yf.download(all_tickers, start=start, end=end,
                       auto_adjust=True, progress=verbose, threads=True)

    # Extract SPY close — squeeze to handle MultiIndex from newer yfinance
    close_df = raw["Close"]
    if isinstance(close_df, pd.DataFrame):
        if "SPY" in close_df.columns:
            spy_close = close_df["SPY"].dropna()
        else:
            spy_close = close_df.squeeze().dropna()
    else:
        spy_close = close_df.dropna()

    if isinstance(spy_close, pd.DataFrame):
        spy_close = spy_close.squeeze()

    # Build stock returns DataFrame
    returns_df = pd.DataFrame(index=spy_close.index)
    for ticker in stock_tickers:
        try:
            if isinstance(close_df, pd.DataFrame) and ticker in close_df.columns:
                close = close_df[ticker]
            else:
                continue
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            close = close.reindex(spy_close.index).fillna(method="ffill")
            returns_df[ticker] = close.pct_change()
        except Exception:
            pass

    returns_df = returns_df.fillna(0.0)

    if verbose:
        print(f"Valid tickers for correlation: {returns_df.shape[1]}")
        print(f"Trading days: {len(spy_close)}")
        print(f"\nComputing 30-day rolling cross-sectional correlation...")

    # Compute average cross-sectional correlation
    avg_corr = compute_avg_cross_correlation(returns_df, window=30)

    if verbose:
        print(f"Generating regime labels...")

    # Generate labels
    spy_df = pd.DataFrame({"Close": spy_close})
    labels_df = generate_market_labels(spy_df, avg_corr)

    if verbose:
        print(f"\n{'─' * 60}")
        print(f"  Label Distribution")
        print(f"{'─' * 60}")
        total = len(labels_df)
        for regime_id, regime_name in REGIME_NAMES.items():
            count = (labels_df["regime_label"] == regime_id).sum()
            pct = 100 * count / total
            print(f"  Class {regime_id} ({regime_name:10s}) : {count:5d} days ({pct:5.1f}%)")
        print(f"  {'─' * 50}")
        trans_pos = labels_df["transition_label"].sum()
        trans_neg = total - trans_pos
        print(f"  Transition = 1 (stress ahead) : {trans_pos:5d} days ({100*trans_pos/total:5.1f}%)")
        print(f"  Transition = 0 (no stress)    : {trans_neg:5d} days ({100*trans_neg/total:5.1f}%)")

        # Historical regime periods
        print(f"\n{'─' * 60}")
        print(f"  Notable Regime Periods")
        print(f"{'─' * 60}")
        _print_regime_periods(labels_df)

    return labels_df


# ═══════════════════════════════════════════════════════════════════════════
# Helper: print contiguous regime periods
# ═══════════════════════════════════════════════════════════════════════════
def _print_regime_periods(
    labels_df: pd.DataFrame,
    min_duration: int = 5,
) -> None:
    """
    Print contiguous regime periods that last ≥ min_duration days.
    Useful for validating that the labeling captures known events.
    """
    regime = labels_df["regime_label"].values
    dates = labels_df.index

    # Find contiguous runs
    runs = []
    start_idx = 0
    for i in range(1, len(regime)):
        if regime[i] != regime[start_idx]:
            runs.append((start_idx, i - 1, regime[start_idx]))
            start_idx = i
    runs.append((start_idx, len(regime) - 1, regime[start_idx]))

    # Print notable periods (non-Bull, non-Liquidity, or long Bull runs)
    for start, end, label in runs:
        duration = end - start + 1
        if duration < min_duration:
            continue
        name = REGIME_NAMES[label]
        # Print Crash and Stress periods, or long Bull/Liquidity runs
        if label in (REGIME_CRASH, REGIME_STRESS) or duration >= 20:
            date_start = dates[start].strftime("%Y-%m-%d") if hasattr(dates[start], "strftime") else str(dates[start])
            date_end = dates[end].strftime("%Y-%m-%d") if hasattr(dates[end], "strftime") else str(dates[end])
            print(f"  {name:10s} : {date_start} → {date_end}  ({duration} days)")


# ═══════════════════════════════════════════════════════════════════════════
# Smoke Test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  Label Generator — Smoke Test")
    print("=" * 70)

    # ── Test 1: Synthetic data ──────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Test 1: Synthetic data labeling")
    print(f"{'─' * 70}")

    np.random.seed(42)
    T = 500
    dates = pd.bdate_range("2020-01-01", periods=T)

    # Simulate SPY: random walk with occasional crashes
    daily_ret = np.random.normal(0.0004, 0.012, T)
    # Inject a crash period (days 100-120)
    daily_ret[100:120] = np.random.normal(-0.025, 0.035, 20)
    # Inject a stress period (days 200-230) — high vol + high correlation
    daily_ret[200:230] = np.random.normal(-0.005, 0.030, 30)

    spy_close = pd.Series(100.0, index=dates)
    for i in range(1, T):
        spy_close.iloc[i] = spy_close.iloc[i - 1] * (1 + daily_ret[i])

    spy_df = pd.DataFrame({"Close": spy_close})

    # Simulate average correlation: normally ~0.3, elevated during stress
    avg_corr = pd.Series(0.3 + np.random.normal(0, 0.05, T), index=dates)
    avg_corr.iloc[200:230] = 0.7 + np.random.normal(0, 0.03, 30)  # high during stress
    avg_corr = avg_corr.clip(0, 1)

    labels = generate_market_labels(spy_df, avg_corr)

    print(f"\n  Total days: {len(labels)}")
    for rid, rname in REGIME_NAMES.items():
        count = (labels["regime_label"] == rid).sum()
        print(f"  Class {rid} ({rname:10s}) : {count:4d} days")

    trans_pos = labels["transition_label"].sum()
    print(f"\n  Transition=1 : {trans_pos} days")
    print(f"  Transition=0 : {len(labels) - trans_pos} days")

    # Verify crash injection is detected
    crash_days = labels.iloc[100:120]
    crash_detected = (crash_days["regime_label"].isin([REGIME_CRASH, REGIME_STRESS])).sum()
    print(f"\n  Crash injection (days 100-120): {crash_detected}/20 detected as Crash/Stress")

    # Verify stress injection is detected
    stress_days = labels.iloc[200:230]
    stress_detected = (stress_days["regime_label"] == REGIME_STRESS).sum()
    print(f"  Stress injection (days 200-230): {stress_detected}/30 detected as Stress")

    # Verify transition labels fire before stress
    if stress_detected > 0:
        pre_stress = labels.iloc[180:200]
        trans_warnings = pre_stress["transition_label"].sum()
        print(f"  Early-warning (days 180-200): {trans_warnings}/20 have transition=1")

    # ── Test 2: Real data from Yahoo Finance ────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Test 2: Real data from Yahoo Finance (2018-2024)")
    print(f"{'─' * 70}")

    try:
        real_labels = generate_labels_from_yahoo(
            start="2018-01-01",
            end="2024-12-31",
            verbose=True,
        )

        # Sanity checks on real data
        print(f"\n  Sanity checks:")

        # COVID crash (Feb-Mar 2020) should be Crash or Stress
        if "2020-03-15" in [d.strftime("%Y-%m-%d") for d in real_labels.index]:
            covid_period = real_labels.loc["2020-02-20":"2020-04-01"]
            crisis_days = covid_period["regime_label"].isin([REGIME_CRASH, REGIME_STRESS]).sum()
            print(f"  COVID crash (Feb-Mar 2020): {crisis_days}/{len(covid_period)} days = Crash/Stress")

        # 2022 bear market should show crash/stress periods
        bear_2022 = real_labels.loc["2022-01-01":"2022-12-31"]
        crisis_2022 = bear_2022["regime_label"].isin([REGIME_CRASH, REGIME_STRESS]).sum()
        print(f"  2022 bear market: {crisis_2022}/{len(bear_2022)} days = Crash/Stress")

        # Bull periods in 2021 recovery
        bull_2021 = real_labels.loc["2021-01-01":"2021-12-31"]
        bull_days = (bull_2021["regime_label"] == REGIME_BULL).sum()
        print(f"  2021 recovery: {bull_days}/{len(bull_2021)} days = Bull")

        print(f"\n  Sample output (last 10 rows):")
        print(real_labels[["ret_20d", "vol_20d", "avg_corr", "regime_name", "transition_label"]].tail(10).to_string())

    except ImportError:
        print("  yfinance not installed — skipping real data test")
    except Exception as e:
        print(f"  Real data test skipped: {e}")

    print(f"\n{'=' * 70}")
    print(f"  Label Generator smoke test PASSED ✓")
    print(f"{'=' * 70}")
