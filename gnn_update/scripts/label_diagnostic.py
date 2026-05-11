"""
Label-distribution diagnostic for the workbook flow.

Loads the Bloomberg workbook the same way run_sp500_workbook_experiment.py does
(synthetic market index + 30-day rolling avg cross-sectional correlation),
then sweeps a few labeling configurations and prints:
    - day-level regime distribution
    - day-level transition prevalence
    - coverage of named historical stress events
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "GNN_MarketRegimeDetectionandEarlyWarning"))

from market_regime_gnn._legacy.data import label_generator as lg


XLSX = "/scratch/ss3414/gnn_regime/sp500_prices 1.xlsx"


def _load_workbook(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sheets = pd.read_excel(path, sheet_name=None)
    rows = pd.concat(sheets.values(), ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["ticker"] = rows["ticker"].astype(str)
    close = rows.pivot_table(index="date", columns="ticker", values="px_last", aggfunc="last").sort_index()
    volume = rows.pivot_table(index="date", columns="ticker", values="px_volume", aggfunc="last").sort_index()
    common = close.columns.intersection(volume.columns)
    close = close[common].ffill().bfill()
    volume = volume[common].fillna(0.0)
    return close, volume


def _fast_avg_corr(returns_df: pd.DataFrame, window: int = 30) -> pd.Series:
    T, N = returns_df.shape
    out = pd.Series(np.nan, index=returns_df.index, dtype=np.float64)
    if N < 2:
        return out.fillna(0.0)
    arr = returns_df.to_numpy(dtype=np.float64)
    for t in range(window - 1, T):
        block = arr[t - window + 1 : t + 1]
        # Pearson correlation across columns
        block = block - block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        block = block / sd
        c = (block.T @ block) / window
        iu = np.triu_indices(N, k=1)
        vals = c[iu]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out.iloc[t] = vals.mean()
    return out.ffill().fillna(0.0)


def _rolling_percentile(series: pd.Series, percentile: float, window: int, min_periods: int) -> pd.Series:
    return series.rolling(window=window, min_periods=min_periods).quantile(percentile / 100.0)


def classify_with_config(
    metrics: pd.DataFrame,
    avg_corr: pd.Series,
    cfg: dict,
) -> pd.Series:
    vol = metrics["vol_20d"]
    ret = metrics["ret_20d"]
    if cfg["window_kind"] == "expanding":
        f = lambda s, p: lg._expanding_percentile(s, p, cfg["min_history"])
    else:
        f = lambda s, p: _rolling_percentile(s, p, cfg["window_size"], cfg["min_history"])
    vol_p_stress = f(vol, cfg["stress_vol_p"])
    vol_p_crash = f(vol, cfg["crash_vol_p"])
    vol_p_bull = f(vol, cfg["bull_vol_p"])
    corr_p_stress = f(avg_corr, cfg["stress_corr_p"])

    labels = pd.Series(lg.REGIME_LIQUIDITY, index=metrics.index, dtype=np.int64)
    labels[(ret > cfg["bull_ret"]) & (vol < vol_p_bull)] = lg.REGIME_BULL
    labels[(ret < cfg["crash_ret"]) & (vol > vol_p_crash)] = lg.REGIME_CRASH
    labels[(vol > vol_p_stress) & (avg_corr > corr_p_stress)] = lg.REGIME_STRESS

    insufficient = vol.isna() | ret.isna() | vol_p_stress.isna() | corr_p_stress.isna()
    labels[insufficient] = lg.REGIME_LIQUIDITY
    # Drop labels in warm-up
    if cfg.get("warmup_days"):
        warm = labels.index < labels.index[0] + pd.Timedelta(days=cfg["warmup_days"] * 1.5)
        # Actually use position-based warmup
        n = cfg["warmup_days"]
        labels.iloc[:n] = lg.REGIME_LIQUIDITY
    return labels


NAMED_EVENTS = {
    "COVID crash":      ("2020-02-20", "2020-04-15"),
    "2022 rate hike Q2": ("2022-04-01", "2022-06-30"),
    "2022 rate hike Q3": ("2022-08-15", "2022-10-15"),
    "SVB collapse":     ("2023-03-08", "2023-03-25"),
    "Aug 2024 carry":   ("2024-08-01", "2024-08-15"),
}


def event_coverage(labels: pd.Series, kinds: Iterable[int]) -> dict:
    out = {}
    for name, (s, e) in NAMED_EVENTS.items():
        win = labels.loc[s:e]
        if len(win) == 0:
            out[name] = (0, 0, 0.0)
            continue
        n_hit = win.isin(list(kinds)).sum()
        out[name] = (int(n_hit), int(len(win)), n_hit / len(win))
    return out


def print_summary(name: str, labels: pd.Series, transitions: pd.Series) -> None:
    n = len(labels)
    print(f"\n=== {name} ===")
    print(f"  Total days: {n}")
    counts = labels.value_counts().to_dict()
    for rid, rname in lg.REGIME_NAMES.items():
        c = counts.get(rid, 0)
        print(f"    {rname:10s} {c:5d}  ({100*c/n:5.1f}%)")
    print(f"  Transition=1: {int(transitions.sum())} ({100*transitions.mean():.1f}%)")
    cov = event_coverage(labels, kinds=(lg.REGIME_CRASH, lg.REGIME_STRESS))
    print(f"  Event coverage (Crash+Stress %):")
    for ev, (hit, total, frac) in cov.items():
        print(f"    {ev:22s} {hit:3d}/{total:3d}  ({100*frac:5.1f}%)")


def main() -> None:
    print("Loading workbook...")
    close, volume = _load_workbook(XLSX)
    print(f"  close: {close.shape}  range: {close.index.min().date()} → {close.index.max().date()}")
    rets = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_ret = rets.mean(axis=1)
    market_close = (1.0 + market_ret).cumprod() * 100.0
    print("  computing avg cross-correlation (30d)...")
    avg_corr = _fast_avg_corr(rets, window=30)
    print(f"  avg_corr range: [{avg_corr.min():.3f}, {avg_corr.max():.3f}]  mean: {avg_corr.mean():.3f}")

    metrics = lg.compute_rolling_metrics(market_close)

    configs = {
        "current (expanding p80/75/60, min_history=60)": dict(
            window_kind="expanding", min_history=60,
            stress_vol_p=80, stress_corr_p=75,
            crash_vol_p=75, crash_ret=-0.05,
            bull_vol_p=60, bull_ret=0.0,
            warmup_days=60,
        ),
        "rolling-504, p80/75/60 (just window change)": dict(
            window_kind="rolling", window_size=504, min_history=252,
            stress_vol_p=80, stress_corr_p=75,
            crash_vol_p=75, crash_ret=-0.05,
            bull_vol_p=60, bull_ret=0.0,
            warmup_days=252,
        ),
        "rolling-504, p90/85, bull p70 ret>1%": dict(
            window_kind="rolling", window_size=504, min_history=252,
            stress_vol_p=90, stress_corr_p=85,
            crash_vol_p=80, crash_ret=-0.05,
            bull_vol_p=70, bull_ret=0.01,
            warmup_days=252,
        ),
        "rolling-756, p90/85, bull p75 ret>1%": dict(
            window_kind="rolling", window_size=756, min_history=252,
            stress_vol_p=90, stress_corr_p=85,
            crash_vol_p=80, crash_ret=-0.05,
            bull_vol_p=75, bull_ret=0.01,
            warmup_days=252,
        ),
    }

    for name, cfg in configs.items():
        labels = classify_with_config(metrics, avg_corr, cfg)
        transitions = lg.compute_transition_labels(labels)
        print_summary(name, labels, transitions)


if __name__ == "__main__":
    main()
