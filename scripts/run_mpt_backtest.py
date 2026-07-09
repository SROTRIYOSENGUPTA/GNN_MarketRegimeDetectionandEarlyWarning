"""
Markowitz mean-variance backtest on top of run_xsec_rank.py predictions.

Loads prediction bundles produced by
`run_xsec_rank.py --save-predictions ...` and evaluates six portfolio
configurations over the same validation periods used in RESULTS.md,
reporting Sharpe, Sortino, max drawdown, turnover, and net-of-cost return.
This is the economic-significance layer the classification-only results
(RESULTS.md) don't have: beating a proxy graph on macro-F1 doesn't by
itself say whether the signal is worth trading.

Six configurations (see Report.md discussion of "Can this be made better"):
    A. bloomberg mu + GNN-window (30d) covariance, mean-variance sized   [main]
    B. bloomberg mu + long-window (252d) covariance, mean-variance sized [cov ablation]
    C. none-edge-mode mu + GNN-window covariance, mean-variance sized    [signal-source ablation]
    D. none-edge-mode mu + long-window covariance, mean-variance sized   [fully naive MV]
    E. bloomberg mu picks, equal-weighted sizing (no optimization)       [value-of-MV ablation]
    F. equal-weight buy-and-hold over the full universe                 [market proxy benchmark]

Note on F: the Bloomberg workbook (`sp500_prices 1.xlsx`) contains only
S&P 500 constituent prices, not a separate SPY series, so this is an
equal-weight-universe proxy for "the market," not a true cap-weighted
index return. Treat it as a benchmark, not a precise SPY replication.

Usage:
    python scripts/run_mpt_backtest.py \
        --predictions results/predictions/bloomberg_s42.npz \
        --none-predictions results/predictions/none_s42.npz \
        --output results/mpt_backtest/s42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn.portfolio.mean_variance import (  # noqa: E402
    estimate_covariance,
    optimize_weights,
    quintile_direction,
    shrink_signal,
)

GNN_COV_WINDOW = 30      # matches the graph's own correlation-edge window
SAMPLE_COV_WINDOW = 252  # traditional trailing-year sample covariance


def load_bundle(path):
    d = np.load(path, allow_pickle=True)
    return {
        "tickers": d["tickers"],
        "val_dates": d["val_dates"],
        "val_probs": d["val_probs"].astype(np.float64),      # (n_val, N, 3)
        "val_fwd_ret": d["val_fwd_ret"].astype(np.float64),  # (n_val, N)
        "rets_hist": d["rets_hist"].astype(np.float64),      # (T, N)
        "hist_dates": d["hist_dates"],
    }


def score_mu(val_probs):
    """P(Up) - P(Down) per stock/date, bounded in [-1, 1]."""
    return val_probs[..., 2] - val_probs[..., 0]


def rets_window_before(rets_hist, hist_dates, target_date, window):
    idx = int(np.searchsorted(hist_dates, target_date))
    lo = max(0, idx - window)
    return rets_hist[lo:idx]


def size_portfolio(mu_k, cov, sizing, gross_cap, dollar_neutral):
    direction = quintile_direction(mu_k)
    if sizing == "mean_variance":
        return optimize_weights(
            mu_k, cov, risk_aversion=1.0, gross_cap=gross_cap,
            dollar_neutral=dollar_neutral, allowed_sign=direction,
        )
    if sizing == "equal_weight_picks":
        w = np.zeros_like(mu_k)
        n_long = max(int((direction == 1).sum()), 1)
        n_short = max(int((direction == -1).sum()), 1)
        w[direction == 1] = (gross_cap / 2.0) / n_long
        w[direction == -1] = -(gross_cap / 2.0) / n_short
        return w
    if sizing == "equal_weight_universe":
        return np.full_like(mu_k, 1.0 / len(mu_k))
    raise ValueError(f"unknown sizing: {sizing}")


def infer_periods_per_year(val_dates):
    dates = np.array(val_dates, dtype="datetime64[D]")
    gaps = np.diff(dates).astype(np.float64)
    median_gap = float(np.median(gaps)) if len(gaps) else 5.0
    return 365.25 / max(median_gap, 1.0)


def run_config(name, mu_all, cov_window, sizing, rets_hist, hist_dates,
               val_dates, val_fwd_ret, shrinkage, gross_cap, dollar_neutral,
               cost_bps):
    hist_dates_dt = np.array(hist_dates, dtype="datetime64[D]")
    val_dates_dt = np.array(val_dates, dtype="datetime64[D]")
    n_periods, n_assets = val_fwd_ret.shape

    history = []
    prev_w = np.zeros(n_assets)
    equity = 1.0
    period_returns = []

    for k in range(n_periods):
        mu_k = shrink_signal(mu_all[k], shrinkage) if mu_all is not None else np.zeros(n_assets)
        if sizing == "equal_weight_universe":
            w = size_portfolio(mu_k, None, sizing, gross_cap, dollar_neutral)
        else:
            rw = rets_window_before(rets_hist, hist_dates_dt, val_dates_dt[k], cov_window)
            cov = estimate_covariance(rw)
            w = size_portfolio(mu_k, cov, sizing, gross_cap, dollar_neutral)

        turnover = float(np.abs(w - prev_w).sum())
        gross_return = float(w @ val_fwd_ret[k])
        cost = (cost_bps / 1e4) * turnover
        period_return = gross_return - cost
        equity *= (1.0 + period_return)
        period_returns.append(period_return)
        history.append({
            "date": str(val_dates[k]),
            "gross_return": gross_return,
            "cost": cost,
            "period_return": period_return,
            "turnover": turnover,
            "equity": equity,
        })
        prev_w = w

    period_returns = np.array(period_returns)
    periods_per_year = infer_periods_per_year(val_dates)
    mean_r, std_r = period_returns.mean(), period_returns.std(ddof=1) if len(period_returns) > 1 else 0.0
    downside = period_returns[period_returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0

    equity_curve = np.array([h["equity"] for h in history])
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    sharpe = float(mean_r / std_r * np.sqrt(periods_per_year)) if std_r > 0 else float("nan")
    sortino = float(mean_r / downside_std * np.sqrt(periods_per_year)) if downside_std > 0 else float("nan")
    avg_turnover = float(period_returns.__len__() and np.mean([h["turnover"] for h in history]))

    return {
        "name": name,
        "n_periods": n_periods,
        "total_return": float(equity_curve[-1] - 1.0) if len(equity_curve) else 0.0,
        "annualized_return": float((equity_curve[-1]) ** (periods_per_year / max(n_periods, 1)) - 1.0) if len(equity_curve) else 0.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="npz from --save-predictions for the main (e.g. bloomberg) edge mode")
    p.add_argument("--none-predictions", default=None, help="npz for the no-graph edge mode, needed for configs C/D")
    p.add_argument("--output", required=True)
    p.add_argument("--shrinkage", type=float, default=0.3)
    p.add_argument("--gross-cap", type=float, default=2.0)
    p.add_argument("--dollar-neutral", action="store_true", default=True)
    p.add_argument("--cost-bps", type=float, default=5.0)
    args = p.parse_args()

    main_bundle = load_bundle(args.predictions)
    none_bundle = load_bundle(args.none_predictions) if args.none_predictions else None
    if none_bundle is not None:
        assert list(main_bundle["tickers"]) == list(none_bundle["tickers"]), \
            "predictions and none-predictions must share the same ticker universe"

    mu_bloomberg = score_mu(main_bundle["val_probs"])
    mu_none = score_mu(none_bundle["val_probs"]) if none_bundle is not None else None

    configs = [
        dict(name="A_bloomberg_gnn_cov_mv", mu=mu_bloomberg, cov_window=GNN_COV_WINDOW, sizing="mean_variance"),
        dict(name="B_bloomberg_sample_cov_mv", mu=mu_bloomberg, cov_window=SAMPLE_COV_WINDOW, sizing="mean_variance"),
        dict(name="E_bloomberg_equal_weight_picks", mu=mu_bloomberg, cov_window=GNN_COV_WINDOW, sizing="equal_weight_picks"),
        dict(name="F_equal_weight_universe", mu=None, cov_window=GNN_COV_WINDOW, sizing="equal_weight_universe"),
    ]
    if mu_none is not None:
        configs.append(dict(name="C_none_gnn_cov_mv", mu=mu_none, cov_window=GNN_COV_WINDOW, sizing="mean_variance"))
        configs.append(dict(name="D_none_sample_cov_mv", mu=mu_none, cov_window=SAMPLE_COV_WINDOW, sizing="mean_variance"))
    else:
        print("WARNING: --none-predictions not given, skipping configs C and D", flush=True)

    results = {}
    for cfg in configs:
        print(f"running config {cfg['name']}...", flush=True)
        results[cfg["name"]] = run_config(
            cfg["name"], cfg["mu"], cfg["cov_window"], cfg["sizing"],
            main_bundle["rets_hist"], main_bundle["hist_dates"],
            main_bundle["val_dates"], main_bundle["val_fwd_ret"],
            shrinkage=args.shrinkage, gross_cap=args.gross_cap,
            dollar_neutral=args.dollar_neutral, cost_bps=args.cost_bps,
        )
        r = results[cfg["name"]]
        print(f"  sharpe={r['sharpe']:.3f} sortino={r['sortino']:.3f} "
              f"max_dd={r['max_drawdown']:.3f} total_return={r['total_return']:.3f} "
              f"avg_turnover={r['avg_turnover']:.3f}", flush=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "predictions_source": args.predictions,
        "none_predictions_source": args.none_predictions,
        "shrinkage": args.shrinkage,
        "gross_cap": args.gross_cap,
        "dollar_neutral": args.dollar_neutral,
        "cost_bps": args.cost_bps,
        "configs": results,
    }, indent=2))
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); raise
