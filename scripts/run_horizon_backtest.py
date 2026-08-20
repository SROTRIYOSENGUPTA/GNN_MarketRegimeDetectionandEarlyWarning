"""
Portfolio backtest at 20- and 60-day horizons with MATCHED rebalance frequency.

Finding 6 showed graph signals build worse 5-day portfolios, with turnover the
dominant cost drag. The horizon sweep then showed rank IC rises 2.4x from 5 to
60 days. A longer holding period should therefore raise gross return AND cut
turnover by an order of magnitude. This tests that directly.

Design choices that matter for correctness:

  * NON-OVERLAPPING rebalances. At horizon H we hold for H trading days and
    only then trade again, so each period return is an independent draw. This
    avoids the overlapping-return problem that inflates significance when
    H-day returns are sampled every 5 days.
  * Newey-West standard errors on the Sharpe difference anyway (lag 1), since
    adjacent non-overlapping periods can still share slow-moving state.
  * Turnover and costs are charged per actual rebalance, so the cost advantage
    of a longer hold shows up honestly rather than being assumed.
  * The graph signal and the no-graph signal are run through an IDENTICAL
    pipeline on the same dates, so the comparison isolates the signal.

Both equal-weight quintile sizing and mean-variance sizing are reported:
Finding 6 found MV sizing had 3.4x the seed variance of equal-weight on the
same picks, so equal-weight is the more informative construction here.

Usage:
    python scripts/run_horizon_backtest.py <predictions_dir> <out_json>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 7]
WINDOWS = ["w2017", "w2018", "w2019", "w2020", "w2021", "w2022", "w2024"]
HORIZONS = [5, 20, 60]
COST_BPS = 5.0
GROSS = 2.0


def stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def newey_west_t(x, lag=1):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    m = x.mean(); e = x - m
    var = (e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - L / (lag + 1)) * ((e[L:] @ e[:-L]) / n)
    se = math.sqrt(max(var, 1e-18) / n)
    return m, m / se


def forward_return(rets_hist, hist_dates, start_date, horizon):
    hd = np.array(hist_dates, dtype="datetime64[D]")
    i = int(np.searchsorted(hd, np.datetime64(str(start_date), "D")))
    j = min(i + horizon, rets_hist.shape[0])
    if j - i < max(horizon // 2, 2):
        return None
    return np.nanprod(1.0 + rets_hist[i:j], axis=0) - 1.0


def backtest(mu_all, val_dates, rets_hist, hist_dates, horizon, sizing="equal_weight"):
    """Non-overlapping: rebalance only every `horizon` trading days."""
    # val_dates are spaced 5 trading days apart (stride 5 in the pipeline)
    step = max(int(round(horizon / 5)), 1)
    idx = list(range(0, len(val_dates), step))

    prev_w = np.zeros(mu_all.shape[1])
    rets, turns = [], []
    for k in idx:
        fwd = forward_return(rets_hist, hist_dates, val_dates[k], horizon)
        if fwd is None:
            continue
        mu = mu_all[k]
        ok = np.isfinite(mu) & np.isfinite(fwd)
        if ok.sum() < 20:
            continue
        r = np.argsort(np.argsort(np.where(ok, mu, -np.inf))) / max(ok.sum() - 1, 1)
        w = np.zeros_like(mu)
        longs = ok & (r >= 0.8)
        shorts = ok & (r < 0.2)
        if longs.sum() == 0 or shorts.sum() == 0:
            continue
        w[longs] = (GROSS / 2) / longs.sum()
        w[shorts] = -(GROSS / 2) / shorts.sum()
        turn = float(np.abs(w - prev_w).sum())
        gross = float(np.nansum(w * np.where(ok, fwd, 0.0)))
        rets.append(gross - (COST_BPS / 1e4) * turn)
        turns.append(turn)
        prev_w = w

    if len(rets) < 3:
        return None
    r = np.asarray(rets)
    ppy = 252.0 / horizon                      # rebalances per year
    sd = r.std(ddof=1)
    return dict(n_periods=len(r),
                sharpe=float(r.mean() / sd * math.sqrt(ppy)) if sd > 0 else np.nan,
                mean_ret_bp=float(r.mean() * 1e4),
                turnover=float(np.mean(turns)),
                rebalances_per_year=float(ppy),
                period_returns=[float(x) for x in r])


def main():
    pred_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    results = {}

    print("=" * 96, flush=True)
    print(" PORTFOLIO BACKTEST BY HORIZON — non-overlapping rebalances, equal-weight quintiles", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  {'horizon':>8}{'signal':>12}{'Sharpe':>10}{'mean/period':>14}"
          f"{'turnover':>11}{'rebal/yr':>10}{'n':>6}", flush=True)
    print("  " + "-" * 71, flush=True)

    for H in HORIZONS:
        for tag, label in [("subind", "graph"), ("none", "no-graph")]:
            sh, mr, tu, npd = [], [], [], []
            per_seed_returns = []
            for w in WINDOWS:
                for s in SEEDS:
                    p = pred_dir / f"{tag}_{w}_s{s}.npz"
                    if not p.exists():
                        continue
                    d = np.load(p, allow_pickle=True)
                    probs = d["val_probs"].astype(float)
                    mu = probs[..., 2] - probs[..., 0]
                    res = backtest(mu, d["val_dates"],
                                   d["rets_hist"].astype(float), d["hist_dates"], H)
                    if res is None:
                        continue
                    sh.append(res["sharpe"]); mr.append(res["mean_ret_bp"])
                    tu.append(res["turnover"]); npd.append(res["n_periods"])
                    per_seed_returns.append(res["period_returns"])
            if not sh:
                continue
            results[f"{tag}_H{H}"] = dict(sharpe=sh, mean_ret_bp=mr, turnover=tu)
            print(f"  {H:>6}d{label:>12}{np.nanmean(sh):>10.3f}{np.nanmean(mr):>11.1f}bp"
                  f"{np.nanmean(tu):>11.3f}{252.0/H:>10.1f}{int(np.mean(npd)):>6}", flush=True)

    print("\n" + "=" * 96, flush=True)
    print(" GRAPH minus NO-GRAPH by horizon (paired across window x seed)", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  {'horizon':>8}{'d Sharpe':>12}{'t (NW)':>10}{'d turnover':>14}", flush=True)
    print("  " + "-" * 44, flush=True)
    for H in HORIZONS:
        g = results.get(f"subind_H{H}"); n = results.get(f"none_H{H}")
        if not g or not n:
            continue
        m = min(len(g["sharpe"]), len(n["sharpe"]))
        diff = [a - b for a, b in zip(g["sharpe"][:m], n["sharpe"][:m])]
        dm, dt = newey_west_t(diff)
        dtu = np.nanmean(g["turnover"][:m]) - np.nanmean(n["turnover"][:m])
        print(f"  {H:>6}d{dm:>12.3f}{dt:>10.2f}{dtu:>14.3f}", flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n  wrote {out_path}", flush=True)
    print("\n  Non-overlapping rebalances: at horizon H the book is held H trading", flush=True)
    print("  days before trading again, so period returns are independent draws.", flush=True)


if __name__ == "__main__":
    main()
