"""
Does the signal work as INDUSTRY ROTATION rather than stock selection?

Motivation. The Fama-MacBeth regressions (RESULTS.md Finding 9) showed the
graph signal's predictive content is largely BETWEEN industries, not within
them: the coefficient collapses once industry fixed effects absorb
between-group variation. But every portfolio test so far used a quintile
long-short book over all ~500 stocks, which by construction is dominated by
within-industry variation and largely nets out industry bets.

That is a mismatch: we measured a group-level signal with a stock-selection
portfolio. This script tests the construction that matches the signal's
actual content.

Two experiments:

  A. INDUSTRY ROTATION. Aggregate mu to the sub-industry level, rank groups,
     go long the top-quintile groups and short the bottom-quintile groups,
     equal-weighting members inside each group. Compared head-to-head with
     the stock-level quintile book on identical predictions and dates.

  B. HORIZON SWEEP. Group-level information may propagate more slowly than
     5 days. Forward returns at 5/10/20/60 days are compounded from the daily
     return panel stored in each bundle, and both constructions are evaluated
     at each horizon. Rank IC (Spearman) is reported alongside spread since it
     is the standard signal-quality metric and is less sensitive to the
     tails than a quintile spread.

Usage:
    python scripts/analyze_industry_rotation.py <predictions_dir> <sector_csv>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 123, 7]
WINDOWS = [("2016Q4-2017", "w2017"), ("2017Q4-2018", "w2018"),
           ("2018Q4-2019", "w2019"), ("2019Q4-2020", "w2020"),
           ("2020Q4-2021", "w2021"), ("2021Q4-2022", "w2022"),
           ("2022Q4-2024", "w2024")]
HORIZONS = [5, 10, 20, 60]
GROUP_COL = "gics_sub_industry"


def stats(v):
    v = np.asarray([x for x in v if np.isfinite(x)], float)
    if len(v) == 0:
        return np.nan, np.nan
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def t_p(a, df):
    tbl = {2: [(2.92, "<.10"), (4.30, "<.05"), (9.92, "<.01")],
           6: [(1.94, "<.10"), (2.45, "<.05"), (3.71, "<.01")],
           20: [(1.72, "<.10"), (2.09, "<.05"), (2.85, "<.01")]}
    key = min(tbl, key=lambda k: abs(k - df))
    res = "n.s."
    for thr, lbl in tbl[key]:
        if a >= thr:
            res = lbl
    return res


def paired_t(a, b):
    d = [x - y for x, y in zip(a, b) if np.isfinite(x) and np.isfinite(y)]
    if len(d) < 3:
        return np.nan, np.nan, "n.s."
    m, s = stats(d)
    t = m / (s / math.sqrt(len(d))) if s > 0 else float("inf")
    return m, t, t_p(abs(t), len(d) - 1)


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / den) if den > 0 else np.nan


def forward_return(rets_hist, hist_dates, val_dates, horizon):
    """Compound the daily return panel forward `horizon` trading days."""
    hd = np.array(hist_dates, dtype="datetime64[D]")
    vd = np.array(val_dates, dtype="datetime64[D]")
    T, N = rets_hist.shape
    out = np.full((len(vd), N), np.nan)
    for k, d in enumerate(vd):
        i = int(np.searchsorted(hd, d))
        j = min(i + horizon, T)
        if j - i < max(horizon // 2, 2):
            continue
        w = rets_hist[i:j]
        out[k] = np.nanprod(1.0 + w, axis=0) - 1.0
    return out


def stock_ls(mu, fwd, q=0.2):
    """Stock-level quintile long-short spread on one date."""
    ok = np.isfinite(mu) & np.isfinite(fwd)
    if ok.sum() < 20:
        return np.nan
    m = mu.copy(); m[~ok] = -np.inf
    b = mu.copy(); b[~ok] = np.inf
    k = max(int(ok.sum() * q), 1)
    return float(fwd[np.argsort(-m)[:k]].mean() - fwd[np.argsort(b)[:k]].mean())


def industry_ls(mu, fwd, groups, q=0.2, min_members=2):
    """Group-level rotation: rank sub-industries by mean mu, long/short groups."""
    ok = np.isfinite(mu) & np.isfinite(fwd)
    if ok.sum() < 20:
        return np.nan
    gs, gmu, gret = [], [], []
    for g in np.unique(groups):
        m = (groups == g) & ok
        if m.sum() >= min_members:
            gs.append(g); gmu.append(mu[m].mean()); gret.append(fwd[m].mean())
    if len(gs) < 10:
        return np.nan
    gmu = np.asarray(gmu); gret = np.asarray(gret)
    k = max(int(len(gs) * q), 1)
    top = np.argsort(-gmu)[:k]
    bot = np.argsort(gmu)[:k]
    return float(gret[top].mean() - gret[bot].mean())


def main():
    pred_dir = Path(sys.argv[1])
    sect = pd.read_csv(sys.argv[2], keep_default_na=False)
    lut = {str(r["ticker"]): (str(r[GROUP_COL]) or "UNK") for _, r in sect.iterrows()}

    # ---------------- A. industry rotation vs stock selection, 5-day --------
    print("=" * 94, flush=True)
    print(" A. INDUSTRY ROTATION vs STOCK SELECTION — same signal, same dates, 5-day horizon", flush=True)
    print("=" * 94, flush=True)
    print(f"\n  {'window':<15}{'stock-level LS':>20}{'industry rotation':>22}{'difference':>16}", flush=True)
    print("  " + "-" * 72, flush=True)

    per_window_stock, per_window_ind = [], []
    for label, tag in WINDOWS:
        s_vals, i_vals = [], []
        for sd in SEEDS:
            p = pred_dir / f"subind_{tag}_s{sd}.npz"
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            tick = [str(t) for t in d["tickers"]]
            grp = np.array([lut.get(t, "UNK") for t in tick])
            probs = d["val_probs"].astype(float)
            fwd = d["val_fwd_ret"].astype(float)
            mu = probs[..., 2] - probs[..., 0]
            s_vals.append(np.nanmean([stock_ls(mu[k], fwd[k]) for k in range(mu.shape[0])]))
            i_vals.append(np.nanmean([industry_ls(mu[k], fwd[k], grp) for k in range(mu.shape[0])]))
        if not s_vals:
            continue
        sm = np.nanmean(s_vals); im = np.nanmean(i_vals)
        per_window_stock.append(sm); per_window_ind.append(im)
        print(f"  {label:<15}{sm*1e4:>17.2f}bp{im*1e4:>19.2f}bp{(im-sm)*1e4:>13.2f}bp", flush=True)

    dm, dt, dp = paired_t(per_window_ind, per_window_stock)
    print("  " + "-" * 72, flush=True)
    print(f"  {'MEAN':<15}{np.nanmean(per_window_stock)*1e4:>17.2f}bp"
          f"{np.nanmean(per_window_ind)*1e4:>19.2f}bp{dm*1e4:>13.2f}bp", flush=True)
    print(f"\n  rotation minus stock-level: {dm*1e4:+.2f}bp  t={dt:+.2f}  p{dp}  "
          f"(n={len(per_window_ind)} windows)", flush=True)
    wins = sum(1 for a, b in zip(per_window_ind, per_window_stock) if a > b)
    print(f"  rotation better in {wins}/{len(per_window_ind)} windows", flush=True)

    # ---------------- B. horizon sweep -------------------------------------
    print("\n" + "=" * 94, flush=True)
    print(" B. HORIZON SWEEP — does a group-level signal need longer to pay off?", flush=True)
    print("=" * 94, flush=True)
    print(f"\n  {'horizon':>9}{'stock LS':>14}{'rotation LS':>16}"
          f"{'stock rankIC':>16}{'rotation rankIC':>18}", flush=True)
    print("  " + "-" * 71, flush=True)

    for H in HORIZONS:
        s_ls, i_ls, s_ic, i_ic = [], [], [], []
        for label, tag in WINDOWS:
            for sd in SEEDS:
                p = pred_dir / f"subind_{tag}_s{sd}.npz"
                if not p.exists():
                    continue
                d = np.load(p, allow_pickle=True)
                tick = [str(t) for t in d["tickers"]]
                grp = np.array([lut.get(t, "UNK") for t in tick])
                probs = d["val_probs"].astype(float)
                mu = probs[..., 2] - probs[..., 0]
                fwd = forward_return(d["rets_hist"].astype(float), d["hist_dates"],
                                     d["val_dates"], H)
                s_ls.append(np.nanmean([stock_ls(mu[k], fwd[k]) for k in range(mu.shape[0])]))
                i_ls.append(np.nanmean([industry_ls(mu[k], fwd[k], grp) for k in range(mu.shape[0])]))
                s_ic.append(np.nanmean([spearman(mu[k], fwd[k]) for k in range(mu.shape[0])]))
                # group-level IC
                gi = []
                for k in range(mu.shape[0]):
                    ok = np.isfinite(mu[k]) & np.isfinite(fwd[k])
                    gm, gr = [], []
                    for g in np.unique(grp):
                        m = (grp == g) & ok
                        if m.sum() >= 2:
                            gm.append(mu[k][m].mean()); gr.append(fwd[k][m].mean())
                    if len(gm) >= 10:
                        gi.append(spearman(np.asarray(gm), np.asarray(gr)))
                i_ic.append(np.nanmean(gi) if gi else np.nan)
        print(f"  {H:>7}d{np.nanmean(s_ls)*1e4:>11.2f}bp{np.nanmean(i_ls)*1e4:>13.2f}bp"
              f"{np.nanmean(s_ic):>16.4f}{np.nanmean(i_ic):>18.4f}", flush=True)

    print("\n  rankIC is the Spearman correlation between signal and realised forward", flush=True)
    print("  return, computed per date then averaged — the standard signal-quality", flush=True)
    print("  measure, and less tail-sensitive than a quintile spread.", flush=True)


if __name__ == "__main__":
    main()
