"""
Fama-MacBeth: is the graph signal distinct from known cross-sectional predictors?

For each rebalance date t we run a cross-sectional regression

    fwd_ret[i,t] = a_t + b_t * mu[i,t] + controls + industry FE + e[i,t]

then average b_t over dates and test it with Newey-West standard errors
(lag 5, matching the 5-day forward horizon and the overlapping-return
structure it induces).

Controls computed from the saved return history in each prediction bundle:
  mom_12_2   12-month momentum skipping the most recent month
  rev_1m     prior-month return (short-term reversal) -- the most likely
             confound for a 5-day-horizon cross-sectional signal
  vol_60     60-day realised volatility
  industry FE  demeaning within GICS sector, so the coefficient is
             identified off WITHIN-industry variation. This is the direct
             test of "is the graph signal just industry momentum?"

NOT INCLUDED, and this is a real limitation: size (market capitalisation)
and book-to-market. Both require data this project does not have
(Bloomberg CUR_MKT_CAP / PX_TO_BOOK_RATIO, or CRSP/Compustat via WRDS).
Any claim of full orthogonality to the standard factor set must wait for
those. What follows is a partial but non-trivial test.

Usage: python scripts/analyze_fama_macbeth.py <predictions_dir> <sector_csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 123, 7]
WINDOWS = [("2016Q4-2017", "w2017"), ("2017Q4-2018", "w2018"),
           ("2018Q4-2019", "w2019"), ("2019Q4-2020", "w2020"),
           ("2020Q4-2021", "w2021"), ("2021Q4-2022", "w2022"),
           ("2022Q4-2024", "w2024")]


def newey_west_t(x, lag=5):
    """Mean of x with Newey-West corrected t-stat (handles overlap/autocorr)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    m = x.mean()
    e = x - m
    gamma0 = (e @ e) / n
    var = gamma0
    for L in range(1, min(lag, n - 1) + 1):
        g = (e[L:] @ e[:-L]) / n
        var += 2 * (1 - L / (lag + 1)) * g
    se = np.sqrt(max(var, 1e-18) / n)
    return m, m / se


def zscore(v):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return np.full_like(v, np.nan)
    s = v[ok].std()
    out = np.full_like(v, np.nan)
    out[ok] = (v[ok] - v[ok].mean()) / (s if s > 0 else 1.0)
    return out


def demean_by_group(v, g):
    """Industry fixed effects, implemented as within-group demeaning."""
    out = np.array(v, float)
    for grp in np.unique(g):
        m = (g == grp) & np.isfinite(out)
        if m.sum() >= 2:
            out[m] -= out[m].mean()
    return out


def ols_beta(y, X):
    """Return coefficient vector; X already includes intercept column."""
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if ok.sum() < X.shape[1] + 5:
        return None
    Xo, yo = X[ok], y[ok]
    try:
        return np.linalg.lstsq(Xo, yo, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None


def build_controls(rets_hist, hist_dates, val_dates):
    """Per (date, stock) momentum, reversal, volatility from the return panel."""
    hd = np.array(hist_dates, dtype="datetime64[D]")
    vd = np.array(val_dates, dtype="datetime64[D]")
    out = []
    for d in vd:
        i = int(np.searchsorted(hd, d))
        w = rets_hist[max(0, i - 252):i]
        mom = np.nansum(w[:-21], axis=0) if w.shape[0] > 42 else np.full(rets_hist.shape[1], np.nan)
        rev = np.nansum(w[-21:], axis=0) if w.shape[0] >= 21 else np.full(rets_hist.shape[1], np.nan)
        vol = np.nanstd(w[-60:], axis=0) if w.shape[0] >= 60 else np.full(rets_hist.shape[1], np.nan)
        out.append((mom, rev, vol))
    return out


def run_bundle(path, sector_lut):
    d = np.load(path, allow_pickle=True)
    tickers = [str(t) for t in d["tickers"]]
    probs = d["val_probs"].astype(float)
    fwd = d["val_fwd_ret"].astype(float)
    mu_all = probs[..., 2] - probs[..., 0]
    ctrl = build_controls(d["rets_hist"].astype(float), d["hist_dates"], d["val_dates"])
    sect = np.array([sector_lut.get(t, "UNK") for t in tickers])

    raw, full = [], []
    for k in range(mu_all.shape[0]):
        y = fwd[k]
        mom, rev, vol = ctrl[k]
        mu = zscore(mu_all[k])
        n = len(y)
        # (a) signal alone
        b = ols_beta(y, np.column_stack([np.ones(n), mu]))
        if b is not None:
            raw.append(b[1])
        # (b) + controls, within-industry
        yd = demean_by_group(y, sect)
        Xd = np.column_stack([np.ones(n), demean_by_group(mu, sect),
                              zscore(mom), zscore(rev), zscore(vol)])
        b2 = ols_beta(yd, Xd)
        if b2 is not None:
            full.append(b2[1])
    return raw, full


def main():
    pred_dir = Path(sys.argv[1])
    df = pd.read_csv(sys.argv[2], keep_default_na=False)
    lut = {str(r["ticker"]): str(r["sector"]) or "UNK" for _, r in df.iterrows()}

    print("=" * 96, flush=True)
    print(" FAMA-MacBETH — is the graph signal distinct from momentum, reversal, volatility,", flush=True)
    print(" and industry effects?  (coefficient on standardised signal, bp per 5-day period)", flush=True)
    print("=" * 96, flush=True)
    print("\n  NOTE: size and book-to-market are NOT controlled for (data unavailable).", flush=True)
    print("  Industry FE = within-GICS-sector demeaning of both y and the signal.\n", flush=True)
    print(f"  {'window':<15}{'signal alone':>26}{'+ controls + industry FE':>30}", flush=True)
    print("  " + "-" * 71, flush=True)

    agg_raw, agg_full = [], []
    for label, tag in WINDOWS:
        r_all, f_all = [], []
        for s in SEEDS:
            p = pred_dir / f"subind_{tag}_s{s}.npz"
            if not p.exists():
                continue
            r, f = run_bundle(p, lut)
            r_all.append(r); f_all.append(f)
        if not r_all:
            continue
        r_m = np.mean(np.array(r_all), axis=0)
        f_m = np.mean(np.array(f_all), axis=0)
        rb, rt = newey_west_t(r_m)
        fb, ft = newey_west_t(f_m)
        agg_raw.extend(r_m); agg_full.extend(f_m)
        print(f"  {label:<15}{rb*1e4:>14.2f}bp (t={rt:+5.2f}){fb*1e4:>16.2f}bp (t={ft:+5.2f})", flush=True)

    rb, rt = newey_west_t(agg_raw)
    fb, ft = newey_west_t(agg_full)
    print("  " + "-" * 71, flush=True)
    print(f"  {'POOLED':<15}{rb*1e4:>14.2f}bp (t={rt:+5.2f}){fb*1e4:>16.2f}bp (t={ft:+5.2f})", flush=True)
    print(f"\n  pooled n = {len(agg_full)} cross-sectional regressions", flush=True)
    if np.isfinite(ft):
        keep = abs(fb) / abs(rb) if rb else float("nan")
        print(f"  signal retains {keep:.0%} of its univariate magnitude after controls", flush=True)


if __name__ == "__main__":
    main()
