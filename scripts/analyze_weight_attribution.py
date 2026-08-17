"""
Weight-level attribution: does graph message passing smooth predictions
toward peer-group means, producing concentrated group bets?

Two tests, both comparing a graph signal against the no-graph signal:

  A. SIGNAL SMOOTHING (cheap, no optimisation).  Decompose the
     cross-sectional variance of mu = P(Up)-P(Down) on each date into
     within-sub-industry and between-sub-industry components. If graph
     message passing homogenises predictions inside a peer group, the graph
     signal's WITHIN share should be lower -- its dispersion lives between
     groups, not inside them. That is exactly the property that lifts
     per-date macro-F1 (whole groups ranked correctly) while starving
     portfolio construction, which needs dispersion in the tails.

  B. PORTFOLIO CONCENTRATION (expensive, re-solves the optimiser).  Rebuild
     the actual mean-variance weights and measure how concentrated gross
     exposure is across sub-industries: Herfindahl index and the implied
     effective number of groups held.

Usage:
    python scripts/analyze_weight_attribution.py <predictions_dir> <sector_csv> [n_seeds_for_B]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn.portfolio.mean_variance import (  # noqa: E402
    estimate_covariance, optimize_weights, quintile_direction, shrink_signal,
)

SEEDS = [42, 123, 7, 101, 2025]
COV_WINDOW, GROSS_CAP, SHRINK = 30, 2.0, 0.3


def load(path):
    d = np.load(path, allow_pickle=True)
    return (d["tickers"], d["val_dates"], d["val_probs"].astype(np.float64),
            d["rets_hist"].astype(np.float64), d["hist_dates"])


def group_ids(tickers, sector_csv, col="gics_sub_industry"):
    df = pd.read_csv(sector_csv, keep_default_na=False)
    lut = {str(r["ticker"]): str(r[col]) for _, r in df.iterrows()}
    names = sorted({v for v in lut.values() if v.strip()})
    idx = {n: i for i, n in enumerate(names)}
    unk = len(names)
    return np.array([idx.get(lut.get(str(t), ""), unk) for t in tickers]), len(names) + 1


def variance_decomposition(mu, groups, n_groups):
    """Return within-group share of cross-sectional variance of mu."""
    total = mu.var()
    if total <= 0:
        return np.nan
    within_ss, n = 0.0, len(mu)
    for g in range(n_groups):
        m = groups == g
        k = int(m.sum())
        if k > 1:
            within_ss += ((mu[m] - mu[m].mean()) ** 2).sum()
        # singleton groups contribute zero within-variance by construction
    return (within_ss / n) / total


def group_hhi(w, groups, n_groups):
    """Herfindahl of gross exposure across groups, and effective #groups."""
    aw = np.abs(w)
    tot = aw.sum()
    if tot <= 0:
        return np.nan, np.nan
    shares = np.array([aw[groups == g].sum() for g in range(n_groups)]) / tot
    hhi = float((shares ** 2).sum())
    return hhi, (1.0 / hhi if hhi > 0 else np.nan)


def main():
    pred_dir = Path(sys.argv[1])
    sector_csv = sys.argv[2]
    n_seeds_b = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    configs = [("sub-industry", "subind_long_s{}.npz"),
               ("bloomberg", "bloomberg_long_s{}.npz"),
               ("no-graph", "none_long_s{}.npz")]

    print("=" * 88)
    print(" A. SIGNAL SMOOTHING — within-sub-industry share of mu variance (n=5 seeds)")
    print("=" * 88)
    print("    lower share  =>  predictions homogeneous inside peer groups")
    print(f"\n  {'signal':<16}{'within-group share':>22}{'std':>10}")
    print("  " + "-" * 48)
    shares = {}
    for name, pat in configs:
        vals = []
        for s in SEEDS:
            p = pred_dir / pat.format(s)
            if not p.exists():
                continue
            tickers, _, probs, _, _ = load(p)
            g, ng = group_ids(tickers, sector_csv)
            mu = probs[..., 2] - probs[..., 0]
            vals.append(np.nanmean([variance_decomposition(mu[k], g, ng)
                                    for k in range(mu.shape[0])]))
        if vals:
            shares[name] = vals
            print(f"  {name:<16}{np.mean(vals):>22.4f}{np.std(vals, ddof=1):>10.4f}")

    if "no-graph" in shares:
        for name in ("sub-industry", "bloomberg"):
            if name in shares:
                d = np.mean(shares[name]) - np.mean(shares["no-graph"])
                print(f"\n  {name} vs no-graph: {d:+.4f} "
                      f"({'MORE' if d < 0 else 'LESS'} within-group homogenisation)")

    print("\n" + "=" * 88)
    print(f" B. PORTFOLIO CONCENTRATION — mean-variance weights (n={n_seeds_b} seeds)")
    print("=" * 88)
    print(f"\n  {'signal':<16}{'group HHI':>12}{'eff. #groups':>15}{'max grp wt':>13}")
    print("  " + "-" * 56)
    for name, pat in configs:
        hhis, effs, maxw = [], [], []
        for s in SEEDS[:n_seeds_b]:
            p = pred_dir / pat.format(s)
            if not p.exists():
                continue
            tickers, val_dates, probs, rets_hist, hist_dates = load(p)
            g, ng = group_ids(tickers, sector_csv)
            hd = np.array(hist_dates, dtype="datetime64[D]")
            vd = np.array(val_dates, dtype="datetime64[D]")
            mu_all = probs[..., 2] - probs[..., 0]
            for k in range(mu_all.shape[0]):
                i = int(np.searchsorted(hd, vd[k]))
                rw = rets_hist[max(0, i - COV_WINDOW):i]
                if rw.shape[0] < 2:
                    continue
                mu = shrink_signal(mu_all[k], SHRINK)
                try:
                    w = optimize_weights(mu, estimate_covariance(rw), risk_aversion=1.0,
                                         gross_cap=GROSS_CAP, dollar_neutral=True,
                                         allowed_sign=quintile_direction(mu))
                except Exception:
                    continue
                h, e = group_hhi(w, g, ng)
                if not np.isnan(h):
                    hhis.append(h); effs.append(e)
                    aw = np.abs(w); tot = aw.sum()
                    if tot > 0:
                        maxw.append(max(aw[g == q].sum() for q in range(ng)) / tot)
        if hhis:
            print(f"  {name:<16}{np.mean(hhis):>12.4f}{np.mean(effs):>15.1f}{np.mean(maxw):>13.1%}")

    print("\n  Higher HHI / fewer effective groups => the book is a concentrated bet")
    print("  on a handful of sub-industries rather than a diversified stock selection.")


if __name__ == "__main__":
    main()
