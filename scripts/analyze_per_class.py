"""
Where does the graph models' accuracy advantage actually live?

macro-F1 averages Down/Neutral/Up equally, but a quintile long-short book
only ever touches the tails. If the graph signals' gain sits in the Neutral
class -- 60% of the probability mass, and irrelevant to the portfolio --
that reconciles "ranks better" with "trades worse" with no exotic mechanism.

Computed from the saved prediction bundles; true labels are reconstructed
from val_fwd_ret with the same rank-quantile rule as build_xsec_labels
(bottom 20% Down, middle 60% Neutral, top 20% Up), so nothing is retrained.

Also reports the metrics the portfolio actually cares about:
  - top/bottom quintile PRECISION by the model's own mu ranking
  - the realised long-short SPREAD those picks earn, in bp per period

Usage: python scripts/analyze_per_class.py <predictions_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 7, 101, 2025]
CONFIGS = [("sub-industry", "subind_long_s{}.npz"),
           ("bloomberg", "bloomberg_long_s{}.npz"),
           ("no-graph", "none_long_s{}.npz")]


def stats(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def true_labels(fwd):
    """Rank-quantile labels per date: 0=Down, 1=Neutral, 2=Up, -1=invalid."""
    lab = np.full(fwd.shape, -1, dtype=np.int64)
    for t in range(fwd.shape[0]):
        row = fwd[t]
        ok = np.isfinite(row)
        if ok.sum() < 10:
            continue
        r = np.full(row.shape, np.nan)
        order = np.argsort(np.argsort(row[ok]))
        r[ok] = order / max(ok.sum() - 1, 1)
        lab[t, ok & (r < 0.20)] = 0
        lab[t, ok & (r >= 0.20) & (r < 0.80)] = 1
        lab[t, ok & (r >= 0.80)] = 2
    return lab


def f1_per_class(y, p):
    out = []
    for c in range(3):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return out


def main():
    pred_dir = Path(sys.argv[1])
    print("=" * 90, flush=True)
    print(" PER-CLASS F1 — where does the graph advantage live? (long OOS, n=5 seeds)", flush=True)
    print("=" * 90, flush=True)
    print(f"\n  {'signal':<15}{'Down F1':>10}{'Neutral F1':>13}{'Up F1':>9}"
          f"{'macro':>9}{'tails avg':>11}", flush=True)
    print("  " + "-" * 67, flush=True)

    agg = {}
    for name, pat in CONFIGS:
        rows = []
        for s in SEEDS:
            p = pred_dir / pat.format(s)
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            probs = d["val_probs"].astype(np.float64)
            y = true_labels(d["val_fwd_ret"].astype(np.float64))
            pred = probs.argmax(-1)
            m = y >= 0
            rows.append(f1_per_class(y[m], pred[m]))
        if not rows:
            continue
        a = np.array(rows)
        agg[name] = a
        dn, nt, up = a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean()
        print(f"  {name:<15}{dn:>10.4f}{nt:>13.4f}{up:>9.4f}"
              f"{a.mean(axis=1).mean():>9.4f}{(dn+up)/2:>11.4f}", flush=True)

    if "no-graph" in agg:
        print("\n  Deltas vs no-graph (positive = graph model better):", flush=True)
        base = agg["no-graph"]
        for name in ("sub-industry", "bloomberg"):
            if name not in agg:
                continue
            d = agg[name] - base
            print(f"    {name:<14} Down {d[:,0].mean():+.4f}   Neutral {d[:,1].mean():+.4f}"
                  f"   Up {d[:,2].mean():+.4f}   tails {(d[:,0].mean()+d[:,2].mean())/2:+.4f}",
                  flush=True)

    print("\n" + "=" * 90, flush=True)
    print(" WHAT THE PORTFOLIO ACTUALLY TRADES — quintile precision and realised spread", flush=True)
    print("=" * 90, flush=True)
    print(f"\n  {'signal':<15}{'top-Q prec':>12}{'bot-Q prec':>12}{'LS spread/period':>19}", flush=True)
    print("  " + "-" * 58, flush=True)
    for name, pat in CONFIGS:
        tp_, bp_, sp_ = [], [], []
        for s in SEEDS:
            p = pred_dir / pat.format(s)
            if not p.exists():
                continue
            d = np.load(p, allow_pickle=True)
            probs = d["val_probs"].astype(np.float64)
            fwd = d["val_fwd_ret"].astype(np.float64)
            y = true_labels(fwd)
            mu = probs[..., 2] - probs[..., 0]
            tps, bps, sps = [], [], []
            for t in range(mu.shape[0]):
                ok = np.isfinite(fwd[t]) & (y[t] >= 0)
                if ok.sum() < 20:
                    continue
                m = mu[t].copy()
                m[~ok] = -np.inf
                k = max(int(ok.sum() * 0.2), 1)
                top = np.argsort(-m)[:k]
                m2 = mu[t].copy(); m2[~ok] = np.inf
                bot = np.argsort(m2)[:k]
                tps.append((y[t][top] == 2).mean())
                bps.append((y[t][bot] == 0).mean())
                sps.append(fwd[t][top].mean() - fwd[t][bot].mean())
            tp_.append(np.mean(tps)); bp_.append(np.mean(bps)); sp_.append(np.mean(sps))
        if tp_:
            print(f"  {name:<15}{np.mean(tp_):>12.1%}{np.mean(bp_):>12.1%}"
                  f"{np.mean(sp_)*1e4:>16.1f}bp", flush=True)
    print("\n  LS spread is the raw long-short return the signal's own quintile picks earn,", flush=True)
    print("  before optimisation or costs — the cleanest measure of tradeable edge.", flush=True)


if __name__ == "__main__":
    main()
