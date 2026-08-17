"""
Does the granularity gradient survive when measured on what portfolios trade?

RESULTS.md Finding 2 reports a monotone macro-F1 improvement from sector
(11 groups) to sub-industry (124). But per-class analysis showed the graph
advantage concentrates in the NEUTRAL class -- 60% of the mass, and the one
region a quintile long-short book never touches. So the gradient may be an
artifact of a metric that rewards untradeable accuracy.

This re-measures the same four partitions on three metrics side by side:
  macro-F1     -- the original (Down/Neutral/Up averaged equally)
  tail-F1      -- (Down F1 + Up F1)/2, what a quintile book depends on
  LS spread    -- realised long-short return of the signal's own quintile
                  picks, in bp per rebalance; the direct economic measure

Usage: python scripts/analyze_granularity_metrics.py <predictions_dir>
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SEEDS = [42, 123, 7, 101, 2025]
LEVELS = [("sector (11)", "gran_sector_s{}.npz"),
          ("industry_group (25)", "gran_indgroup_s{}.npz"),
          ("industry (67)", "gran_industry_s{}.npz"),
          ("sub-industry (124)", "subind_s{}.npz"),
          ("no-graph", "none_s{}.npz")]


def stats(v):
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def t_p(a):
    for thr, l in reversed([(1.53, "<.20"), (2.13, "<.10"), (2.78, "<.05"),
                            (4.60, "<.01"), (8.61, "<.001")]):
        if a >= thr:
            return l
    return "n.s."


def paired_t(a, b):
    d = [x - y for x, y in zip(a, b)]
    m, s = stats(d)
    t = m / (s / math.sqrt(len(d))) if s > 0 else float("inf")
    return m, t, t_p(abs(t))


def true_labels(fwd):
    lab = np.full(fwd.shape, -1, dtype=np.int64)
    for t in range(fwd.shape[0]):
        row = fwd[t]
        ok = np.isfinite(row)
        if ok.sum() < 10:
            continue
        r = np.full(row.shape, np.nan)
        r[ok] = np.argsort(np.argsort(row[ok])) / max(ok.sum() - 1, 1)
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
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return out


def metrics(path):
    d = np.load(path, allow_pickle=True)
    probs = d["val_probs"].astype(np.float64)
    fwd = d["val_fwd_ret"].astype(np.float64)
    y = true_labels(fwd)
    pred = probs.argmax(-1)
    m = y >= 0
    dn, nt, up = f1_per_class(y[m], pred[m])
    mu = probs[..., 2] - probs[..., 0]
    spreads = []
    for t in range(mu.shape[0]):
        ok = np.isfinite(fwd[t]) & (y[t] >= 0)
        if ok.sum() < 20:
            continue
        a = mu[t].copy(); a[~ok] = -np.inf
        b = mu[t].copy(); b[~ok] = np.inf
        k = max(int(ok.sum() * 0.2), 1)
        top, bot = np.argsort(-a)[:k], np.argsort(b)[:k]
        spreads.append(fwd[t][top].mean() - fwd[t][bot].mean())
    return dict(macro=(dn + nt + up) / 3, tail=(dn + up) / 2,
                neutral=nt, spread=float(np.mean(spreads)) * 1e4)


def main():
    pd_ = Path(sys.argv[1])
    data = {}
    for name, pat in LEVELS:
        rows = [metrics(pd_ / pat.format(s)) for s in SEEDS if (pd_ / pat.format(s)).exists()]
        if rows:
            data[name] = rows

    print("=" * 92, flush=True)
    print(" GRANULARITY GRADIENT ACROSS THREE METRICS (n=5 seeds, standard OOS)", flush=True)
    print("=" * 92, flush=True)
    print(f"\n  {'partition':<22}{'macro-F1':>18}{'tail-F1':>18}{'LS spread':>16}", flush=True)
    print("  " + "-" * 72, flush=True)
    for name, _ in LEVELS:
        if name not in data:
            continue
        r = data[name]
        ma, ms = stats([x["macro"] for x in r])
        ta, ts = stats([x["tail"] for x in r])
        sa, ss = stats([x["spread"] for x in r])
        print(f"  {name:<22}{ma:>11.4f}±{ms:<6.4f}{ta:>11.4f}±{ts:<6.4f}"
              f"{sa:>10.2f}±{ss:<5.2f}", flush=True)

    if "sector (11)" in data:
        print("\n  Paired tests vs SECTOR (the coarsest partition):", flush=True)
        base = data["sector (11)"]
        for name in ("industry_group (25)", "industry (67)", "sub-industry (124)"):
            if name not in data:
                continue
            out = []
            for key, lbl in (("macro", "macro-F1"), ("tail", "tail-F1"), ("spread", "spread")):
                d, t, p = paired_t([x[key] for x in data[name]], [x[key] for x in base])
                unit = "bp" if key == "spread" else ""
                out.append(f"{lbl} {d:+.4f}{unit} (t={t:+5.2f} p{p})")
            print(f"    {name:<21} " + "   ".join(out), flush=True)

    if "no-graph" in data:
        print("\n  Paired tests vs NO-GRAPH:", flush=True)
        base = data["no-graph"]
        for name, _ in LEVELS:
            if name == "no-graph" or name not in data:
                continue
            out = []
            for key, lbl in (("macro", "macro-F1"), ("tail", "tail-F1"), ("spread", "spread")):
                d, t, p = paired_t([x[key] for x in data[name]], [x[key] for x in base])
                unit = "bp" if key == "spread" else ""
                out.append(f"{lbl} {d:+.4f}{unit} (p{p})")
            print(f"    {name:<21} " + "   ".join(out), flush=True)


if __name__ == "__main__":
    main()
