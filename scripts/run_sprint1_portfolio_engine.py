"""
Sprint 1 — does the "no economic value" result survive a better portfolio map?

Motivation. Per-class analysis (RESULTS.md Finding 7) showed the graph's
accuracy gain sits almost entirely in the NEUTRAL class. Every portfolio we
have tested so far trades only the tails (quintile long-short), so it
structurally discards the region the graph improves. That makes the earlier
"no economic value" conclusion partly an artefact of the mapping from
prediction to position. This script tests four mappings on identical
predictions, plus the information-content metrics that do not depend on a
portfolio at all.

Pre-registered primary specification, declared before running:
    20-day horizon, continuous weights, sub-industry graph, full sample.
Everything else reported here is exploratory and labelled as such. This
matters because the grid below spans 3 horizons x 4 constructions x 3
thresholds; without a nominated primary spec the exercise is unfalsifiable.

Portfolio constructions
  A quintile      top/bottom 20% by score, equal weight     (current baseline)
  B decile        top/bottom 10%                            (sharper tails)
  C continuous    w_i proportional to score, dollar-neutral, position-capped
                  -- uses the WHOLE cross-section, including Neutral names
  D confidence    trade only names with |score| above a percentile threshold
                  -- directly tests whether Neutral-region information is
                     genuinely useless or merely discarded

Information metrics (portfolio-free)
  Pearson IC, Rank IC (Spearman), ICIR = mean(IC)/sd(IC), and cross-sectional
  incremental R^2 of graph over no-graph.

Risk metrics
  Sharpe, Sortino, max drawdown, turnover, and certainty-equivalent return
  CER = E[R] - (gamma/2)Var(R) at gamma = 5, which prices the variance a
  raw Sharpe comparison hides.

Inference: Newey-West on paired differences, plus a moving-block bootstrap
for the Sharpe difference, which the seed-level paired t-test cannot address.

Usage:
    python scripts/run_sprint1_portfolio_engine.py <predictions_dir> <out_json>
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
CONF_Q = [0.25, 0.50, 0.75]      # fraction of the cross-section EXCLUDED
COST_BPS, GROSS, POS_CAP, GAMMA = 5.0, 2.0, 0.05, 5.0
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- utilities
def nw_t(x, lag=1):
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


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = math.sqrt((rx @ rx) * (ry @ ry))
    return float(rx @ ry / d) if d > 0 else np.nan


def pearson(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    a, b = x[ok] - x[ok].mean(), y[ok] - y[ok].mean()
    d = math.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else np.nan


def block_bootstrap_sharpe_diff(ra, rb, n_boot=2000, block=4):
    """Moving-block bootstrap for H0: Sharpe(a) == Sharpe(b), paired series.

    The two series are resampled with the SAME block indices so their
    contemporaneous dependence is preserved. The null is imposed by centring
    the bootstrap distribution of the difference on zero rather than by
    rescaling the inputs -- an earlier version shifted means to "impose" the
    null, which does not hold Sharpe fixed and produced degenerate p-values.
    """
    ra, rb = np.asarray(ra, float), np.asarray(rb, float)
    n = min(len(ra), len(rb))
    if n < 8:
        return np.nan, np.nan
    ra, rb = ra[:n], rb[:n]

    def sharpe(r):
        s = r.std(ddof=1)
        return r.mean() / s if s > 0 else 0.0

    obs = sharpe(ra) - sharpe(rb)
    nb = max(int(np.ceil(n / block)), 1)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        starts = RNG.integers(0, max(n - block, 1), size=nb)
        idx = np.concatenate([np.arange(t, min(t + block, n)) for t in starts])[:n]
        boot[i] = sharpe(ra[idx]) - sharpe(rb[idx])
    centred = boot - boot.mean()                      # distribution under H0
    p = (np.sum(np.abs(centred) >= abs(obs)) + 1) / (n_boot + 1)
    return float(obs), float(p)


def forward_return(rets, hist_dates, start, horizon):
    hd = np.array(hist_dates, dtype="datetime64[D]")
    i = int(np.searchsorted(hd, np.datetime64(str(start), "D")))
    j = min(i + horizon, rets.shape[0])
    if j - i < max(horizon // 2, 2):
        return None
    return np.nanprod(1.0 + rets[i:j], axis=0) - 1.0


# ------------------------------------------------------------ constructions
def weights(score, kind, ok, conf_q=None):
    w = np.zeros_like(score)
    if ok.sum() < 20:
        return w
    s = np.where(ok, score, np.nan)
    r = np.full(len(s), np.nan)
    r[ok] = np.argsort(np.argsort(s[ok])) / max(ok.sum() - 1, 1)

    if kind in ("quintile", "decile"):
        q = 0.2 if kind == "quintile" else 0.1
        lo, hi = ok & (r < q), ok & (r >= 1 - q)
        if lo.sum() == 0 or hi.sum() == 0:
            return w
        w[hi] = (GROSS / 2) / hi.sum()
        w[lo] = -(GROSS / 2) / lo.sum()
        return w

    if kind == "continuous":
        c = np.where(ok, score - np.nanmean(s), 0.0)      # dollar-neutral by centring
        denom = np.abs(c).sum()
        if denom <= 0:
            return w
        w = GROSS * c / denom
        return np.clip(w, -POS_CAP, POS_CAP)

    if kind == "confidence":
        c = np.where(ok, score - np.nanmean(s), 0.0)
        thr = np.nanquantile(np.abs(c[ok]), conf_q)
        keep = ok & (np.abs(c) >= thr)
        if keep.sum() < 10:
            return w
        cc = np.where(keep, c, 0.0)
        denom = np.abs(cc).sum()
        if denom <= 0:
            return w
        w = GROSS * cc / denom
        return np.clip(w, -POS_CAP, POS_CAP)

    raise ValueError(kind)


def backtest(score_all, val_dates, rets, hist_dates, horizon, kind, conf_q=None):
    step = max(int(round(horizon / 5)), 1)          # non-overlapping holds
    prev = np.zeros(score_all.shape[1])
    rr, tt = [], []
    for k in range(0, len(val_dates), step):
        fwd = forward_return(rets, hist_dates, val_dates[k], horizon)
        if fwd is None:
            continue
        ok = np.isfinite(score_all[k]) & np.isfinite(fwd)
        w = weights(score_all[k], kind, ok, conf_q)
        if not np.any(w):
            continue
        turn = float(np.abs(w - prev).sum())
        gross = float(np.nansum(w * np.where(ok, fwd, 0.0)))
        rr.append(gross - (COST_BPS / 1e4) * turn); tt.append(turn); prev = w
    if len(rr) < 4:
        return None
    r = np.asarray(rr); ppy = 252.0 / horizon
    sd = r.std(ddof=1)
    dn = r[r < 0].std(ddof=1) if (r < 0).sum() > 1 else np.nan
    eq = np.cumprod(1 + r); dd = eq / np.maximum.accumulate(eq) - 1
    return dict(
        n=len(r),
        sharpe=float(r.mean() / sd * math.sqrt(ppy)) if sd > 0 else np.nan,
        sortino=float(r.mean() / dn * math.sqrt(ppy)) if dn and dn > 0 else np.nan,
        max_dd=float(dd.min()),
        turnover_yr=float(np.mean(tt) * ppy),
        cer=float((r.mean() - 0.5 * GAMMA * r.var(ddof=1)) * ppy),
        rets=[float(v) for v in r],
    )


def main():
    pred_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    store = {}

    # ------------------------------- information metrics (portfolio-free)
    print("=" * 96, flush=True)
    print(" 1. INFORMATION CONTENT — portfolio-free (mean over 7 windows x 3 seeds)", flush=True)
    print("=" * 96, flush=True)
    print(f"\n  {'horizon':>8}{'signal':>11}{'IC':>10}{'RankIC':>10}{'ICIR':>9}{'inc.R2':>10}", flush=True)
    print("  " + "-" * 58, flush=True)
    for H in HORIZONS:
        cache = {}
        for tag in ("subind", "none"):
            ics, rics, icirs = [], [], []
            for w in WINDOWS:
                for s in SEEDS:
                    p = pred_dir / f"{tag}_{w}_s{s}.npz"
                    if not p.exists():
                        continue
                    d = np.load(p, allow_pickle=True)
                    pr = d["val_probs"].astype(float)
                    sc = pr[..., 2] - pr[..., 0]
                    fwd = np.array([forward_return(d["rets_hist"].astype(float),
                                                   d["hist_dates"], dt, H)
                                    for dt in d["val_dates"]], dtype=object)
                    per_ic, per_ric = [], []
                    for k, f in enumerate(fwd):
                        if f is None:
                            continue
                        per_ic.append(pearson(sc[k], f)); per_ric.append(spearman(sc[k], f))
                    if per_ic:
                        ics.append(np.nanmean(per_ic)); rics.append(np.nanmean(per_ric))
                        sd = np.nanstd(per_ic, ddof=1)
                        icirs.append(np.nanmean(per_ic) / sd if sd > 0 else np.nan)
            cache[tag] = (np.nanmean(ics), np.nanmean(rics), np.nanmean(icirs))
        for tag, lbl in (("subind", "graph"), ("none", "no-graph")):
            ic, ric, icir = cache[tag]
            inc = (ic ** 2 - cache["none"][0] ** 2) if tag == "subind" else 0.0
            print(f"  {H:>6}d{lbl:>11}{ic:>10.4f}{ric:>10.4f}{icir:>9.3f}"
                  f"{inc:>10.5f}", flush=True)
        store[f"info_H{H}"] = {k: list(map(float, v)) for k, v in cache.items()}

    # ------------------------------- portfolio constructions
    print("\n" + "=" * 96, flush=True)
    print(" 2. PORTFOLIO CONSTRUCTIONS — does a better map recover the value?", flush=True)
    print("=" * 96, flush=True)
    print("\n  PRIMARY (pre-registered): 20d horizon, continuous weights\n", flush=True)
    print(f"  {'horizon':>8}{'construction':>16}{'signal':>11}{'Sharpe':>9}"
          f"{'CER':>9}{'maxDD':>9}{'turn/yr':>9}", flush=True)
    print("  " + "-" * 71, flush=True)

    kinds = [("quintile", None), ("decile", None), ("continuous", None)] + \
            [("confidence", q) for q in CONF_Q]

    for H in HORIZONS:
        for kind, cq in kinds:
            label = kind if cq is None else f"conf@{int(cq*100)}%"
            agg = {}
            for tag in ("subind", "none"):
                res = []
                for w in WINDOWS:
                    for s in SEEDS:
                        p = pred_dir / f"{tag}_{w}_s{s}.npz"
                        if not p.exists():
                            continue
                        d = np.load(p, allow_pickle=True)
                        pr = d["val_probs"].astype(float)
                        sc = pr[..., 2] - pr[..., 0]
                        r = backtest(sc, d["val_dates"], d["rets_hist"].astype(float),
                                     d["hist_dates"], H, kind, cq)
                        if r:
                            res.append(r)
                if res:
                    agg[tag] = res
            if len(agg) < 2:
                continue
            for tag, lbl in (("subind", "graph"), ("none", "no-graph")):
                R = agg[tag]
                star = " *" if (H == 20 and kind == "continuous") else ""
                print(f"  {H:>6}d{label:>16}{lbl:>11}"
                      f"{np.nanmean([x['sharpe'] for x in R]):>9.3f}"
                      f"{np.nanmean([x['cer'] for x in R]):>9.4f}"
                      f"{np.nanmean([x['max_dd'] for x in R]):>9.3f}"
                      f"{np.nanmean([x['turnover_yr'] for x in R]):>9.2f}{star}", flush=True)
            dg = [x["sharpe"] for x in agg["subind"]]
            dn = [x["sharpe"] for x in agg["none"]]
            m = min(len(dg), len(dn))
            dm, dt = nw_t([a - b for a, b in zip(dg[:m], dn[:m])])
            pooled_g = np.concatenate([x["rets"] for x in agg["subind"]])
            pooled_n = np.concatenate([x["rets"] for x in agg["none"]])
            obs, pboot = block_bootstrap_sharpe_diff(pooled_g, pooled_n)
            print(f"  {'':>8}{'  -> graph-nograph':>16}{'':>11}{dm:>9.3f}"
                  f"   t={dt:+.2f}  bootstrap p={pboot:.3f}", flush=True)
            store[f"pf_H{H}_{label}"] = dict(
                d_sharpe=float(dm) if np.isfinite(dm) else None,
                t=float(dt) if np.isfinite(dt) else None,
                boot_p=float(pboot) if np.isfinite(pboot) else None)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(store, indent=2))
    print(f"\n  wrote {out_path}", flush=True)
    print("\n  * = pre-registered primary specification. All other rows are", flush=True)
    print("  exploratory; with 3 horizons x 6 constructions the grid invites", flush=True)
    print("  selection, so treat non-primary results as descriptive only.", flush=True)


if __name__ == "__main__":
    main()
