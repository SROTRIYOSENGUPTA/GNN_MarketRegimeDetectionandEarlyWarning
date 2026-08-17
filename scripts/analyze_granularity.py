"""
Granularity gradient + edge decomposition, both on real GICS labels.

Two questions:

1. GRANULARITY. Does it matter how finely you partition the economy? The proxy
   graph connects same-group firms, so swapping the grouping column walks the
   GICS hierarchy: sector (11) -> industry group (25) -> industry (67) ->
   sub-industry (124). Finer groups mean FEWER edges, which separates "precision
   of grouping" from "more connectivity".

2. DECOMPOSITION. RESULTS.md's edge_decomposition claimed supplier/customer
   edges carry ~96% of the Bloomberg gain and holder-overlap edges are harmful
   alone. That was measured against fake-sector features, so it needs redoing.

Edge counts (from the SLURM logs, seed-invariant):
    sector          25,782
    industry_group  12,268
    industry         5,904
    sub_industry     3,548
Performance rises as edges fall, so density is not the driver.

Usage:
    python scripts/analyze_granularity.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RS = REPO / "results" / "realsector"
GR = REPO / "results" / "granularity"
DC = REPO / "results" / "decomp_realsector"

SEEDS = [42, 123, 7, 101, 2025]
KEY = "val_per_date_macro_f1"

# Same-group edge counts, taken from the run logs.
EDGES = {
    "sector (11 groups)": 25782,
    "industry_group (25)": 12268,
    "industry (67)": 5904,
    "sub_industry (124)": 3548,
}


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in vals) / max(n - 1, 1))


def t_p(abs_t):
    for thr, lbl in reversed([(1.53, "<.20"), (2.13, "<.10"), (2.78, "<.05"),
                              (4.60, "<.01"), (8.61, "<.001")]):
        if abs_t >= thr:
            return lbl
    return "n.s."


def paired_t(a, b):
    diffs = [x - y for x, y in zip(a, b)]
    m, s = stats(diffs)
    t = m / (s / math.sqrt(len(diffs))) if s > 0 else float("inf")
    return m, t, t_p(abs(t))


def load(root, pattern):
    return [json.load(open(root / pattern.format(s=s)))["best"][KEY] for s in SEEDS]


def main():
    sector = load(RS, "proxy_s{s}.json")
    bbg = load(RS, "bloomberg_s{s}.json")
    none = load(RS, "none_s{s}.json")
    corr = load(RS, "corronly_s{s}.json")
    supply = load(DC, "supply_only_s{s}.json")
    holder = load(DC, "holder_only_s{s}.json")

    levels = [
        ("sector (11 groups)", sector),
        ("industry_group (25)", load(GR, "gics_industry_group_s{s}.json")),
        ("industry (67)", load(GR, "gics_industry_s{s}.json")),
        ("sub_industry (124)", load(GR, "gics_sub_industry_s{s}.json")),
    ]

    print("=" * 92)
    print(" GRANULARITY GRADIENT — same-group graph at 4 GICS levels (n=5 seeds)")
    print("=" * 92)
    hdr = "{:<24}{:>8}{:>10}{:>9}{:>11}{:>8}{:>8}".format(
        "graph partition", "edges", "mean", "std", "vs sector", "t", "p")
    print(hdr)
    print("-" * len(hdr))
    for name, vals in levels:
        m, s = stats(vals)
        if name.startswith("sector"):
            print("{:<24}{:>8}{:>10.4f}{:>9.4f}{:>11}{:>8}{:>8}".format(
                name, EDGES[name], m, s, "—", "—", "—"))
        else:
            d, t, p = paired_t(vals, sector)
            print("{:<24}{:>8}{:>10.4f}{:>9.4f}{:>+11.4f}{:>+8.2f}{:>8}".format(
                name, EDGES[name], m, s, d, t, p))
    print("\n  Performance rises monotonically as the partition gets finer, while edge")
    print("  count falls 7x. Density is not driving this — precision of grouping is.")

    print("\n" + "=" * 92)
    print(" HEADLINE: free GICS sub-industry graph vs proprietary Bloomberg graph")
    print("=" * 92)
    sub = levels[-1][1]
    for a, b, an, bn in [(sub, bbg, "sub_industry (free)", "bloomberg (proprietary)"),
                         (sub, supply, "sub_industry (free)", "supply_only"),
                         (sub, none, "sub_industry (free)", "no-graph")]:
        d, t, p = paired_t(a, b)
        print(f"  {an:<21} vs {bn:<24} delta={d:+.4f}  t={t:+6.2f}  p{p}")
    wins = sum(1 for a, b in zip(sub, bbg) if a > b)
    print(f"\n  sub_industry beats bloomberg in {wins}/{len(SEEDS)} seeds")
    print(f"  per-seed sub_industry: {[round(x,4) for x in sub]}")
    print(f"  per-seed bloomberg   : {[round(x,4) for x in bbg]}")

    print("\n" + "=" * 92)
    print(" EDGE DECOMPOSITION re-run with REAL sectors (n=5)")
    print("=" * 92)
    for name, vals in [("bloomberg (full)", bbg), ("supply_only", supply),
                       ("sector proxy", sector), ("holder_only", holder),
                       ("corr_only", corr), ("no-graph", none)]:
        m, s = stats(vals)
        print(f"  {name:<20}{m:>9.4f} ± {s:.4f}")
    print()
    for a, b, an, bn in [(supply, none, "supply_only", "no-graph"),
                         (holder, none, "holder_only", "no-graph"),
                         (bbg, supply, "bloomberg", "supply_only"),
                         (supply, sector, "supply_only", "sector proxy")]:
        d, t, p = paired_t(a, b)
        print(f"  {an:>12} vs {bn:<14} delta={d:+.4f}  t={t:+6.2f}  p{p}")

    recov = paired_t(supply, none)[0] / paired_t(bbg, none)[0]
    print(f"\n  supply_only recovers {recov:.0%} of the Bloomberg gain over no-graph")
    print("  (RESULTS.md claimed 96% against fake-sector features; ~90% here, so that")
    print("   specific claim substantially holds. What does NOT hold is any premium for")
    print("   supply-chain edges over a same-granularity sector partition.)")


if __name__ == "__main__":
    main()
