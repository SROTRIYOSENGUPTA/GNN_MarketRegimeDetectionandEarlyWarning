"""
Recency diagnostic: is the Bloomberg-edge advantage a look-ahead artifact?

The workbook's supplier/customer/holder metadata is a SINGLE static snapshot
(500 metadata rows for 500 tickers, all parked on early-Jan-2015 dates that are
spreadsheet layout artifacts) applied across the whole 2015-2024 panel. Since
the pull happened ~2025/26, look-ahead severity scales with distance from the
snapshot:

    eval 2017 -> ~9 years of look-ahead
    eval 2019 -> ~7 years
    eval 2021 -> ~5 years
    eval 2024 -> ~2 years

Each period trains only on data preceding it (expanding window, cutoff ~15
months before period end). Within a period, `bloomberg` and `none` see
IDENTICAL training data, so the bloomberg-minus-none delta is a fair
within-period comparison even though absolute F1 is not comparable across
periods (training set size and market conditions differ).

Interpretation:
  - delta DECAYS toward recent periods -> effect is largely look-ahead artifact
  - delta STABLE or strongest in recent periods -> look-ahead is not driving it

Usage:
    python scripts/analyze_recency.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results" / "recency"

PERIODS = ["2017", "2019", "2021", "2024"]
LOOKAHEAD = {"2017": "~9 yr", "2019": "~7 yr", "2021": "~5 yr", "2024": "~2 yr"}
SEEDS = [42, 123, 7]
KEY = "val_per_date_macro_f1"


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((x - m) ** 2 for x in vals) / max(n - 1, 1)
    return m, math.sqrt(var)


def t_p(abs_t, df=2):
    table = {2: [(1.89, "<.20"), (2.92, "<.10"), (4.30, "<.05"), (9.92, "<.01")]}
    res = "n.s."
    for thr, lbl in table.get(df, []):
        if abs_t >= thr:
            res = lbl
    return res


def load(mode, period, seed):
    return json.load(open(RES / f"{mode}_p{period}_s{seed}.json"))["best"][KEY]


def main():
    print("=" * 88)
    print(" RECENCY DIAGNOSTIC — per-date macro-F1, bloomberg vs none, n=3 seeds per period")
    print("=" * 88)
    header = (f"{'eval period':<14}{'look-ahead':<12}{'BBG':>16}{'None':>16}"
              f"{'delta':>10}{'t':>9}{'p':>8}")
    print(header)
    print("-" * len(header))

    deltas = []
    for p in PERIODS:
        bbg = [load("bloomberg", p, s) for s in SEEDS]
        none = [load("none", p, s) for s in SEEDS]
        bm, bs = stats(bbg)
        nm, ns = stats(none)
        diffs = [b - n for b, n in zip(bbg, none)]
        dm, ds = stats(diffs)
        t = dm / (ds / math.sqrt(len(diffs))) if ds > 0 else float("inf")
        deltas.append((p, dm))
        print(f"{p:<14}{LOOKAHEAD[p]:<12}{bm:>9.4f}±{bs:<6.4f}{nm:>9.4f}±{ns:<6.4f}"
              f"{dm:>+10.4f}{t:>+9.2f}{t_p(abs(t)):>8}")

    print()
    print("delta trend (most -> least look-ahead):")
    print("  " + "  ->  ".join(f"{p}: {d:+.4f}" for p, d in deltas))
    print()
    worst, best = deltas[0][1], deltas[-1][1]
    if best > worst:
        print("  The advantage is LARGER in the period with the LEAST look-ahead.")
        print("  A look-ahead artifact would predict the opposite ordering, so the")
        print("  static-snapshot contamination does not appear to be driving the effect.")
    else:
        print("  The advantage is larger where look-ahead is worst — consistent with")
        print("  contamination inflating the effect. Treat the headline result as suspect")
        print("  until the graph is rebuilt point-in-time.")


if __name__ == "__main__":
    main()
