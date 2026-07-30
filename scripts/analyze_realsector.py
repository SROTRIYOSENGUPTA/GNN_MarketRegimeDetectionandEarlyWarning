"""
Main ablation re-run with REAL sector labels, vs the original fake-sector run.

Why this matters: run_xsec_rank.py previously assigned sectors as
`np.arange(n) % 11` (alphabetical ticker order modulo 11). build_proxy_holder()
connects same-sector pairs, so `--edge-mode proxy` was a RANDOM block graph.
RESULTS.md describes that arm as a "sector-based proxy" and concludes:

    "random proxy edges provide no significant lift over no-graph at all —
     the gain is specifically attributable to the real economic relationships
     encoded in the Bloomberg data."

The first half of that sentence was true of a *random* graph. This script
re-tests it against a genuine sector graph (486/500 tickers, 11 GICS-like
sectors + unknown bucket) and shows the conclusion does not survive.

Usage:
    python scripts/analyze_realsector.py
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NEW = REPO / "results" / "realsector"
OLD = REPO / "results" / "main_ablation"

MODES = ["bloomberg", "proxy", "corronly", "none"]
SEEDS = [42, 123, 7, 101, 2025]
KEY = "val_per_date_macro_f1"


def stats(vals):
    n = len(vals)
    m = sum(vals) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in vals) / max(n - 1, 1))


def t_p(abs_t, df=4):
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


def load(root, mode):
    return [json.load(open(root / f"{mode}_s{s}.json"))["best"][KEY] for s in SEEDS]


def main():
    new = {m: load(NEW, m) for m in MODES}
    old = {m: load(OLD, m) for m in MODES}

    print("=" * 96)
    print(" MAIN ABLATION WITH REAL SECTORS (n=5)   vs   original FAKE-sector run")
    print("=" * 96)
    hdr = (f"{'config':<12}" + "".join(f"{'s'+str(s):>9}" for s in SEEDS)
           + f"{'mean':>9}{'std':>8}{'  | OLD mean':>13}{'delta':>9}")
    print(hdr)
    print("-" * len(hdr))
    for m in MODES:
        mm, ss = stats(new[m])
        om, _ = stats(old[m])
        print(f"{m:<12}" + "".join(f"{v:>9.4f}" for v in new[m])
              + f"{mm:>9.4f}{ss:>8.4f}{om:>13.4f}{mm-om:>+9.4f}")

    print("\nPaired t-tests, REAL sectors (n=5, df=4):")
    for a, b in itertools.combinations(MODES, 2):
        d, t, p = paired_t(new[a], new[b])
        print(f"  {a:>10} vs {b:<10}  delta={d:+.4f}  t={t:+6.2f}  p{p}")

    print("\nSame tests on the ORIGINAL fake-sector run, for comparison:")
    for a, b in [("bloomberg", "proxy"), ("bloomberg", "none"), ("proxy", "none")]:
        d, t, p = paired_t(old[a], old[b])
        print(f"  {a:>10} vs {b:<10}  delta={d:+.4f}  t={t:+6.2f}  p{p}")

    print("\n" + "=" * 96)
    print(" WHAT CHANGED")
    print("=" * 96)
    bp_new = paired_t(new["bloomberg"], new["proxy"])
    bp_old = paired_t(old["bloomberg"], old["proxy"])
    pn_new = paired_t(new["proxy"], new["none"])
    pn_old = paired_t(old["proxy"], old["none"])
    print(f"  bloomberg vs proxy : fake sectors p{bp_old[2]} (t={bp_old[1]:+.2f})"
          f"  ->  real sectors p{bp_new[2]} (t={bp_new[1]:+.2f})")
    print(f"  proxy vs no-graph  : fake sectors p{pn_old[2]} (t={pn_old[1]:+.2f})"
          f"  ->  real sectors p{pn_new[2]} (t={pn_new[1]:+.2f})")
    print()
    print("  RESULTS.md claims proxy edges give 'no significant lift over no-graph'")
    print("  and that the gain is 'specifically attributable to the real economic")
    print("  relationships encoded in the Bloomberg data'. With a genuine sector")
    print("  graph, proxy DOES beat no-graph and Bloomberg's edge over proxy is NOT")
    print("  significant. Most of the graph benefit is sector co-membership, not")
    print("  supply-chain specificity.")


if __name__ == "__main__":
    main()
