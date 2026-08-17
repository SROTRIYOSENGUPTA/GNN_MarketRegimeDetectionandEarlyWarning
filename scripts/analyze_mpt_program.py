"""
MPT program: does the classification signal survive portfolio construction?

Four experiments, all on the sub-industry signal (the best classifier from
RESULTS.md Finding 2) unless noted, n=5 seeds:

  1. First portfolio run of the sub-industry signal.
  2. Shrinkage sweep, lambda in [0, 0.95] -- the Michaud (1989)
     error-maximization curve.
  3. Turnover-penalised mean-variance (lambda = 5bps, matching the cost).
  4. Long out-of-sample (2019 cutoff, 383 rebalances) vs the standard
     2022 cutoff (163), to cut Sharpe standard errors.

Config keys in the result JSONs are historical: 'A_bloomberg_*' means "the
main signal passed via --predictions", which here is the sub-industry or
long-OOS signal, not necessarily Bloomberg.

Usage: python scripts/analyze_mpt_program.py
"""
from __future__ import annotations
import json, math
from pathlib import Path

B = Path(__file__).resolve().parent.parent / "results" / "mpt_program"
SEEDS = [42, 123, 7, 101, 2025]
KS = ["0.0", "0.15", "0.5", "0.7", "0.85", "0.95"]
MAIN = "A_bloomberg_gnn_cov_mv"          # main signal, GNN-window cov, MV sized
EW_PICKS = "E_bloomberg_equal_weight_picks"
NONE_MV = "C_none_gnn_cov_mv"


def stats(v):
    n = len(v); m = sum(v) / n
    return m, math.sqrt(sum((x - m) ** 2 for x in v) / max(n - 1, 1))


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


def val(p, cfg, field="sharpe"):
    return json.load(open(p))["configs"][cfg][field]


def main():
    print("=" * 86)
    print(" 1. SUB-INDUSTRY SIGNAL — first portfolio run (standard OOS, n=5)")
    print("=" * 86)
    for cfg, lbl in [(MAIN, "sub-industry mu + GNN cov, MV"),
                     (EW_PICKS, "sub-industry mu, equal-wt picks"),
                     (NONE_MV, "no-graph mu + GNN cov, MV"),
                     ("D_none_sample_cov_mv", "no-graph mu + sample cov, MV"),
                     ("F_equal_weight_universe", "equal-weight universe")]:
        v = [val(B / "subind" / f"s{s}.json", cfg) for s in SEEDS]
        m, sd = stats(v)
        print(f"  {lbl:<34}{m:>8.3f} ± {sd:.3f}")
    sub = [val(B / "subind" / f"s{s}.json", MAIN) for s in SEEDS]
    non = [val(B / "subind" / f"s{s}.json", NONE_MV) for s in SEEDS]
    ew = [val(B / "subind" / f"s{s}.json", EW_PICKS) for s in SEEDS]
    d, t, p = paired_t(sub, non)
    print(f"\n  sub-industry MV vs no-graph MV : {d:+.3f}  t={t:+5.2f}  p{p}")
    d, t, p = paired_t(ew, sub)
    print(f"  equal-wt sizing vs MV sizing   : {d:+.3f}  t={t:+5.2f}  p{p}")
    print(f"  seed-std:  MV {stats(sub)[1]:.3f}  vs  equal-wt {stats(ew)[1]:.3f}")

    print("\n" + "=" * 86)
    print(" 2. SHRINKAGE SWEEP — Michaud error-maximization curve (n=5)")
    print("=" * 86)
    print(f"  {'shrinkage':>10}{'Sharpe mean':>14}{'seed-std':>11}{'avg turnover':>15}")
    print("  " + "-" * 48)
    curve = []
    for k in KS:
        v = [val(B / "shrinkage" / f"s{s}_k{k}.json", MAIN) for s in SEEDS]
        to = [val(B / "shrinkage" / f"s{s}_k{k}.json", MAIN, "avg_turnover") for s in SEEDS]
        m, sd = stats(v); curve.append((k, m, sd))
        print(f"  {k:>10}{m:>14.3f}{sd:>11.3f}{stats(to)[0]:>15.3f}")
    base = [val(B / "shrinkage" / f"s{s}_k0.0.json", MAIN) for s in SEEDS]
    top = [val(B / "shrinkage" / f"s{s}_k0.95.json", MAIN) for s in SEEDS]
    d, t, p = paired_t(top, base)
    print(f"\n  k=0.95 vs k=0.00 : {d:+.3f}  t={t:+5.2f}  p{p}")
    print(f"  seed-std {curve[0][2]:.3f} -> {curve[-1][2]:.3f} "
          f"({curve[-1][2]/curve[0][2]:.0%} of unshrunk) — shrinkage barely bites")

    print("\n" + "=" * 86)
    print(" 3. TURNOVER-PENALISED MV (lambda = 5bps, n=5)")
    print("=" * 86)
    pen = [val(B / "turnover" / f"s{s}.json", MAIN) for s in SEEDS]
    pto = [val(B / "turnover" / f"s{s}.json", MAIN, "avg_turnover") for s in SEEDS]
    raw = [val(B / "subind" / f"s{s}.json", MAIN) for s in SEEDS]
    rto = [val(B / "subind" / f"s{s}.json", MAIN, "avg_turnover") for s in SEEDS]
    print(f"  unpenalised : Sharpe {stats(raw)[0]:+.3f} ± {stats(raw)[1]:.3f}   turnover {stats(rto)[0]:.3f}")
    print(f"  penalised   : Sharpe {stats(pen)[0]:+.3f} ± {stats(pen)[1]:.3f}   turnover {stats(pto)[0]:.3f}")
    d, t, p = paired_t(pen, raw)
    print(f"  delta       : {d:+.3f}  t={t:+5.2f}  p{p}  (penalty cuts turnover but costs return)")

    print("\n" + "=" * 86)
    print(" 4. LONG OOS (2019 cutoff, 383 rebalances) — the powered test")
    print("=" * 86)
    for nm, key in [("sub-industry", "subind"), ("bloomberg", "bloomberg")]:
        v = [val(B / "long" / f"{key}_s{s}.json", MAIN) for s in SEEDS]
        n = [val(B / "long" / f"{key}_s{s}.json", NONE_MV) for s in SEEDS]
        m, sd = stats(v); d, t, p = paired_t(v, n)
        print(f"  {nm:<14} Sharpe {m:+.3f} ± {sd:.3f}  vs no-graph {stats(n)[0]:+.3f}"
              f" : d={d:+.3f} t={t:+5.2f} p{p}")
    print("\n  With ~2.4x the rebalances, both graph signals now UNDERPERFORM the")
    print("  no-graph signal significantly. The classification advantage does not")
    print("  merely fail to transfer — it reverses under portfolio construction.")


if __name__ == "__main__":
    main()
