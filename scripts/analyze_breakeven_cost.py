"""
Breakeven transaction costs for the Sprint 1 portfolio grid.

Motivation. The paper's economic conclusion rests on a single flat assumption:
5 bp per unit of turnover. A practitioner's first question is how sensitive
that conclusion is to the number, and the honest answer is a breakeven, not a
sensitivity band: at what cost does this stop working, and how does that
compare with what I actually pay?

Three breakevens, per cell of the 3 horizons x 6 constructions grid:

  c_gross   cost at which the book's mean net return hits zero -- gross alpha
            exactly pays for the trading.
  c_cer     cost at which the certainty-equivalent return hits zero, i.e. the
            risk-adjusted breakeven at gamma = 5. Always below c_gross, by the
            variance penalty.
  c_delta   cost at which the graph's CER advantage over no-graph vanishes.
            A NEGATIVE value is the informative case: it means the graph is
            behind even at zero cost, so no execution improvement can rescue
            it and the deficit is informational rather than transactional.

Method. The engine computes per-period net return as

    r_k = gross_k - c * turn_k                          (linear in cost)
    sharpe = mean(r)/sd(r) * sqrt(ppy),  ppy = 252/H
    cer    = (mean(r) - 0.5*gamma*var(r)) * ppy
    turnover_yr = mean(turn) * ppy

so (sharpe, cer, turnover_yr, H) determines (mean, sd) exactly. Eliminating
mean between the sharpe and cer definitions gives a quadratic in sd,

    0.5*gamma*sd^2 - (sharpe/sqrt(ppy))*sd + cer/ppy = 0,

whose positive root is the period volatility. Shifting cost from c0 to c moves
the mean by (c0-c)*turn_per_period and leaves the rest in place, which yields

    c_cer   = c0 + cer / turnover_yr
    c_gross = c0 + (mean * ppy) / turnover_yr
    c_delta = c0 + (cer_graph - cer_nograph) / (turn_graph - turn_nograph)

Assumption and why it is safe. This holds return VARIANCE fixed as cost
varies. Strictly, var(gross - c*turn) also carries c^2*var(turn) and a
covariance term. Both are negligible here: the cost term is ~3 bp per period
against a period volatility of ~200-240 bp, so even a +-30% swing in turnover
perturbs it by well under a basis point. The script reports the reconstruction
residuals so this is checkable rather than asserted.

What this does NOT give. Point estimates only. The bootstrap p-values and
window-level t-statistics are not recomputed at each cost level; that needs
the per-period return series in the .npz prediction bundles on the cluster.
Treat the breakevens as exact arithmetic on the published aggregates, and the
inference as unchanged from the 5 bp run.

Usage:  python scripts/analyze_breakeven_cost.py
Writes: results/sprint1/breakeven_cost.json
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONSOLE = REPO / "results" / "sprint1" / "sprint1_console.txt"
OUT = REPO / "results" / "sprint1" / "breakeven_cost.json"

C0 = 5.0 / 1e4          # the cost the grid was run at
GAMMA = 5.0             # must match the engine

ROW = re.compile(
    r"^\s*(\d+)d\s+(\S+)\s+(graph|no-graph)\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


def recover_moments(sharpe, cer, horizon):
    """Return (mean, sd) per period from the reported annualised stats.

    Eliminating mean between sharpe and cer leaves a quadratic in sd. The
    negative root is spurious (volatility is positive); the positive root is
    taken. Returns (nan, nan) if the discriminant is negative, which would
    mean the inputs are mutually inconsistent.
    """
    ppy = 252.0 / horizon
    disc = sharpe * sharpe - 2.0 * GAMMA * cer
    if disc < 0:
        return float("nan"), float("nan")
    sd = (sharpe / math.sqrt(ppy) + math.sqrt(disc / ppy)) / GAMMA
    mean = sharpe * sd / math.sqrt(ppy)
    return mean, sd


def parse(path):
    cells = {}
    for line in path.read_text().splitlines():
        m = ROW.match(line)
        if not m:
            continue
        H, kind, sig, sharpe, cer, mdd, turn = m.groups()
        cells.setdefault((int(H), kind), {})[sig] = dict(
            sharpe=float(sharpe), cer=float(cer),
            max_dd=float(mdd), turnover_yr=float(turn))
    return cells


def main():
    cells = parse(CONSOLE)
    if not cells:
        raise SystemExit(f"no rows parsed from {CONSOLE}")

    out, worst_resid = {}, 0.0
    rows = []
    for (H, kind), sigs in sorted(cells.items()):
        if "graph" not in sigs or "no-graph" not in sigs:
            continue
        ppy = 252.0 / H
        rec = {}
        for sig, d in sigs.items():
            mean, sd = recover_moments(d["sharpe"], d["cer"], H)
            # reconstruction residuals -- proves the inversion, not asserts it
            rs = (mean / sd * math.sqrt(ppy)) if sd > 0 else float("nan")
            rc = (mean - 0.5 * GAMMA * sd * sd) * ppy
            worst_resid = max(worst_resid,
                              abs(rs - d["sharpe"]), abs(rc - d["cer"]))
            turn_yr = d["turnover_yr"]
            rec[sig] = dict(
                d,                      # sharpe / cer / max_dd / turnover_yr
                mean_period=mean, sd_period=sd, ann_mean=mean * ppy,
                c_cer=C0 + d["cer"] / turn_yr,
                c_gross=C0 + (mean * ppy) / turn_yr)
        dturn = rec["graph"]["turnover_yr"] - rec["no-graph"]["turnover_yr"]
        dcer = rec["graph"]["cer"] - rec["no-graph"]["cer"]
        c_delta = C0 + dcer / dturn if dturn != 0 else float("nan")
        out[f"H{H}_{kind}"] = dict(graph=rec["graph"], nograph=rec["no-graph"],
                                   d_cer=dcer, d_turnover=dturn,
                                   c_delta=c_delta)
        rows.append((H, kind, rec, dcer, c_delta))

    print(f"Breakeven costs (bp). Grid run at {C0*1e4:.0f} bp, gamma={GAMMA:g}.")
    print(f"Max reconstruction residual: {worst_resid:.2e}  "
          f"({'exact' if worst_resid < 1e-6 else 'CHECK'})\n")
    hdr = (f"{'horizon':>8}{'construction':>14}{'c_gross g':>11}{'c_cer g':>9}"
           f"{'c_gross n':>11}{'c_cer n':>9}{'c_delta':>10}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for H, kind, rec, dcer, c_delta in rows:
        star = " *" if (H == 20 and kind == "continuous") else ""
        print(f"{H:>7}d{kind:>14}"
              f"{rec['graph']['c_gross']*1e4:>11.1f}"
              f"{rec['graph']['c_cer']*1e4:>9.1f}"
              f"{rec['no-graph']['c_gross']*1e4:>11.1f}"
              f"{rec['no-graph']['c_cer']*1e4:>9.1f}"
              f"{c_delta*1e4:>10.1f}{star}")

    print("\n  c_delta < 0 means the graph trails even at zero cost: the")
    print("  shortfall cannot be executed away. * = pre-registered primary.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        dict(cost_bps_run_at=C0 * 1e4, gamma=GAMMA,
             max_reconstruction_residual=worst_resid, cells=out), indent=2))
    print(f"\n  wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
