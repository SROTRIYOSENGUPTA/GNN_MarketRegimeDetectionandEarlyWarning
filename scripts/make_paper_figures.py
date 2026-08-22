"""
Generate the paper's figures as vector PDFs.

Four figures, matching the restructured Results section:

  fig1_dissociation   the paper's thesis in one image: macro-F1 improves in
                      7/7 windows while realised long-short spread worsens in
                      6/7. Two panels, shared window axis.
  fig2_granularity    the step-not-gradient result, with edge count on a twin
                      axis to show performance rising as the graph gets
                      SPARSER — i.e. grouping precision, not connectivity.
  fig3_perclass       the mechanism: per-class F1 showing the gain sits in the
                      Neutral class that a quintile book never trades.
  fig4_horizon        holding-period effect: Sharpe up, turnover down ~7x, for
                      both signals.

Provenance. macro-F1 per window is read directly from results/periods/*.json.
Metrics derived from the per-stock prediction bundles (tail-F1, long-short
spread, per-class F1) are not recoverable from the JSONs, so they are held in
the SUMMARY block below, transcribed from:
    results/attribution/per_class_f1.txt
    results/attribution/granularity_metrics.txt
    results/horizon/horizon_backtest.txt
    RESULTS.md Findings 2, 8 and 10
If the underlying runs are regenerated, refresh those files and this block
together.

Output: figures/paper/*.pdf (vector, serif, colourblind-safe, legible in
grayscale via distinct hatches and markers as well as colour).

Usage: python scripts/make_paper_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures" / "paper"

# --- publication style -------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Colourblind-safe (Okabe–Ito). Graph = blue, no-graph = orange, neutral grey.
C_GRAPH, C_NONE, C_GREY = "#0072B2", "#D55E00", "#666666"

WINDOWS = ["2017", "2018", "2019", "2020", "2021", "2022", "2024"]

SUMMARY = {
    # Finding 8 — LS spread delta (graph minus no-graph), bp per rebalance
    "ls_spread_delta": [-14.33, -12.09, -10.02, -12.46, -10.45, +5.69, -3.97],
    # Finding 2 — 10 disjoint seeds
    "granularity": {
        "labels": ["Sector\n(11)", "Industry\ngroup (25)", "Industry\n(67)", "Sub-industry\n(124)"],
        "edges": [25782, 12268, 5904, 3548],
        "tail_f1": [0.2344, 0.2524, 0.2528, 0.2526],
        "tail_sd": [0.0114, 0.0113, 0.0040, 0.0064],
    },
    # Finding 7 — per-class F1, long OOS
    "perclass": {
        "classes": ["Down", "Neutral", "Up"],
        "graph": [0.2579, 0.5852, 0.2355],
        "none":  [0.2298, 0.5472, 0.2774],
    },
    # Finding 10 — non-overlapping horizon backtests
    "horizon": {
        "h": [5, 20, 60],
        "sharpe_graph": [0.368, 0.653, 0.878],
        "sharpe_none":  [0.110, 0.713, 1.491],
        "turn_graph":   [32.7, 11.9, 4.7],
        "turn_none":    [20.4, 7.0, 3.0],
    },
}


def macro_f1_deltas():
    """Read per-window macro-F1 delta (graph minus no-graph) from results JSON."""
    base = REPO / "results" / "periods"
    tags = {"2017": "w2017", "2018": "w2018", "2019": "w2019", "2020": "w2020",
            "2021": "w2021", "2022": "w2022", "2024": "w2024"}
    out = []
    for w in WINDOWS:
        g, n = [], []
        for s in (42, 123, 7):
            for lst, mode in ((g, "subind"), (n, "none")):
                p = base / f"{mode}_{tags[w]}_s{s}.json"
                if p.exists():
                    lst.append(json.load(open(p))["best"]["val_per_date_macro_f1"])
        out.append(np.mean(g) - np.mean(n) if g and n else np.nan)
    return out


def fig1_dissociation():
    macro = macro_f1_deltas()
    spread = SUMMARY["ls_spread_delta"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.5))
    fig.subplots_adjust(wspace=0.34)
    x = np.arange(len(WINDOWS))

    a1.bar(x, macro, color=C_GRAPH, edgecolor="black", linewidth=0.5, width=0.66)
    a1.axhline(0, color="black", lw=0.7)
    a1.set_xticks(x); a1.set_xticklabels(WINDOWS)
    a1.set_ylabel(r"$\Delta$ macro-F1")
    a1.set_title("(a) Classification: better in 7/7 windows", loc="left")
    a1.yaxis.grid(True); a1.set_axisbelow(True)

    cols = [C_GRAPH if v > 0 else C_NONE for v in spread]
    a2.bar(x, spread, color=cols, edgecolor="black", linewidth=0.5, width=0.66, hatch="//")
    a2.axhline(0, color="black", lw=0.7)
    a2.set_xticks(x); a2.set_xticklabels(WINDOWS)
    a2.set_ylabel(r"$\Delta$ long-short spread (bp)", labelpad=2)
    a2.set_title("(b) Tradeable spread: worse in 6/7 windows", loc="left")
    a2.yaxis.grid(True); a2.set_axisbelow(True)

    for a in (a1, a2):
        a.set_xlabel("evaluation window (year ending)")
    fig.savefig(OUT / "fig1_dissociation.pdf")
    plt.close(fig)


def fig2_granularity():
    g = SUMMARY["granularity"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(len(g["labels"]))
    ax.errorbar(x, g["tail_f1"], yerr=g["tail_sd"], marker="o", ms=4.5, lw=1.2,
                color=C_GRAPH, capsize=2.5, capthick=0.7, label="tail-F1 (Down/Up avg)")
    ax.set_xticks(x); ax.set_xticklabels(g["labels"])
    ax.set_ylabel("tail-F1")
    ax.set_xlabel("graph partition (number of groups)")
    ax.yaxis.grid(True); ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, np.array(g["edges"]) / 1000, marker="s", ms=4, lw=1.0, ls="--",
             color=C_GREY, label="same-group edges")
    ax2.set_ylabel("edges (thousands)", color=C_GREY)
    ax2.tick_params(axis="y", colors=C_GREY)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              ncol=2, frameon=False, handlelength=2.2, columnspacing=1.4)
    ax.set_title("Finer grouping, fewer edges, better tails", loc="left")
    fig.savefig(OUT / "fig2_granularity.pdf")
    plt.close(fig)


def fig3_perclass():
    p = SUMMARY["perclass"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, p["graph"], w, label="graph", color=C_GRAPH,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, p["none"], w, label="no graph", color=C_NONE,
           edgecolor="black", linewidth=0.5, hatch="//")
    for i, (a, b) in enumerate(zip(p["graph"], p["none"])):
        d = a - b
        ax.annotate(f"{d:+.3f}", (i, max(a, b) + 0.02), ha="center", fontsize=6.5,
                    color=("black" if d > 0 else C_NONE))
    ax.set_xticks(x); ax.set_xticklabels(p["classes"])
    ax.set_ylabel("per-class F1")
    ax.set_ylim(0, 0.68)
    ax.legend(frameon=False, loc="upper left")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title("The gain sits in the untraded middle", loc="left")
    fig.savefig(OUT / "fig3_perclass.pdf")
    plt.close(fig)


def fig4_horizon():
    h = SUMMARY["horizon"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    x = np.arange(len(h["h"])); w = 0.36

    a1.bar(x - w/2, h["sharpe_graph"], w, label="graph", color=C_GRAPH,
           edgecolor="black", linewidth=0.5)
    a1.bar(x + w/2, h["sharpe_none"], w, label="no graph", color=C_NONE,
           edgecolor="black", linewidth=0.5, hatch="//")
    a1.set_xticks(x); a1.set_xticklabels([f"{v}d" for v in h["h"]])
    a1.set_ylabel("Sharpe"); a1.set_xlabel("holding period")
    a1.set_title("(a) Performance rises with holding period", loc="left")
    a1.legend(frameon=False, loc="upper left")
    a1.yaxis.grid(True); a1.set_axisbelow(True)

    a2.plot(x, h["turn_graph"], marker="o", ms=4.5, lw=1.2, color=C_GRAPH, label="graph")
    a2.plot(x, h["turn_none"], marker="s", ms=4, lw=1.2, ls="--", color=C_NONE, label="no graph")
    a2.set_xticks(x); a2.set_xticklabels([f"{v}d" for v in h["h"]])
    a2.set_ylabel("annualised turnover"); a2.set_xlabel("holding period")
    a2.set_title(r"(b) Turnover falls ${\sim}7\times$", loc="left")
    a2.legend(frameon=False)
    a2.yaxis.grid(True); a2.set_axisbelow(True)

    fig.savefig(OUT / "fig4_horizon.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig1_dissociation(); fig2_granularity(); fig3_perclass(); fig4_horizon()
    for f in sorted(OUT.glob("*.pdf")):
        print(f"  wrote {f.relative_to(REPO)}  ({f.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
