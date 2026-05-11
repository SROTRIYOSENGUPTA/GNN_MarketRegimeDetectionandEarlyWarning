"""
Reproduce the main RESULTS.md tables and figures from the JSON results.

Usage:
    python scripts/analyze_results.py            # prints all three tables
    python scripts/analyze_results.py --figures  # also writes PNGs to figures/v2/
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"


def load_final(p: Path, key: str = "val_per_date_macro_f1") -> float:
    return json.load(open(p))["best"][key]


def stats(vs):
    n = len(vs)
    m = sum(vs) / n
    var = sum((x - m) ** 2 for x in vs) / max(n - 1, 1)
    return m, math.sqrt(var)


def t_p(at, df):
    table = {
        2: [(1.89, "<.20"), (2.92, "<.10"), (4.30, "<.05"), (9.92, "<.01")],
        4: [(1.53, "<.20"), (2.13, "<.10"), (2.78, "<.05"), (4.60, "<.01"), (8.61, "<.001")],
    }
    if df not in table:
        return "n.s."
    res = "n.s."
    for thr, lbl in table[df]:
        if at >= thr:
            res = lbl
    return res


def paired_t(a_vals, b_vals):
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    m, s = stats(diffs)
    t = m / (s / math.sqrt(len(diffs))) if s > 0 else float("inf")
    return m, t, t_p(abs(t), df=len(diffs) - 1)


def main_ablation():
    print("=" * 92)
    print(" MAIN ABLATION  —  per-date macro-F1, n=5 seeds, train cutoff 2022-09-30")
    print("=" * 92)
    seeds = (42, 123, 7, 101, 2025)
    modes = ("bloomberg", "proxy", "corronly", "none")
    data = {m: [load_final(RES / "main_ablation" / f"{m}_s{s}.json") for s in seeds] for m in modes}

    header = f"{'config':<14}" + "".join(f"{f's{s}':>10}" for s in seeds) + f"{'mean':>10}{'std':>8}"
    print(header)
    print("-" * len(header))
    for m in modes:
        mm, ss = stats(data[m])
        print(f"{m:<14}" + "".join(f"{v:>10.4f}" for v in data[m]) + f"{mm:>10.4f}{ss:>8.4f}")

    print("\nPaired t-tests (n=5, df=4):")
    for a, b in [("bloomberg", "proxy"), ("bloomberg", "corronly"), ("bloomberg", "none"),
                 ("proxy", "corronly"), ("proxy", "none"), ("corronly", "none")]:
        d, t, p = paired_t(data[a], data[b])
        print(f"  {a:>10} vs {b:<10}  Δ={d:+.4f}  t={t:+5.2f}  p{p}")
    return data


def edge_decomposition():
    print("\n" + "=" * 92)
    print(" EDGE DECOMPOSITION  —  per-date macro-F1, n=3 seeds")
    print("=" * 92)
    seeds = (42, 123, 7)
    modes = ("bloomberg", "supply_only", "holder_only", "corronly", "proxy", "none")
    data = {}
    for m in modes:
        files = []
        for s in seeds:
            if m in ("supply_only", "holder_only"):
                files.append(RES / "edge_decomposition" / f"{m}_s{s}.json")
            else:
                files.append(RES / "main_ablation" / f"{m}_s{s}.json")
        data[m] = [load_final(f) for f in files]

    header = f"{'config':<14}" + "".join(f"{f's{s}':>10}" for s in seeds) + f"{'mean':>10}{'std':>8}"
    print(header)
    print("-" * len(header))
    for m in modes:
        mm, ss = stats(data[m])
        print(f"{m:<14}" + "".join(f"{v:>10.4f}" for v in data[m]) + f"{mm:>10.4f}{ss:>8.4f}")

    print("\nKey paired t-tests (n=3, df=2):")
    for a, b in [("bloomberg", "supply_only"), ("bloomberg", "holder_only"),
                 ("supply_only", "holder_only"),
                 ("supply_only", "none"), ("holder_only", "none"),
                 ("supply_only", "corronly")]:
        d, t, p = paired_t(data[a], data[b])
        print(f"  {a:>12} vs {b:<14}  Δ={d:+.4f}  t={t:+6.2f}  p{p}")
    return data


def walk_forward():
    print("\n" + "=" * 92)
    print(" WALK-FORWARD  —  BBG vs NoGraph at 3 train cutoffs, n=3 seeds")
    print("=" * 92)
    cutoffs = ("2021-09-30", "2022-09-30", "2023-09-30")
    seeds = (42, 123, 7)
    out = {}
    for cut in cutoffs:
        bbg, ng = [], []
        for s in seeds:
            if cut == "2022-09-30":
                bbg.append(load_final(RES / "main_ablation" / f"bloomberg_s{s}.json"))
                ng.append(load_final(RES / "main_ablation" / f"none_s{s}.json"))
            else:
                bbg.append(load_final(RES / "walk_forward" / f"bloomberg_s{s}_cut{cut}.json"))
                ng.append(load_final(RES / "walk_forward" / f"none_s{s}_cut{cut}.json"))
        out[cut] = (bbg, ng)

    print(f"{'cutoff':<14}{'BBG mean ± std':>22}{'None mean ± std':>22}{'Δ':>10}{'t':>10}{'p':>10}")
    print("-" * 88)
    for cut in cutoffs:
        b, n = out[cut]
        bm, bs = stats(b)
        nm, ns = stats(n)
        d, t, p = paired_t(b, n)
        print(f"{cut:<14}{bm:>10.4f} ± {bs:>5.4f}{nm:>10.4f} ± {ns:>5.4f}{d:>+10.4f}{t:>+10.2f}{p:>10}")
    return out


def write_figures():
    """Optional matplotlib figures for the three findings."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = REPO / "figures" / "v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: main ablation bar chart ──
    seeds_all = (42, 123, 7, 101, 2025)
    modes = ("bloomberg", "proxy", "corronly", "none")
    labels = ("Bloomberg\n(holder+supply)", "Proxy\n(sector+synthetic)", "Corr only", "No graph\n(LSTM)")
    data = {m: [load_final(RES / "main_ablation" / f"{m}_s{s}.json") for s in seeds_all] for m in modes}
    means = [stats(data[m])[0] for m in modes]
    stds = [stats(data[m])[1] for m in modes]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(modes))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=["#1f77b4", "#aec7e8", "#aec7e8", "#aec7e8"], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Per-date macro-F1 (val, best epoch)")
    ax.set_title("Main ablation (n=5 seeds): only Bloomberg edges help")
    ax.set_ylim(0.34, 0.385)
    ax.axhline(0.333, color="red", linestyle="--", linewidth=0.8, label="random baseline (1/3)")
    ax.legend(loc="upper right", fontsize=8)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_main_ablation.png", dpi=200)
    plt.close()

    # ── Figure 2: edge decomposition ──
    seeds3 = (42, 123, 7)
    decomp_modes = ("bloomberg", "supply_only", "holder_only", "corronly", "proxy", "none")
    dec_labels = ("Bloomberg\nfull", "Supply\nonly", "Holder\nonly", "Corr\nonly", "Proxy", "No\ngraph")
    dec_data = {}
    for m in decomp_modes:
        files = []
        for s in seeds3:
            files.append(RES / ("edge_decomposition" if m in ("supply_only", "holder_only") else "main_ablation") / f"{m}_s{s}.json")
        dec_data[m] = [load_final(f) for f in files]
    dec_means = [stats(dec_data[m])[0] for m in decomp_modes]
    dec_stds = [stats(dec_data[m])[1] for m in decomp_modes]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#aec7e8", "#aec7e8", "#aec7e8"]

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(decomp_modes))
    bars = ax.bar(x, dec_means, yerr=dec_stds, capsize=4, color=colors, edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(dec_labels, fontsize=9)
    ax.set_ylabel("Per-date macro-F1")
    ax.set_title("Edge decomposition (n=3): supplier/customer edges carry the signal")
    ax.set_ylim(0.32, 0.385)
    ax.axhline(0.333, color="red", linestyle="--", linewidth=0.8, label="random baseline")
    ax.legend(loc="upper right", fontsize=8)
    for bar, v in zip(bars, dec_means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_edge_decomposition.png", dpi=200)
    plt.close()

    # ── Figure 3: walk-forward ──
    cutoffs = ("2021-09-30", "2022-09-30", "2023-09-30")
    bbg_means, bbg_stds, none_means, none_stds = [], [], [], []
    for cut in cutoffs:
        bbg, ng = [], []
        for s in seeds3:
            if cut == "2022-09-30":
                bbg.append(load_final(RES / "main_ablation" / f"bloomberg_s{s}.json"))
                ng.append(load_final(RES / "main_ablation" / f"none_s{s}.json"))
            else:
                bbg.append(load_final(RES / "walk_forward" / f"bloomberg_s{s}_cut{cut}.json"))
                ng.append(load_final(RES / "walk_forward" / f"none_s{s}_cut{cut}.json"))
        bm, bs = stats(bbg); nm, ns = stats(ng)
        bbg_means.append(bm); bbg_stds.append(bs)
        none_means.append(nm); none_stds.append(ns)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(cutoffs)); w = 0.35
    ax.bar(x - w/2, bbg_means, width=w, yerr=bbg_stds, capsize=4, label="Bloomberg GNN", color="#1f77b4", edgecolor="black")
    ax.bar(x + w/2, none_means, width=w, yerr=none_stds, capsize=4, label="No graph (LSTM)", color="#aec7e8", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(cutoffs)
    ax.set_xlabel("Train cutoff")
    ax.set_ylabel("Per-date macro-F1")
    ax.set_title("Walk-forward (n=3 per cutoff): effect direction stable across regimes")
    ax.set_ylim(0.34, 0.38)
    ax.axhline(0.333, color="red", linestyle="--", linewidth=0.8, label="random baseline")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_walk_forward.png", dpi=200)
    plt.close()
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    main_ablation()
    edge_decomposition()
    walk_forward()
    if args.figures:
        write_figures()
