"""
Per-stock 20-day forward drawdown prediction.

Task: y[i,t] = 1 if min(forward 5-to-20-day return of stock i from day t) < -threshold.

This script is self-contained — it shares only the workbook loader with the
existing repo. It builds:

    - per-stock features (37-d, same as workbook script)
    - daily heterogeneous graphs (correlation + holder + supplier-customer)
    - per-stock binary labels (forward drawdown indicator)
    - a small per-stock GNN/LSTM model that outputs a logit per stock per day

Then trains + evaluates with macro-AUC across stocks.

Edge mode controls which non-correlation edges are present:
    bloomberg : real holder Jaccard + Bloomberg supplier/customer
    proxy     : sector-based co-holding + synthetic supply chain
    corronly  : only correlation edges
    none      : no graph at all → LSTM-only baseline (no message passing)
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (workbook)
# ─────────────────────────────────────────────────────────────────────────────
def _split_list(s):
    if pd.isna(s):
        return []
    return [t.strip() for t in re.split(r"[;,]", str(s)) if t.strip()]


def _base_ticker(t):
    """First whitespace-separated token, upper-cased. 'FLEX US Equity'→'FLEX'."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return ""
    return str(t).split()[0].upper()


def load_workbook(path):
    sheets = pd.read_excel(path, sheet_name=None)
    rows = pd.concat(sheets.values(), ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"])
    rows["ticker"] = rows["ticker"].astype(str)
    close = rows.pivot_table(index="date", columns="ticker", values="px_last", aggfunc="last").sort_index()
    volume = rows.pivot_table(index="date", columns="ticker", values="px_volume", aggfunc="last").sort_index()
    common = close.columns.intersection(volume.columns)
    close = close[common].ffill().bfill()
    volume = volume[common].fillna(0.0)
    # Metadata: rows where the 'Ticker' (capital T) column is non-null carry the
    # supplier/customer/holder lists. The price 'ticker' (lower-case) column on
    # those rows holds the same Bloomberg code as 'Ticker'.
    meta_src = rows[rows["Ticker"].notna()].copy()
    meta_src["Ticker"] = meta_src["Ticker"].astype(str)
    meta = meta_src.drop_duplicates("Ticker", keep="last").set_index(
        "Ticker"
    )[["Top Suppliers", "Top Customers", "Top 20 Holders"]]
    return close, volume, meta


# ─────────────────────────────────────────────────────────────────────────────
# Edge builders
# ─────────────────────────────────────────────────────────────────────────────
def build_supply_chain(meta, tickers):
    """Map price tickers (e.g. 'AAPL UW') ↔ suppliers/customers (e.g. 'AAPL US Equity').

    We match by the base ticker symbol (first whitespace-token), which is the same
    in both formats.
    """
    N = len(tickers)
    base_to_idx = {_base_ticker(t): i for i, t in enumerate(tickers)}
    adj = np.zeros((N, N), dtype=np.float32)
    for ticker in tickers:
        if ticker not in meta.index:
            continue
        my_idx = base_to_idx[_base_ticker(ticker)]
        row = meta.loc[ticker]
        for sup in _split_list(row.get("Top Suppliers")):
            sb = _base_ticker(sup)
            if sb in base_to_idx and base_to_idx[sb] != my_idx:
                adj[base_to_idx[sb], my_idx] = 1.0
        for cus in _split_list(row.get("Top Customers")):
            cb = _base_ticker(cus)
            if cb in base_to_idx and base_to_idx[cb] != my_idx:
                adj[my_idx, base_to_idx[cb]] = 1.0
    return adj


def build_holder_jaccard(meta, tickers, threshold=0.10):
    """Build holder-overlap adjacency using full institutional names (not tickers)."""
    holder_sets = {}
    for ticker in tickers:
        if ticker not in meta.index:
            holder_sets[ticker] = set()
            continue
        # Holders are entity names like 'Vanguard Group Inc/T'; use raw string,
        # stripped/upper for canonical form.
        raw = meta.loc[ticker].get("Top 20 Holders")
        holder_sets[ticker] = set(h.strip().upper() for h in _split_list(raw))
    N = len(tickers)
    adj = np.zeros((N, N), dtype=np.float32)
    for i, ti in enumerate(tickers):
        si = holder_sets[ti]
        if not si:
            continue
        for j, tj in enumerate(tickers):
            if i == j:
                continue
            sj = holder_sets[tj]
            if not sj:
                continue
            union = si | sj
            inter = si & sj
            jac = len(inter) / len(union)
            if jac >= threshold:
                adj[i, j] = jac
    return adj


def build_proxy_supply(meta, tickers, n_synthetic=2000, seed=0):
    # synthetic random adjacency, same density as real supply chain
    rng = np.random.default_rng(seed)
    N = len(tickers)
    adj = np.zeros((N, N), dtype=np.float32)
    for _ in range(n_synthetic):
        i, j = rng.integers(0, N, size=2)
        if i != j:
            adj[i, j] = 1.0
    return adj


def build_proxy_holder(tickers, sector_groups=11, seed=0):
    rng = np.random.default_rng(seed + 1)
    N = len(tickers)
    sectors = rng.integers(0, sector_groups, size=N)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j and sectors[i] == sectors[j]:
                adj[i, j] = 1.0
    return adj


def correlation_edges_topk(rets_window, top_k=10):
    """Return adjacency from top-K positive corr per node over a returns window."""
    N = rets_window.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(rets_window.T)
    # Replace NaNs (zero-variance stocks) with 0
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, -np.inf)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        idx = np.argsort(-c[i])[:top_k]
        adj[i, idx] = np.clip(c[i, idx], 0, 1)
    return adj


# ─────────────────────────────────────────────────────────────────────────────
# Per-stock features (subset of workbook's 37-d, ~12 most informative)
# ─────────────────────────────────────────────────────────────────────────────
def build_features(close, volume):
    rets = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_ret = rets.mean(axis=1)
    market_vol_20 = market_ret.rolling(20).std().fillna(0.0)
    log_close = np.log1p(close.clip(lower=0.01))
    log_vol = np.log1p(volume.clip(lower=0.0))
    feats = {}  # ticker → (T, F)
    for tk in close.columns:
        cs = close[tk]
        rs = rets[tk]
        df = pd.DataFrame(index=close.index)
        df["log_px"] = log_close[tk]
        df["log_vol"] = log_vol[tk]
        df["ret"] = rs
        df["ret_5"] = cs.pct_change(5).fillna(0.0)
        df["ret_20"] = cs.pct_change(20).fillna(0.0)
        df["ret_60"] = cs.pct_change(60).fillna(0.0)
        df["vol_20"] = rs.rolling(20).std().fillna(0.0)
        df["vol_60"] = rs.rolling(60).std().fillna(0.0)
        df["excess_ret"] = rs - market_ret
        df["mkt_ret"] = market_ret.values
        df["mkt_vol_20"] = market_vol_20.values
        df["abs_ret"] = rs.abs()
        feats[tk] = df.fillna(0.0).astype(np.float32).to_numpy()
    return feats, rets


# ─────────────────────────────────────────────────────────────────────────────
# Per-stock labels: forward 20-day max drawdown indicator
# ─────────────────────────────────────────────────────────────────────────────
def build_labels(close, horizon_min=5, horizon_max=20, threshold=-0.10):
    """y[t, i] = 1 if min_{k in [h_min, h_max]} (close[t+k,i]/close[t,i] - 1) < threshold."""
    arr = close.to_numpy(dtype=np.float64)
    T, N = arr.shape
    labels = np.zeros((T, N), dtype=np.float32)
    for t in range(T - horizon_min):
        end = min(t + horizon_max + 1, T)
        window = arr[t + horizon_min : end]  # (h, N)
        if window.shape[0] == 0:
            continue
        rel = window / arr[t : t + 1] - 1.0
        min_ret = rel.min(axis=0)  # (N,)
        labels[t] = (min_ret < threshold).astype(np.float32)
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class GraphLayer(nn.Module):
    """Simple multi-relational message passing. Each relation has its own linear."""
    def __init__(self, in_dim, out_dim, n_relations):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_rel = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_relations)])

    def forward(self, x, adjs):  # x: (N, F)  adjs: list of (N, N)
        out = self.w_self(x)
        for w, adj in zip(self.w_rel, adjs):
            # row-normalised aggregation
            deg = adj.sum(dim=1, keepdim=True).clamp(min=1.0)
            agg = (adj / deg) @ x
            out = out + w(agg)
        return F.relu(out)


class PerStockGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_relations, seq_len, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        self.feat_proj = nn.Linear(in_dim, hidden)
        if use_graph:
            self.gnn1 = GraphLayer(hidden, hidden, n_relations)
            self.gnn2 = GraphLayer(hidden, hidden, n_relations)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_seq, adjs_seq):
        # x_seq: (T_seq, N, F); adjs_seq: list[T_seq] of list[R] of (N, N)
        T, N, _ = x_seq.shape
        node_t = []
        for t in range(T):
            h = self.feat_proj(x_seq[t])
            if self.use_graph:
                h = self.gnn1(h, adjs_seq[t])
                h = self.gnn2(h, adjs_seq[t])
            node_t.append(h)  # (N, H)
        # Stack: (T, N, H) → (N, T, H) for per-stock LSTM
        node_seq = torch.stack(node_t, dim=0).transpose(0, 1)
        lstm_out, _ = self.lstm(node_seq)  # (N, T, H)
        last = lstm_out[:, -1, :]  # (N, H)
        return self.head(last).squeeze(-1)  # (N,)


# ─────────────────────────────────────────────────────────────────────────────
# Training driver
# ─────────────────────────────────────────────────────────────────────────────
def build_dataset(close, feats, labels, seq_len=30, stride=5):
    """Return list of (target_date_idx, feature_window_TNF)."""
    tickers = list(close.columns)
    N = len(tickers)
    T = len(close)
    feat_arr = np.stack([feats[tk] for tk in tickers], axis=1)  # (T, N, F)
    samples = []
    for t in range(seq_len - 1, T - 25):  # leave room for horizon_max
        samples.append(t)
    samples = samples[::stride]
    return tickers, feat_arr, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="/scratch/ss3414/gnn_regime/sp500_prices 1.xlsx")
    p.add_argument("--output", required=True)
    p.add_argument("--edge-mode", choices=("bloomberg", "proxy", "corronly", "none"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--train-cutoff", default="2022-09-30")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--corr-top-k", type=int, default=10)
    p.add_argument("--corr-window", type=int, default=30)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--drawdown-threshold", type=float, default=-0.10)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[{time.strftime('%H:%M:%S')}] loading workbook...", flush=True)
    close, volume, meta = load_workbook(args.xlsx)
    # restrict date range
    close = close.loc[args.start:args.end]; volume = volume.loc[args.start:args.end]
    tickers = list(close.columns)
    N = len(tickers)
    print(f"  {N} tickers, {len(close)} dates", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] building features...", flush=True)
    feats, rets = build_features(close, volume)

    print(f"[{time.strftime('%H:%M:%S')}] building labels (forward drawdown <= {args.drawdown_threshold})...", flush=True)
    labels = build_labels(close, threshold=args.drawdown_threshold)
    pos_rate_train = labels[:int(len(close)*0.75)].mean()
    pos_rate_val = labels[int(len(close)*0.75):].mean()
    print(f"  pos rate (train/val approx): {pos_rate_train:.3f} / {pos_rate_val:.3f}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] building static edges...", flush=True)
    if args.edge_mode == "bloomberg":
        holder_adj = build_holder_jaccard(meta, tickers)
        supply_adj = build_supply_chain(meta, tickers)
        use_graph = True; n_relations = 3
    elif args.edge_mode == "proxy":
        holder_adj = build_proxy_holder(tickers, seed=args.seed)
        supply_adj = build_proxy_supply(meta, tickers, seed=args.seed)
        use_graph = True; n_relations = 3
    elif args.edge_mode == "corronly":
        holder_adj = np.zeros((N, N), dtype=np.float32)
        supply_adj = np.zeros((N, N), dtype=np.float32)
        use_graph = True; n_relations = 1   # only correlation
    else:  # none
        holder_adj = np.zeros((N, N), dtype=np.float32)
        supply_adj = np.zeros((N, N), dtype=np.float32)
        use_graph = False; n_relations = 0
    print(f"  holder edges: {(holder_adj > 0).sum()}, supply edges: {(supply_adj > 0).sum()}", flush=True)
    holder_t = torch.tensor(holder_adj, device=device)
    supply_t = torch.tensor(supply_adj, device=device)

    print(f"[{time.strftime('%H:%M:%S')}] building dataset samples...", flush=True)
    tickers, feat_arr, samples = build_dataset(close, feats, labels, seq_len=args.seq_len, stride=args.stride)
    print(f"  {len(samples)} samples, feat_arr={feat_arr.shape}", flush=True)
    feat_dim = feat_arr.shape[2]
    rets_arr = rets.to_numpy().astype(np.float32)

    cutoff_idx = close.index.get_indexer([pd.Timestamp(args.train_cutoff)], method="nearest")[0]
    train_samples = [t for t in samples if t <= cutoff_idx]
    val_samples = [t for t in samples if t > cutoff_idx]
    print(f"  train: {len(train_samples)}  val: {len(val_samples)}", flush=True)

    in_dim = feat_dim
    model = PerStockGNN(in_dim, args.hidden, n_relations, args.seq_len, use_graph=use_graph).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"[{time.strftime('%H:%M:%S')}] model: {n_params:,} params, use_graph={use_graph}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    pos_weight = torch.tensor([max((1 - pos_rate_train) / max(pos_rate_train, 1e-3), 1.0)], device=device)
    print(f"  pos_weight={pos_weight.item():.2f}", flush=True)

    def make_adjs(t):
        adjs = []
        # correlation top-K over last corr_window days
        rw = rets_arr[max(0, t - args.corr_window + 1) : t + 1]
        if rw.shape[0] < 2:
            corr_adj = np.zeros((N, N), dtype=np.float32)
        else:
            corr_adj = correlation_edges_topk(rw, top_k=args.corr_top_k)
        adjs.append(torch.tensor(corr_adj, device=device))
        if args.edge_mode in ("bloomberg", "proxy"):
            adjs.append(holder_t)
            adjs.append(supply_t)
        return adjs

    def run_epoch(samples_list, train_mode):
        if train_mode:
            model.train()
        else:
            model.eval()
        rng = np.random.default_rng(args.seed)
        order = list(samples_list)
        if train_mode:
            rng.shuffle(order)
        total_loss, total_n = 0.0, 0
        all_pred, all_true = [], []
        for t in order:
            seq = feat_arr[t - args.seq_len + 1 : t + 1]  # (seq_len, N, F)
            seq_t = torch.tensor(seq, device=device)
            # adjacency snapshot is the same for all seq days (static + corr at t)
            # for efficiency we use the day-t correlation for all T_seq frames
            adjs = make_adjs(t)
            adjs_seq = [adjs] * args.seq_len if use_graph else [None] * args.seq_len
            y = torch.tensor(labels[t], device=device)
            with torch.set_grad_enabled(train_mode):
                logits = model(seq_t, adjs_seq)
                loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
                if train_mode:
                    opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item()) * N
            total_n += N
            with torch.no_grad():
                all_pred.append(torch.sigmoid(logits).cpu().numpy())
                all_true.append(y.cpu().numpy())
        pred = np.concatenate(all_pred); true = np.concatenate(all_true)
        return total_loss / total_n, pred, true

    history = []
    for ep in range(args.epochs):
        t0 = time.time()
        train_loss, _, _ = run_epoch(train_samples, True)
        val_loss, val_pred, val_true = run_epoch(val_samples, False)
        # Metrics
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auc_micro = roc_auc_score(val_true, val_pred)
        except ValueError:
            auc_micro = float("nan")
        try:
            ap_micro = average_precision_score(val_true, val_pred)
        except ValueError:
            ap_micro = float("nan")
        per_stock_auc = []
        for i in range(N):
            yi = val_true.reshape(-1, N)[:, i]
            pi = val_pred.reshape(-1, N)[:, i]
            if yi.sum() > 0 and yi.sum() < len(yi):
                per_stock_auc.append(roc_auc_score(yi, pi))
        macro_auc = float(np.mean(per_stock_auc)) if per_stock_auc else float("nan")
        history.append({
            "epoch": ep + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc_micro": float(auc_micro),
            "val_ap_micro": float(ap_micro),
            "val_macro_auc": macro_auc,
            "elapsed_s": time.time() - t0,
        })
        print(f"[ep {ep+1}/{args.epochs}] train={train_loss:.4f} val={val_loss:.4f} "
              f"AUC_micro={auc_micro:.4f} AUC_macro={macro_auc:.4f} AP={ap_micro:.4f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    result = {
        "edge_mode": args.edge_mode,
        "seed": args.seed,
        "n_params": n_params,
        "n_stocks": N,
        "n_train_samples": len(train_samples),
        "n_val_samples": len(val_samples),
        "pos_rate_train": float(pos_rate_train),
        "pos_rate_val": float(pos_rate_val),
        "history": history,
        "final": history[-1],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
