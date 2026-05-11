"""
Cross-sectional 5-day return rank prediction.

Task: For each date t and stock i, classify into one of three classes based
      on rank of forward 5-day return across all stocks:
        - Down    (bottom 20%)
        - Neutral (middle 60%)
        - Up      (top 20%)

This is a balanced 3-class per-stock classification. Cross-sectional rank
cannot be reverse-engineered from aggregate market features — every stock
on day t has the same market context, so the only signal that distinguishes
ranks is per-stock / inter-stock. That's where GNNs should shine.

Edge modes:
    bloomberg : real holder Jaccard (≥0.4) + supplier/customer + correlation
    proxy     : sector-based holder + synthetic supply + correlation
    corronly  : only correlation
    none      : no graph (per-stock LSTM, but with attention over time)
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


# ─── data loaders (same as run_per_stock.py) ──────────────────────────────────
def _split_list(s):
    if pd.isna(s):
        return []
    return [t.strip() for t in re.split(r"[;,]", str(s)) if t.strip()]


def _base_ticker(t):
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
    meta_src = rows[rows["Ticker"].notna()].copy()
    meta_src["Ticker"] = meta_src["Ticker"].astype(str)
    meta = meta_src.drop_duplicates("Ticker", keep="last").set_index(
        "Ticker"
    )[["Top Suppliers", "Top Customers", "Top 20 Holders"]]
    return close, volume, meta


def build_supply_chain(meta, tickers):
    N = len(tickers)
    base_to_idx = {_base_ticker(t): i for i, t in enumerate(tickers)}
    adj = np.zeros((N, N), dtype=np.float32)
    for ticker in tickers:
        if ticker not in meta.index:
            continue
        my = base_to_idx[_base_ticker(ticker)]
        row = meta.loc[ticker]
        for sup in _split_list(row.get("Top Suppliers")):
            sb = _base_ticker(sup)
            if sb in base_to_idx and base_to_idx[sb] != my:
                adj[base_to_idx[sb], my] = 1.0
        for cus in _split_list(row.get("Top Customers")):
            cb = _base_ticker(cus)
            if cb in base_to_idx and base_to_idx[cb] != my:
                adj[my, base_to_idx[cb]] = 1.0
    return adj


def build_holder_jaccard(meta, tickers, threshold=0.4):
    """Higher threshold → sparser, more meaningful holder overlap."""
    holder_sets = {}
    for ticker in tickers:
        if ticker not in meta.index:
            holder_sets[ticker] = set()
            continue
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


def build_proxy_supply(tickers, n_edges, seed=0):
    rng = np.random.default_rng(seed)
    N = len(tickers)
    adj = np.zeros((N, N), dtype=np.float32)
    for _ in range(n_edges):
        i, j = rng.integers(0, N, size=2)
        if i != j:
            adj[i, j] = 1.0
    return adj


def build_proxy_holder(tickers, sector_assign, seed=0):
    """Sector-based co-holding proxy: stocks in same sector connect."""
    N = len(tickers)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i != j and sector_assign[i] == sector_assign[j]:
                adj[i, j] = 1.0
    return adj


def correlation_edges_topk(rets_window, top_k=10):
    N = rets_window.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.corrcoef(rets_window.T)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(c, -np.inf)
    adj = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        idx = np.argsort(-c[i])[:top_k]
        adj[i, idx] = np.clip(c[i, idx], 0, 1)
    return adj


# ─── features with stock-level variety ────────────────────────────────────────
def build_features(close, volume, n_sectors=11):
    rets = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_ret = rets.mean(axis=1)
    market_vol_20 = market_ret.rolling(20).std().fillna(0.0)
    log_close = np.log1p(close.clip(lower=0.01))
    log_vol = np.log1p(volume.clip(lower=0.0))
    feats = {}
    # Random sector assignment (since workbook lacks GICS) — deterministic by ticker order
    n = close.shape[1]
    sector_assign = np.arange(n) % n_sectors
    for ix, tk in enumerate(close.columns):
        cs = close[tk]; rs = rets[tk]
        df = pd.DataFrame(index=close.index)
        df["log_px"] = log_close[tk]
        df["log_vol"] = log_vol[tk]
        df["ret"] = rs
        df["ret_5"] = cs.pct_change(5).fillna(0.0)
        df["ret_20"] = cs.pct_change(20).fillna(0.0)
        df["vol_20"] = rs.rolling(20).std().fillna(0.0)
        df["vol_60"] = rs.rolling(60).std().fillna(0.0)
        df["excess_ret"] = rs - market_ret
        df["mkt_ret"] = market_ret.values
        df["mkt_vol_20"] = market_vol_20.values
        # Sector one-hot
        for s in range(n_sectors):
            df[f"sect_{s}"] = 1.0 if sector_assign[ix] == s else 0.0
        feats[tk] = df.fillna(0.0).astype(np.float32).to_numpy()
    return feats, rets, sector_assign


def build_xsec_labels(close, horizon=5, low_q=0.20, high_q=0.80):
    """y[t, i] ∈ {0=Down, 1=Neutral, 2=Up} based on rank of forward `horizon`-day return."""
    arr = close.to_numpy(dtype=np.float64)
    T, N = arr.shape
    labels = np.full((T, N), -1, dtype=np.int64)
    for t in range(T - horizon):
        fwd_ret = arr[t + horizon] / arr[t] - 1.0
        if not np.isfinite(fwd_ret).any():
            continue
        valid = np.isfinite(fwd_ret)
        if valid.sum() < 10:
            continue
        ranks = np.full(N, np.nan)
        ranks[valid] = pd.Series(fwd_ret[valid]).rank(pct=True).values
        lab = np.full(N, -1, dtype=np.int64)
        lab[(ranks < low_q) & valid] = 0     # Down
        lab[(ranks >= low_q) & (ranks < high_q) & valid] = 1   # Neutral
        lab[(ranks >= high_q) & valid] = 2   # Up
        labels[t] = lab
    return labels


# ─── model: attention-weighted multi-relational GNN ───────────────────────────
class AttnGraphLayer(nn.Module):
    """Per-relation scaled dot-product attention over the edge-masked graph.

    Memory: O(N^2) per layer for the score matrix only — no (N, N, F) tensor.
    """
    def __init__(self, in_dim, out_dim, n_relations):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_rel = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(n_relations)])
        self.q_rel = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(n_relations)])
        self.k_rel = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(n_relations)])
        self.scale = 1.0 / math.sqrt(in_dim)

    def forward(self, x, adjs):
        # x: (N, F)  adjs: list of (N, N)
        out = self.w_self(x)
        for w, qproj, kproj, adj in zip(self.w_rel, self.q_rel, self.k_rel, adjs):
            mask = (adj > 0)
            if not mask.any():
                continue
            q = qproj(x)               # (N, F)
            k = kproj(x)               # (N, F)
            scores = (q @ k.t()) * self.scale            # (N, N)
            scores = scores.masked_fill(~mask, -1e9)
            a = F.softmax(scores, dim=1)                 # (N, N)
            a = a * mask.float()                         # zero out non-edges to be safe
            agg = a @ x                                  # (N, F)
            out = out + w(agg)
        return F.relu(out)


class XSecRankModel(nn.Module):
    def __init__(self, in_dim, hidden, n_relations, seq_len, use_graph=True, n_classes=3):
        super().__init__()
        self.use_graph = use_graph
        self.feat_proj = nn.Linear(in_dim, hidden)
        if use_graph:
            self.gnn1 = AttnGraphLayer(hidden, hidden, n_relations)
            self.gnn2 = AttnGraphLayer(hidden, hidden, n_relations)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x_seq, adjs):
        T, N, _ = x_seq.shape
        node_t = []
        for t in range(T):
            h = self.feat_proj(x_seq[t])
            if self.use_graph:
                h = self.gnn1(h, adjs)
                h = self.gnn2(h, adjs)
            node_t.append(h)
        seq = torch.stack(node_t, dim=0).transpose(0, 1)  # (N, T, H)
        out, _ = self.lstm(seq)
        last = self.dropout(out[:, -1, :])
        return self.head(last)  # (N, n_classes)


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="/scratch/ss3414/gnn_regime/sp500_prices 1.xlsx")
    p.add_argument("--output", required=True)
    p.add_argument("--edge-mode", choices=("bloomberg", "proxy", "corronly", "none", "holder_only", "supply_only"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--train-cutoff", default="2022-09-30")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--corr-top-k", type=int, default=10)
    p.add_argument("--corr-window", type=int, default=30)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--holder-threshold", type=float, default=0.4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    print(f"[{time.strftime('%H:%M:%S')}] loading workbook…", flush=True)
    close, volume, meta = load_workbook(args.xlsx)
    close = close.loc[args.start:args.end]; volume = volume.loc[args.start:args.end]
    tickers = list(close.columns)
    N = len(tickers)
    print(f"  {N} tickers, {len(close)} dates", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] features…", flush=True)
    feats, rets, sector_assign = build_features(close, volume, n_sectors=11)
    feat_arr = np.stack([feats[tk] for tk in tickers], axis=1)
    feat_dim = feat_arr.shape[2]
    rets_arr = rets.to_numpy().astype(np.float32)

    print(f"[{time.strftime('%H:%M:%S')}] cross-sectional rank labels…", flush=True)
    labels = build_xsec_labels(close, horizon=5)
    valid_mask = labels >= 0
    print(f"  label distribution train (approx): "
          f"down={(labels==0).mean():.3f} neutral={(labels==1).mean():.3f} up={(labels==2).mean():.3f}",
          flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] edges…", flush=True)
    if args.edge_mode == "bloomberg":
        holder_adj = build_holder_jaccard(meta, tickers, threshold=args.holder_threshold)
        supply_adj = build_supply_chain(meta, tickers)
        use_graph = True; n_relations = 3
    elif args.edge_mode == "proxy":
        holder_adj = build_proxy_holder(tickers, sector_assign, seed=args.seed)
        supply_adj = build_proxy_supply(tickers, n_edges=int(build_supply_chain(meta, tickers).sum()), seed=args.seed)
        use_graph = True; n_relations = 3
    elif args.edge_mode == "corronly":
        holder_adj = np.zeros((N, N), dtype=np.float32); supply_adj = np.zeros((N, N), dtype=np.float32)
        use_graph = True; n_relations = 1
    elif args.edge_mode == "holder_only":
        holder_adj = build_holder_jaccard(meta, tickers, threshold=args.holder_threshold)
        supply_adj = np.zeros((N, N), dtype=np.float32)
        use_graph = True; n_relations = 2   # correlation + holder
    elif args.edge_mode == "supply_only":
        holder_adj = np.zeros((N, N), dtype=np.float32)
        supply_adj = build_supply_chain(meta, tickers)
        use_graph = True; n_relations = 2   # correlation + supply
    else:
        holder_adj = np.zeros((N, N), dtype=np.float32); supply_adj = np.zeros((N, N), dtype=np.float32)
        use_graph = False; n_relations = 0
    print(f"  holder edges: {int((holder_adj > 0).sum())}, supply edges: {int((supply_adj > 0).sum())}", flush=True)
    holder_t = torch.tensor(holder_adj, device=device)
    supply_t = torch.tensor(supply_adj, device=device)

    # samples
    T = len(close)
    sample_t = list(range(args.seq_len - 1, T - 6))[:: args.stride]
    cutoff_idx = close.index.get_indexer([pd.Timestamp(args.train_cutoff)], method="nearest")[0]
    train_t = [t for t in sample_t if t <= cutoff_idx]
    val_t = [t for t in sample_t if t > cutoff_idx]
    print(f"  train samples={len(train_t)}  val samples={len(val_t)}", flush=True)

    model = XSecRankModel(feat_dim, args.hidden, n_relations, args.seq_len, use_graph=use_graph).to(device)
    n_params = sum(pp.numel() for pp in model.parameters())
    print(f"[{time.strftime('%H:%M:%S')}] model: {n_params:,} params  use_graph={use_graph}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    # Inverse-frequency class weights so the model can't just predict majority "Neutral"
    valid_labels = labels[labels >= 0]
    counts = np.bincount(valid_labels, minlength=3).astype(np.float32)
    inv = (counts.sum() / np.clip(counts, 1, None)) / 3.0
    class_w = torch.tensor(inv, device=device, dtype=torch.float32)
    print(f"  class weights: {class_w.tolist()}", flush=True)
    ce = nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)

    def make_adjs(t):
        adjs = []
        rw = rets_arr[max(0, t - args.corr_window + 1) : t + 1]
        if rw.shape[0] < 2:
            corr_adj = np.zeros((N, N), dtype=np.float32)
        else:
            corr_adj = correlation_edges_topk(rw, top_k=args.corr_top_k)
        adjs.append(torch.tensor(corr_adj, device=device))
        if args.edge_mode in ("bloomberg", "proxy"):
            adjs.append(holder_t); adjs.append(supply_t)
        elif args.edge_mode == "holder_only":
            adjs.append(holder_t)
        elif args.edge_mode == "supply_only":
            adjs.append(supply_t)
        return adjs

    def epoch_pass(samples, train_mode):
        if train_mode: model.train()
        else: model.eval()
        rng = np.random.default_rng(args.seed)
        order = list(samples)
        if train_mode: rng.shuffle(order)
        all_logits = []
        all_labels = []
        tot_loss = 0.0; tot_n = 0
        for t in order:
            seq = feat_arr[t - args.seq_len + 1 : t + 1]
            seq_t = torch.tensor(seq, device=device)
            adjs = make_adjs(t) if use_graph else None
            y = torch.tensor(labels[t], device=device)
            with torch.set_grad_enabled(train_mode):
                logits = model(seq_t, adjs)
                loss = ce(logits, y)
                if train_mode and torch.isfinite(loss):
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            tot_loss += float(loss.item()) * N; tot_n += N
            with torch.no_grad():
                all_logits.append(logits.cpu().numpy())
                all_labels.append(y.cpu().numpy())
        return tot_loss / tot_n, np.stack(all_logits), np.stack(all_labels)

    from sklearn.metrics import f1_score, accuracy_score

    history = []
    for ep in range(args.epochs):
        t0 = time.time()
        train_loss, _, _ = epoch_pass(train_t, True)
        val_loss, vl, vy = epoch_pass(val_t, False)
        sched.step()
        vp = vl.argmax(-1)
        valid = vy >= 0
        acc = float(accuracy_score(vy[valid], vp[valid]))
        f1m = float(f1_score(vy[valid], vp[valid], average="macro", zero_division=0))
        # Per-day macro-F1: average within-date F1 across dates with valid labels
        per_date_f1 = []
        for d in range(vy.shape[0]):
            vy_d, vp_d = vy[d], vp[d]
            v_d = vy_d >= 0
            if v_d.sum() < 10: continue
            per_date_f1.append(f1_score(vy_d[v_d], vp_d[v_d], average="macro", zero_division=0))
        per_date_f1 = float(np.mean(per_date_f1)) if per_date_f1 else float("nan")
        history.append({
            "epoch": ep + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": acc,
            "val_macro_f1": f1m,
            "val_per_date_macro_f1": per_date_f1,
            "lr": opt.param_groups[0]["lr"],
            "elapsed_s": time.time() - t0,
        })
        print(f"[ep {ep+1:2d}/{args.epochs}] train={train_loss:.4f} val={val_loss:.4f} "
              f"acc={acc:.4f} macro_F1={f1m:.4f} per_date_F1={per_date_f1:.4f} "
              f"lr={opt.param_groups[0]['lr']:.2e} ({time.time()-t0:.1f}s)", flush=True)

    # best epoch by val per_date_F1
    best = max(history, key=lambda h: (h["val_per_date_macro_f1"] if not math.isnan(h["val_per_date_macro_f1"]) else -1))
    result = {
        "edge_mode": args.edge_mode, "seed": args.seed,
        "n_params": n_params, "n_stocks": N,
        "n_train_samples": len(train_t), "n_val_samples": len(val_t),
        "history": history, "final": history[-1], "best": best,
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
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); raise
