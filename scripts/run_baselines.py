"""
Baselines for the GNN regime detection task.

Reuses the same workbook loading + label generator as the GNN flow, but
feeds aggregate market features (not per-stock graphs) into:

    1. Logistic regression  (multi-class for regime, binary for transition)
    2. LSTM-only            (30-day sequence of aggregate features)

If the dynamic heterogeneous GNN cannot beat these, the architecture
isn't doing real work and the project needs a rethink.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

REPO = Path("/scratch/ss3414/gnn_regime/GNN_MarketRegimeDetectionandEarlyWarning")
sys.path.insert(0, str(REPO))
from market_regime_gnn._legacy.data import label_generator as lg


XLSX = "/scratch/ss3414/gnn_regime/sp500_prices 1.xlsx"


def load_workbook():
    sheets = pd.read_excel(XLSX, sheet_name=None)
    rows = pd.concat(sheets.values(), ignore_index=True)
    rows["date"] = pd.to_datetime(rows["date"])
    close = rows.pivot_table(index="date", columns="ticker", values="px_last", aggfunc="last").sort_index()
    return close.ffill().bfill()


def fast_avg_corr(rets: pd.DataFrame, window: int = 30) -> pd.Series:
    T, N = rets.shape
    out = pd.Series(np.nan, index=rets.index, dtype=np.float64)
    if N < 2:
        return out.fillna(0.0)
    arr = rets.to_numpy(dtype=np.float64)
    for t in range(window - 1, T):
        block = arr[t - window + 1 : t + 1]
        block = block - block.mean(axis=0, keepdims=True)
        sd = block.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        block = block / sd
        c = (block.T @ block) / window
        iu = np.triu_indices(N, k=1)
        vals = c[iu]
        vals = vals[~np.isnan(vals)]
        if len(vals):
            out.iloc[t] = vals.mean()
    return out.ffill().fillna(0.0)


def build_data(seed: int):
    close = load_workbook()
    rets = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_ret = rets.mean(axis=1)
    market_close = (1.0 + market_ret).cumprod() * 100.0
    avg_corr = fast_avg_corr(rets, 30)

    # 7 aggregate market features per day
    feats = pd.DataFrame(index=close.index)
    feats["mkt_ret"] = market_ret.values
    feats["mkt_ret_5"] = market_close.pct_change(5).fillna(0.0).values
    feats["mkt_ret_20"] = market_close.pct_change(20).fillna(0.0).values
    feats["mkt_vol_5"] = market_ret.rolling(5).std().fillna(0.0).values
    feats["mkt_vol_20"] = market_ret.rolling(20).std().fillna(0.0).values
    feats["cross_disp"] = rets.std(axis=1).fillna(0.0).values
    feats["avg_corr"] = avg_corr.values
    feats = feats.fillna(0.0).astype(np.float32)

    labels_df = lg.generate_market_labels(
        pd.DataFrame({"Close": market_close}), avg_corr, min_history=60
    ).reindex(close.index)
    regime = labels_df["regime_label"].fillna(2).astype(np.int64).to_numpy()
    transition = labels_df["transition_label"].fillna(0).astype(np.int64).to_numpy()

    seq_len = 30
    feat_arr = feats.to_numpy()
    T = len(feats)
    # Each sample: 30-day window ending at day t (target day = t)
    samples = []
    for t in range(seq_len - 1, T):
        samples.append((t, feat_arr[t - seq_len + 1 : t + 1]))

    # Split by train_cutoff 2022-09-30
    cutoff = pd.Timestamp("2022-09-30")
    train_idx, val_idx, train_X, val_X, train_y, val_y, train_tr, val_tr = [], [], [], [], [], [], [], []
    dates = feats.index
    for t, win in samples:
        d = dates[t]
        if d <= cutoff:
            train_X.append(win); train_y.append(regime[t]); train_tr.append(transition[t]); train_idx.append(t)
        else:
            val_X.append(win); val_y.append(regime[t]); val_tr.append(transition[t]); val_idx.append(t)
    return (np.array(train_X), np.array(train_y), np.array(train_tr),
            np.array(val_X), np.array(val_y), np.array(val_tr))


def eval_regime(y_true, y_pred, name):
    return {
        f"{name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{name}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def eval_transition(y_true, y_prob, y_pred, name):
    out = {
        f"{name}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{name}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{name}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out[f"{name}_roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        out[f"{name}_roc_auc"] = float("nan")
    return out


def run_logistic(tx, ty, ttr, vx, vy, vtr, seed):
    # Flatten 30-day window into single vector
    tx_flat = tx.reshape(len(tx), -1); vx_flat = vx.reshape(len(vx), -1)
    out = {"name": "logistic_regression", "seed": seed}
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf.fit(tx_flat, ty)
    out.update(eval_regime(vy, clf.predict(vx_flat), "regime"))
    clf2 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
    clf2.fit(tx_flat, ttr)
    out.update(eval_transition(vtr, clf2.predict_proba(vx_flat)[:, 1], clf2.predict(vx_flat), "transition"))
    return out


class LSTMBaseline(nn.Module):
    def __init__(self, in_dim, hidden=64, n_regime=4):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True)
        self.reg_head = nn.Linear(hidden, n_regime)
        self.tr_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        last = h[:, -1, :]
        return self.reg_head(last), self.tr_head(last).squeeze(-1)


def run_lstm(tx, ty, ttr, vx, vy, vtr, seed, epochs=30, lr=1e-3, device="cpu"):
    torch.manual_seed(seed); np.random.seed(seed)
    tx_t = torch.tensor(tx, dtype=torch.float32)
    ty_t = torch.tensor(ty, dtype=torch.long)
    ttr_t = torch.tensor(ttr, dtype=torch.float32)
    vx_t = torch.tensor(vx, dtype=torch.float32)

    dev = torch.device(device)
    model = LSTMBaseline(in_dim=tx.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(tx_t, ty_t, ttr_t)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    # Class weights for imbalance
    counts = np.bincount(ty, minlength=4).astype(np.float32)
    class_w = torch.tensor((counts.sum() / np.clip(counts, 1, None)) / 4.0, device=dev)

    ce = nn.CrossEntropyLoss(weight=class_w)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb, trb in loader:
            xb, yb, trb = xb.to(dev), yb.to(dev), trb.to(dev)
            reg_logits, tr_logit = model(xb)
            loss = ce(reg_logits, yb) + bce(tr_logit, trb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        reg_logits, tr_logit = model(vx_t.to(dev))
        reg_pred = reg_logits.argmax(-1).cpu().numpy()
        tr_prob = torch.sigmoid(tr_logit).cpu().numpy()
        tr_pred = (tr_prob > 0.5).astype(np.int64)

    out = {"name": "lstm_baseline", "seed": seed}
    out.update(eval_regime(vy, reg_pred, "regime"))
    out.update(eval_transition(vtr, tr_prob, tr_pred, "transition"))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("Loading + building data once...", flush=True)
    tx, ty, ttr, vx, vy, vtr = build_data(seed=42)
    print(f"  train: X={tx.shape} y_regime={np.bincount(ty)} y_tr_pos={int(ttr.sum())}/{len(ttr)}", flush=True)
    print(f"  val:   X={vx.shape} y_regime={np.bincount(vy)} y_tr_pos={int(vtr.sum())}/{len(vtr)}", flush=True)

    results = []
    for s in args.seeds:
        print(f"\n--- seed {s}: logistic ---", flush=True)
        results.append(run_logistic(tx, ty, ttr, vx, vy, vtr, s))
        print(json.dumps(results[-1], indent=2))
        print(f"--- seed {s}: LSTM ---", flush=True)
        results.append(run_lstm(tx, ty, ttr, vx, vy, vtr, s, device=args.device))
        print(json.dumps(results[-1], indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
