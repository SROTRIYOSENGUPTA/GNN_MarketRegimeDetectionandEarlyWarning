from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from market_regime_gnn._legacy.config import RegimeConfig
from market_regime_gnn._legacy.data.hetero_dataset import (
    RegimeDataset,
    build_regime_dataloader,
)
from market_regime_gnn._legacy.data.label_generator import (
    REGIME_NAMES,
    generate_market_labels,
)
from market_regime_gnn._legacy.models.dynamic_regime_gnn import DynamicRegimeGNN
from market_regime_gnn._legacy.train import Trainer


def _base_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().split()[0]


def _split_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _rolling_corr(a: pd.Series, b: pd.Series, window: int) -> pd.Series:
    return a.rolling(window, min_periods=max(window // 2, 5)).corr(b).fillna(0.0)


def _rolling_beta(stock_ret: pd.Series, mkt_ret: pd.Series, window: int = 60) -> pd.Series:
    cov = stock_ret.rolling(window, min_periods=max(window // 2, 5)).cov(mkt_ret)
    var = mkt_ret.rolling(window, min_periods=max(window // 2, 5)).var()
    return (cov / var.clip(lower=1e-10)).fillna(0.0)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.clip(lower=1e-10)
    return (100 - 100 / (1 + rs)).fillna(50.0) / 100.0


def _fast_avg_cross_correlation(returns_df: pd.DataFrame, window: int = 30) -> pd.Series:
    values = returns_df.fillna(0.0).to_numpy(dtype=np.float64)
    out = np.zeros(values.shape[0], dtype=np.float64)
    n_stocks = values.shape[1]
    if n_stocks < 2:
        return pd.Series(out, index=returns_df.index)

    denom_pairs = n_stocks * (n_stocks - 1)
    for idx in range(window - 1, values.shape[0]):
        block = values[idx - window + 1 : idx + 1]
        centered = block - block.mean(axis=0, keepdims=True)
        std = centered.std(axis=0, ddof=1)
        valid = std > 1e-12
        n_valid = int(valid.sum())
        if n_valid < 2:
            continue
        z = centered[:, valid] / std[valid]
        sum_z = z.sum(axis=1)
        sum_offdiag = float(sum_z @ sum_z - n_valid * (window - 1))
        out[idx] = sum_offdiag / ((window - 1) * denom_pairs)
    return pd.Series(out, index=returns_df.index).ffill().fillna(0.0)


def _load_workbook(path: Path, start: str, end: str, max_stocks: int | None):
    xl = pd.ExcelFile(path)
    price_frames = []
    meta_frames = []

    for sheet in xl.sheet_names:
        price_frames.append(
            xl.parse(
                sheet,
                usecols=["date", "ticker", "px_last", "px_volume"],
            )
        )
        meta_frames.append(
            xl.parse(
                sheet,
                usecols=["Ticker", "Top Suppliers", "Top Customers", "Top 20 Holders"],
            ).dropna(how="all")
        )

    prices = pd.concat(price_frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["base_ticker"] = prices["ticker"].map(_base_ticker)
    prices = prices.dropna(subset=["date"])
    prices = prices[prices["base_ticker"] != ""]
    prices = prices[(prices["date"] >= start) & (prices["date"] <= end)]

    close = prices.pivot_table(
        index="date",
        columns="base_ticker",
        values="px_last",
        aggfunc="last",
    ).sort_index()
    volume = prices.pivot_table(
        index="date",
        columns="base_ticker",
        values="px_volume",
        aggfunc="last",
    ).reindex(close.index)

    coverage = close.notna().sum().sort_values(ascending=False)
    tickers = sorted(coverage.index[:max_stocks] if max_stocks else coverage.index)
    close = close[tickers].ffill().bfill()
    volume = volume[tickers].fillna(0.0)

    meta = pd.concat(meta_frames, ignore_index=True).dropna(how="all")
    meta["base_ticker"] = meta["Ticker"].map(_base_ticker)
    meta = meta[meta["base_ticker"].isin(tickers)].drop_duplicates("base_ticker")
    meta = meta.set_index("base_ticker").reindex(tickers)

    return close, volume, meta, xl.sheet_names


def _build_holder_matrix(meta: pd.DataFrame) -> tuple[np.ndarray, dict[str, int]]:
    holder_lists = [set(_split_list(value)) for value in meta["Top 20 Holders"]]
    holders = sorted({holder for holders_for_stock in holder_lists for holder in holders_for_stock})
    holder_to_idx = {holder: idx for idx, holder in enumerate(holders)}
    matrix = np.zeros((len(holder_lists), len(holders)), dtype=np.float32)
    for row_idx, holders_for_stock in enumerate(holder_lists):
        for holder in holders_for_stock:
            matrix[row_idx, holder_to_idx[holder]] = 1.0
    return matrix, holder_to_idx


def _build_supply_chain_adj(meta: pd.DataFrame, tickers: list[str]) -> np.ndarray:
    ticker_to_idx = {ticker: idx for idx, ticker in enumerate(tickers)}
    adj = np.zeros((len(tickers), len(tickers)), dtype=np.float32)
    for ticker, row in meta.iterrows():
        if ticker not in ticker_to_idx:
            continue
        src_idx = ticker_to_idx[ticker]

        for supplier in _split_list(row["Top Suppliers"]):
            supplier_base = _base_ticker(supplier)
            if supplier_base in ticker_to_idx and supplier_base != ticker:
                adj[ticker_to_idx[supplier_base], src_idx] = 1.0

        for customer in _split_list(row["Top Customers"]):
            customer_base = _base_ticker(customer)
            if customer_base in ticker_to_idx and customer_base != ticker:
                adj[src_idx, ticker_to_idx[customer_base]] = 1.0
    return adj


def _build_features(close: pd.DataFrame, volume: pd.DataFrame):
    returns_df = close.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    market_ret = returns_df.mean(axis=1)
    market_close = (1.0 + market_ret).cumprod() * 100.0
    market_vol_10 = market_ret.rolling(10).std().fillna(0.0)
    market_vol_20 = market_ret.rolling(20).std().fillna(0.0)
    cross_disp = returns_df.std(axis=1).fillna(0.0)
    avg_corr = _fast_avg_cross_correlation(returns_df, window=30)

    features: dict[int, np.ndarray] = {}
    returns: dict[int, np.ndarray] = {}
    dates = [idx.strftime("%Y-%m-%d") for idx in close.index]

    for sid, ticker in enumerate(close.columns):
        close_s = close[ticker].astype(float)
        vol_s = volume[ticker].astype(float)
        ret = returns_df[ticker].astype(float)
        stock_vol_20 = ret.rolling(20).std().fillna(0.0)

        f = pd.DataFrame(index=close.index)
        f["PRC"] = np.log1p(close_s.clip(lower=0.01))
        f["VOL"] = np.log1p(vol_s.clip(lower=0.0))
        f["mom5"] = close_s.pct_change(5).fillna(0.0)
        f["mom20"] = close_s.pct_change(20).fillna(0.0)
        f["mom60"] = close_s.pct_change(60).fillna(0.0)
        f["rev5"] = -close_s.pct_change(5).fillna(0.0)
        f["RSI14"] = _rsi(close_s)
        f["ATR14"] = ret.abs().rolling(14).mean().fillna(0.0)
        f["mktcap"] = np.log1p((close_s * vol_s.rolling(20).mean()).clip(lower=1.0))
        f["bm"] = 0.0
        f["beta_mkt"] = _rolling_beta(ret, market_ret, 60)
        f["beta_smb"] = 0.0
        f["beta_hml"] = 0.0
        f["mkt_rf"] = market_ret.values
        f["smb"] = 0.0
        f["hml"] = 0.0
        f["rf"] = 0.0
        f["umd"] = close_s.pct_change(252).fillna(0.0).values - close_s.pct_change(21).fillna(0.0).values
        f["DCOILWTICO"] = 0.0
        f["DGS10"] = 0.0
        f["DTWEXBGS"] = 0.0
        f["VIX"] = (market_vol_20 * np.sqrt(252)).fillna(0.0).values
        f["garch_vol"] = stock_vol_20.values
        f["excess_ret"] = (ret - market_ret).values
        f["raw_ret"] = ret.values
        f["market_ret"] = market_ret.values
        f["gsector"] = 0.0
        f["gsubind"] = sid / max(len(close.columns) - 1, 1)
        f["corr_mkt_10"] = _rolling_corr(ret, market_ret, 10)
        f["corr_mkt_21"] = _rolling_corr(ret, market_ret, 21)
        f["corr_mkt_63"] = _rolling_corr(ret, market_ret, 63)
        f["corr_sector_21"] = f["corr_mkt_21"]
        f["corr_subind_21"] = 0.0
        f["rvol_sector_20"] = market_vol_20.values
        f["rvol_subind_20"] = stock_vol_20.values
        f["rvol_mkt_10"] = market_vol_10.values
        f["cross_disp"] = cross_disp.values

        f = f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
        if f.shape[1] != 37:
            raise RuntimeError(f"Expected 37 features, got {f.shape[1]}")
        features[sid] = f.to_numpy(dtype=np.float32)
        returns[sid] = ret.to_numpy(dtype=np.float32)

    labels_df = generate_market_labels(pd.DataFrame({"Close": market_close}), avg_corr, min_history=60)
    labels = labels_df.reindex(close.index)
    regime_labels = labels["regime_label"].fillna(2).astype(np.int64).to_numpy()
    transition_labels = labels["transition_label"].fillna(0).astype(np.int64).to_numpy()
    return features, returns, dates, market_close, avg_corr, regime_labels, transition_labels


def _limit_dataset(dataset: RegimeDataset, max_samples: int | None, strategy: str) -> None:
    if max_samples is None or len(dataset.date_indices) <= max_samples:
        return
    if strategy == "head":
        dataset.date_indices = dataset.date_indices[:max_samples]
    elif strategy == "tail":
        dataset.date_indices = dataset.date_indices[-max_samples:]
    else:
        positions = np.linspace(0, len(dataset.date_indices) - 1, max_samples, dtype=int)
        dataset.date_indices = [dataset.date_indices[pos] for pos in positions]


def _label_summary(dataset: RegimeDataset) -> dict[str, int]:
    regimes = [int(dataset.regime_labels[idx]) for idx in dataset.date_indices]
    transitions = [int(dataset.transition_labels[idx]) for idx in dataset.date_indices]
    summary = {REGIME_NAMES[idx]: int(np.sum(np.asarray(regimes) == idx)) for idx in REGIME_NAMES}
    summary["transition_pos"] = int(np.sum(transitions))
    summary["transition_neg"] = int(len(transitions) - np.sum(transitions))
    return summary


def _prediction_summary(model: DynamicRegimeGNN, loader, device: torch.device, cfg: RegimeConfig):
    model.eval()
    regime_preds = []
    regime_true = []
    trans_probs = []
    trans_true = []
    with torch.no_grad():
        for batch in loader:
            logits, trans_logit = model(batch["snapshots"])
            regime_preds.append(logits.argmax(dim=-1).cpu())
            regime_true.append(batch["regime_label"].cpu())
            trans_probs.append(torch.sigmoid(trans_logit).cpu())
            trans_true.append(batch["transition_label"].cpu())

    if not regime_preds:
        return {}
    regime_preds_t = torch.cat(regime_preds)
    regime_true_t = torch.cat(regime_true)
    trans_probs_t = torch.cat(trans_probs)
    trans_true_t = torch.cat(trans_true)
    return {
        "regime_prediction_counts": {
            cfg.regime_names[idx]: {
                "pred": int((regime_preds_t == idx).sum().item()),
                "true": int((regime_true_t == idx).sum().item()),
            }
            for idx in range(cfg.num_regime_classes)
        },
        "transition_probability": {
            "mean": float(trans_probs_t.mean().item()),
            "std": float(trans_probs_t.std().item()) if len(trans_probs_t) > 1 else 0.0,
            "min": float(trans_probs_t.min().item()),
            "max": float(trans_probs_t.max().item()),
            "predicted_positive": int((trans_probs_t >= 0.5).sum().item()),
            "actual_positive": int(trans_true_t.sum().item()),
        },
    }


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return str(value)


def run(args: argparse.Namespace) -> dict:
    print(f"Loading workbook from {args.xlsx}...", flush=True)
    close, volume, meta, sheets = _load_workbook(
        args.xlsx,
        start=args.start,
        end=args.end,
        max_stocks=args.max_stocks,
    )
    print(
        f"Loaded workbook panel: {close.shape[1]} stocks x {close.shape[0]} dates.",
        flush=True,
    )
    tickers = list(close.columns)
    print("Building feature tensors and market labels...", flush=True)
    features, returns, dates, market_close, avg_corr, regime_labels, transition_labels = _build_features(
        close,
        volume,
    )
    print("Building holder and supplier/customer relation matrices...", flush=True)
    holder_matrix, holder_to_idx = _build_holder_matrix(meta)
    supply_chain_adj = _build_supply_chain_adj(meta, tickers)

    cfg = RegimeConfig()
    cfg.num_stocks = len(tickers)
    cfg.epochs = args.epochs
    cfg.batch_size = args.batch_size
    cfg.grad_accum_steps = args.grad_accum_steps
    cfg.warmup_steps = args.warmup_steps
    cfg.lr = args.lr
    cfg.corr_top_k = args.corr_top_k
    cfg.corr_bot_k = args.corr_bot_k
    cfg.seq_len = args.seq_len
    cfg.rolling_zscore_window = args.rolling_zscore_window
    cfg.etf_cohold_threshold = args.holder_threshold

    sector_map = {sid: 0 for sid in features}
    subind_map = {sid: sid for sid in features}
    train_range = (args.start, args.train_cutoff)
    val_range = (pd.Timestamp(args.train_cutoff) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), args.end

    train_ds = RegimeDataset(
        features,
        dates,
        sector_map,
        subind_map,
        returns,
        regime_labels,
        transition_labels,
        cfg=cfg,
        date_range=train_range,
        etf_holdings=holder_matrix,
        supply_chain_adj=supply_chain_adj,
    )
    val_ds = RegimeDataset(
        features,
        dates,
        sector_map,
        subind_map,
        returns,
        regime_labels,
        transition_labels,
        cfg=cfg,
        date_range=val_range,
        etf_holdings=holder_matrix,
        supply_chain_adj=supply_chain_adj,
    )
    _limit_dataset(train_ds, args.max_train_samples, args.train_sample_strategy)
    _limit_dataset(val_ds, args.max_val_samples, args.val_sample_strategy)
    print(
        f"Prepared datasets: {len(train_ds)} train samples, {len(val_ds)} validation samples.",
        flush=True,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    train_loader = build_regime_dataloader(train_ds, cfg, shuffle=True)
    val_loader = build_regime_dataloader(val_ds, cfg, shuffle=False)
    model = DynamicRegimeGNN(cfg).to(device)
    trainer = Trainer(model, train_loader, val_loader, cfg=cfg, device=str(device))

    print("=" * 74)
    print("S&P 500 workbook Dynamic Regime GNN pilot")
    print(f"Workbook: {args.xlsx}")
    print(f"Stocks: {len(tickers)}")
    print(f"Train/val samples: {len(train_ds)} / {len(val_ds)}")
    print(f"Device: {device}")
    print("=" * 74)

    start_time = time.time()
    history = trainer.train()
    elapsed = time.time() - start_time
    predictions = _prediction_summary(model, val_loader, device, cfg)

    n_params = sum(param.numel() for param in model.parameters())
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    result = {
        "experiment": args.experiment,
        "hostname": socket.gethostname(),
        "device": str(device),
        "gpu_name": gpu_name,
        "elapsed_seconds": elapsed,
        "config": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "warmup_steps": cfg.warmup_steps,
            "lr": cfg.lr,
            "corr_top_k": cfg.corr_top_k,
            "corr_bot_k": cfg.corr_bot_k,
            "seq_len": cfg.seq_len,
            "rolling_zscore_window": cfg.rolling_zscore_window,
            "holder_threshold": cfg.etf_cohold_threshold,
            "seed": cfg.seed,
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
        },
        "date_ranges": {
            "workbook": [args.start, args.end],
            "train": list(train_range),
            "val": list(val_range),
        },
        "dataset": {
            "sheets": sheets,
            "valid_tickers": len(tickers),
            "dates": len(dates),
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "train_label_summary": _label_summary(train_ds),
            "val_label_summary": _label_summary(val_ds),
            "holder_count": len(holder_to_idx),
            "holder_memberships": int(holder_matrix.sum()),
            "supply_chain_edges_in_universe": int(supply_chain_adj.sum()),
            "trainable_parameters": n_params,
        },
        "history": history,
        "prediction_summary": predictions,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, default=_json_default)
    print(f"Wrote {args.output}")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xlsx", type=Path, default=Path("sp500_prices 1.xlsx"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", default="sp500_workbook_h200_pilot_seq10_e1")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--train-cutoff", default="2021-12-31")
    parser.add_argument("--max-stocks", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--corr-top-k", type=int, default=3)
    parser.add_argument("--corr-bot-k", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--rolling-zscore-window", type=int, default=30)
    parser.add_argument("--holder-threshold", type=float, default=0.3)
    parser.add_argument("--max-train-samples", type=int, default=240)
    parser.add_argument("--max-val-samples", type=int, default=160)
    parser.add_argument(
        "--train-sample-strategy",
        choices=("head", "tail", "linspace"),
        default="tail",
    )
    parser.add_argument(
        "--val-sample-strategy",
        choices=("head", "tail", "linspace"),
        default="head",
    )
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
