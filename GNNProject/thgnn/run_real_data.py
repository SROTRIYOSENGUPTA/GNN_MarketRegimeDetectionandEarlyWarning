"""
THGNN — Real Market Data Training Run
=======================================
Fetches S&P 500 stock data from Yahoo Finance, constructs the full
37-feature pipeline, and trains the THGNN model.

Uses a smaller universe (30 stocks) and fewer epochs (5) for a
feasibility test on real data. Scale up by adjusting parameters.

Usage:
    python run_real_data.py
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import math
from numbers import Integral
import time

import numpy as np
import torch

try:
    from .config import THGNNConfig
    from .data.real_data import fetch_real_data
except ImportError as exc:
    if __package__:
        raise
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from config import THGNNConfig
    from data.real_data import fetch_real_data


class CLIArgumentError(ValueError):
    """Raised when CLI arguments are malformed or unsupported."""


def _parse_iso_date(name: str, value: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise CLIArgumentError(f"{name} must be in YYYY-MM-DD format, got {value!r}.") from exc


def build_split_date_ranges(
    start: str,
    end: str,
    train_cutoff: str,
) -> tuple[tuple[str, str], tuple[str, str]]:
    start_date = _parse_iso_date("start", start)
    end_date = _parse_iso_date("end", end)
    cutoff_date = _parse_iso_date("train_cutoff", train_cutoff)

    if start_date > end_date:
        raise CLIArgumentError("start must be on or before end.")
    if not start_date <= cutoff_date <= end_date:
        raise CLIArgumentError("train_cutoff must fall within the inclusive [start, end] range.")

    train_range = (start_date.isoformat(), cutoff_date.isoformat())
    val_range = ((cutoff_date + timedelta(days=1)).isoformat(), end_date.isoformat())
    return train_range, val_range


def validate_split_sample_counts(
    train_samples: int,
    val_samples: int,
    train_range: tuple[str, str],
    val_range: tuple[str, str],
) -> None:
    if train_samples <= 0:
        raise ValueError(
            "Training split is empty after warm-up and forecast-horizon filtering "
            f"for range {train_range[0]} -> {train_range[1]}. "
            "Try an earlier --start, a later --train-cutoff, or a larger date range."
        )
    if val_samples <= 0:
        print(
            "  Validation split is empty after filtering; training will run without "
            f"validation for range {val_range[0]} -> {val_range[1]}."
        )


def resolve_device(device: str | torch.device) -> torch.device:
    supported_hint = "Supported values are cpu, cuda, cuda:N, or mps."

    try:
        resolved = torch.device(str(device))
    except (TypeError, RuntimeError, ValueError) as exc:
        raise CLIArgumentError(f"Invalid --device {device!r}. {supported_hint}") from exc

    if resolved.type not in {"cpu", "cuda", "mps"}:
        raise CLIArgumentError(f"Unsupported --device {device!r}. {supported_hint}")

    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise CLIArgumentError(
                f"CUDA device requested via --device={device!r}, "
                "but torch.cuda.is_available() is False."
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise CLIArgumentError(
                f"CUDA device index {resolved.index} is out of range; "
                f"found {torch.cuda.device_count()} visible CUDA device(s)."
            )

    if resolved.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        is_built = bool(mps_backend and getattr(mps_backend, "is_built", lambda: False)())
        is_available = bool(
            mps_backend and getattr(mps_backend, "is_available", lambda: False)()
        )
        if resolved.index not in (None, 0):
            raise CLIArgumentError("MPS exposes a single logical device; use --device mps or mps:0.")
        if not is_built:
            raise CLIArgumentError(
                f"MPS device requested via --device={device!r}, "
                "but this PyTorch build does not include MPS support."
            )
        if not is_available:
            raise CLIArgumentError(
                f"MPS device requested via --device={device!r}, "
                "but torch.backends.mps.is_available() is False."
            )

    return resolved


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise CLIArgumentError(f"{name} must be a positive integer, got {value!r}.")


def _validate_nonnegative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise CLIArgumentError(f"{name} must be a non-negative integer, got {value!r}.")


def validate_runtime_args(
    *,
    n_stocks: int,
    epochs: int,
    batch_size: int,
    grad_accum_steps: int,
    top_k_corr: int,
    bot_k_corr: int,
    rand_mid_k: int,
) -> None:
    if isinstance(n_stocks, bool) or not isinstance(n_stocks, Integral) or n_stocks < 2:
        raise CLIArgumentError(f"n_stocks must be an integer >= 2, got {n_stocks!r}.")

    _validate_positive_int("epochs", epochs)
    _validate_positive_int("batch_size", batch_size)
    _validate_positive_int("grad_accum_steps", grad_accum_steps)
    _validate_nonnegative_int("top_k_corr", top_k_corr)
    _validate_nonnegative_int("bot_k_corr", bot_k_corr)
    _validate_nonnegative_int("rand_mid_k", rand_mid_k)


def main(
    n_stocks: int = 30,
    start: str = "2021-01-01",
    end: str = "2024-06-30",
    train_cutoff: str = "2023-12-31",
    epochs: int = 5,
    batch_size: int = 2,
    grad_accum_steps: int = 2,
    top_k_corr: int = 10,
    bot_k_corr: int = 10,
    rand_mid_k: int = 15,
    device: str = "cpu",
):
    try:
        from .data.dataset import THGNNDataset, build_dataloader
        from .models.thgnn import THGNN
        from .train import Trainer
    except ImportError as exc:
        if __package__:
            raise
        from data.dataset import THGNNDataset, build_dataloader
        from models.thgnn import THGNN
        from train import Trainer

    validate_runtime_args(
        n_stocks=n_stocks,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum_steps=grad_accum_steps,
        top_k_corr=top_k_corr,
        bot_k_corr=bot_k_corr,
        rand_mid_k=rand_mid_k,
    )

    print("=" * 70)
    print("  THGNN — Real Market Data Training")
    print("=" * 70)

    # ── Configuration ────────────────────────────────────────────────────
    cfg = THGNNConfig()

    # Adjust for smaller real-data test
    cfg.epochs = epochs
    cfg.top_k_corr = top_k_corr
    cfg.bot_k_corr = bot_k_corr
    cfg.rand_mid_k = rand_mid_k
    cfg.grad_accum_steps = grad_accum_steps
    cfg.batch_size = batch_size
    train_range, val_range = build_split_date_ranges(start, end, train_cutoff)
    resolved_device = resolve_device(device)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f"  Using device: {resolved_device}")

    # ── Fetch Real Data ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 1: Fetching real market data from Yahoo Finance")
    print(f"{'─' * 70}")

    features, dates, sector_map, subind_map, returns = fetch_real_data(
        n_stocks=n_stocks,
        start=start,
        end=end,
        verbose=True,
    )

    n_real_stocks = len(features)
    T = len(dates)
    print(f"\n  Loaded {n_real_stocks} stocks × {T} trading days")
    print(f"  Feature matrix: ({T}, 37) per stock")

    # ── Construct Datasets ───────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 2: Building train/val datasets")
    print(f"{'─' * 70}")
    print(f"  Train split   : {train_range[0]} -> {train_range[1]}")
    print(f"  Validation    : {val_range[0]} -> {val_range[1]}")

    train_ds = THGNNDataset(
        features=features, dates=dates,
        sector_map=sector_map, subind_map=subind_map,
        returns=returns, cfg=cfg,
        date_range=train_range,
    )

    val_ds = THGNNDataset(
        features=features, dates=dates,
        sector_map=sector_map, subind_map=subind_map,
        returns=returns, cfg=cfg,
        date_range=val_range,
    )

    train_samples = len(train_ds)
    val_samples = len(val_ds)
    validate_split_sample_counts(train_samples, val_samples, train_range, val_range)

    train_loader = build_dataloader(train_ds, cfg, shuffle=True)
    val_loader = build_dataloader(val_ds, cfg, shuffle=False)

    print(f"  Train samples : {train_samples} days")
    print(f"  Val samples   : {val_samples} days")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── Inspect a batch ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 3: Inspecting a sample batch")
    print(f"{'─' * 70}")

    sample_batch = next(iter(train_loader))
    print(f"  x shape         : {tuple(sample_batch.x.shape)}")
    print(f"  edge_index      : {tuple(sample_batch.edge_index.shape)}")
    print(f"  edge_attr       : {tuple(sample_batch.edge_attr.shape)}")
    print(f"  edge_type       : {tuple(sample_batch.edge_type.shape)}")
    print(f"  baseline_z      : {tuple(sample_batch.baseline_z.shape)}")
    print(f"  target_z_resid  : {tuple(sample_batch.target_z_resid.shape)}")

    # Edge type distribution
    for etype in range(3):
        count = (sample_batch.edge_type == etype).sum().item()
        label = ["neg", "mid", "pos"][etype]
        print(f"  edge_type={etype} ({label}) : {count} edges")

    # Target statistics
    tgt = sample_batch.target_z_resid
    print(f"  target_z_resid  : mean={tgt.mean():.4f}, std={tgt.std():.4f}, "
          f"range=[{tgt.min():.4f}, {tgt.max():.4f}]")

    # Baseline correlation statistics
    rho_base = torch.tanh(sample_batch.baseline_z)
    print(f"  baseline ρ      : mean={rho_base.mean():.4f}, std={rho_base.std():.4f}, "
          f"range=[{rho_base.min():.4f}, {rho_base.max():.4f}]")

    # ── Model ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 4: Initialising THGNN model")
    print(f"{'─' * 70}")

    model = THGNN(cfg).to(resolved_device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    groups = model.parameter_groups()
    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_nodecay = sum(p.numel() for p in groups[1]["params"])
    print(f"  With weight decay    : {n_decay:>10,}")
    print(f"  Without weight decay : {n_nodecay:>10,} (bias + LayerNorm)")

    # ── Quick forward pass check ─────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 5: Forward pass sanity check")
    print(f"{'─' * 70}")

    sanity_batch = sample_batch.to(resolved_device)
    with torch.no_grad():
        delta_z, rho = model(
            x=sanity_batch.x,
            edge_index=sanity_batch.edge_index,
            edge_attr=sanity_batch.edge_attr,
            edge_type=sanity_batch.edge_type,
            baseline_z=sanity_batch.baseline_z,
        )
    delta_z = delta_z.cpu()
    rho = rho.cpu()
    print(f"  Δẑ shape : {tuple(delta_z.shape)}")
    print(f"  ρ̂ shape  : {tuple(rho.shape)}")
    print(f"  Δẑ range : [{delta_z.min():.4f}, {delta_z.max():.4f}]")
    print(f"  ρ̂ range  : [{rho.min():.4f}, {rho.max():.4f}]")
    assert torch.isfinite(delta_z).all(), "Non-finite Δẑ predictions!"
    assert torch.isfinite(rho).all(), "Non-finite ρ̂ predictions!"
    print(f"  ✓ All predictions finite")

    # ── Train ────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"  Step 6: Training ({cfg.epochs} epochs)")
    print(f"{'─' * 70}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader if len(val_ds) > 0 else None,
        cfg=cfg,
        device=str(resolved_device),
    )

    t0 = time.time()
    history = trainer.train()
    elapsed = time.time() - t0

    # ── Results ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 7: Training Results")
    print(f"{'─' * 70}")

    print(f"  Total training time  : {elapsed:.1f}s")
    print(f"  Optimizer steps      : {trainer.global_step}")
    print(f"  Final LR             : {trainer.optimizer.param_groups[0]['lr']:.2e}")

    print(f"\n  Epoch | Train Loss | Edge Loss  | Hist Loss  | Val Loss")
    print(f"  {'─' * 62}")
    for i, h in enumerate(history["train"]):
        val_str = ""
        if i < len(history["val"]):
            val_str = f" | {history['val'][i]['loss_total']:.5f}"
        print(f"  {i+1:5d} | {h['loss_total']:.5f}   | {h['loss_edge']:.5f}   "
              f"| {h.get('loss_hist_scaled', 0):.5f}  {val_str}")

    # Check training dynamics
    l0 = history["train"][0]["loss_total"]
    lf = history["train"][-1]["loss_total"]
    improved = lf < l0
    print(f"\n  Train loss: {l0:.5f} → {lf:.5f}  "
          f"({'↓ decreased' if improved else '~ flat/increased'})")

    if history["val"]:
        v0 = history["val"][0]["loss_total"]
        vf = history["val"][-1]["loss_total"]
        print(f"  Val loss  : {v0:.5f} → {vf:.5f}  "
              f"({'↓ decreased' if vf < v0 else '~ flat/increased'})")

    # ── Final inference on validation set ────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 8: Inference on validation data")
    print(f"{'─' * 70}")

    model.eval()
    all_delta_z = []
    all_rho = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(resolved_device)
            dz, rh = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_type=batch.edge_type,
                baseline_z=batch.baseline_z,
            )
            all_delta_z.append(dz.cpu())
            all_rho.append(rh.cpu())
            all_targets.append(batch.target_z_resid.cpu())

    if all_delta_z:
        all_dz = torch.cat(all_delta_z)
        all_rh = torch.cat(all_rho)
        all_tgt = torch.cat(all_targets)

        # Prediction statistics
        print(f"  Total edge predictions: {all_dz.shape[0]:,}")
        print(f"  Δẑ pred: mean={all_dz.mean():.4f}, std={all_dz.std():.4f}")
        print(f"  Δẑ target: mean={all_tgt.mean():.4f}, std={all_tgt.std():.4f}")

        # Correlation between predicted and actual Δz
        pred_np = all_dz.numpy()
        tgt_np = all_tgt.numpy()
        if np.std(pred_np) > 1e-8 and np.std(tgt_np) > 1e-8:
            corr = np.corrcoef(pred_np, tgt_np)[0, 1]
            print(f"  Prediction-target corr: {corr:.4f}")
        else:
            print(f"  Prediction-target corr: N/A (low variance)")

        # MSE / MAE
        mse = ((all_dz - all_tgt) ** 2).mean().item()
        mae = (all_dz - all_tgt).abs().mean().item()
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")

        # ρ̂ statistics
        print(f"\n  ρ̂ predictions: mean={all_rh.mean():.4f}, std={all_rh.std():.4f}")
        print(f"  ρ̂ range: [{all_rh.min():.4f}, {all_rh.max():.4f}]")
    else:
        print(f"  No validation data available for inference.")

    print(f"\n{'=' * 70}")
    print(f"  REAL DATA TEST COMPLETE ✓")
    print(f"{'=' * 70}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the THGNN real-data pipeline.",
    )
    parser.add_argument("--n-stocks", type=int, default=30, help="Number of stocks to fetch.")
    parser.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default="2024-06-30", help="End date (YYYY-MM-DD).")
    parser.add_argument(
        "--train-cutoff",
        default="2023-12-31",
        help="Last training date; validation starts on the next calendar day.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Physical batch size.")
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument("--top-k-corr", type=int, default=10, help="Top-K positive neighbours.")
    parser.add_argument(
        "--bot-k-corr",
        type=int,
        default=10,
        help="Bottom-K negative neighbours.",
    )
    parser.add_argument(
        "--rand-mid-k",
        type=int,
        default=15,
        help="Number of random mid-correlation neighbours.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run on: cpu, cuda, cuda:N, or mps.",
    )
    return parser


def cli(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        main(
            n_stocks=args.n_stocks,
            start=args.start,
            end=args.end,
            train_cutoff=args.train_cutoff,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            top_k_corr=args.top_k_corr,
            bot_k_corr=args.bot_k_corr,
            rand_mid_k=args.rand_mid_k,
            device=args.device,
        )
    except CLIArgumentError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
