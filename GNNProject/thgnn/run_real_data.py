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

import math
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from config import THGNNConfig
from data.real_data import fetch_real_data
from data.dataset import THGNNDataset, build_dataloader
from models.thgnn import THGNN
from train import Trainer


def main():
    print("=" * 70)
    print("  THGNN — Real Market Data Training")
    print("=" * 70)

    # ── Configuration ────────────────────────────────────────────────────
    cfg = THGNNConfig()

    # Adjust for smaller real-data test
    cfg.epochs = 5                     # quick feasibility test
    cfg.top_k_corr = 10                # fewer edges (smaller universe)
    cfg.bot_k_corr = 10
    cfg.rand_mid_k = 15
    cfg.grad_accum_steps = 2           # smaller effective batch
    cfg.batch_size = 2                 # smaller for memory

    N_STOCKS = 30                      # 30 stocks from 11 sectors
    START = "2021-01-01"
    END = "2024-06-30"
    TRAIN_CUTOFF = "2023-12-31"        # train: 2021-2023, val: 2024-H1

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # ── Fetch Real Data ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  Step 1: Fetching real market data from Yahoo Finance")
    print(f"{'─' * 70}")

    features, dates, sector_map, subind_map, returns = fetch_real_data(
        n_stocks=N_STOCKS,
        start=START,
        end=END,
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

    train_ds = THGNNDataset(
        features=features, dates=dates,
        sector_map=sector_map, subind_map=subind_map,
        returns=returns, cfg=cfg,
        date_range=(START, TRAIN_CUTOFF),
    )

    val_ds = THGNNDataset(
        features=features, dates=dates,
        sector_map=sector_map, subind_map=subind_map,
        returns=returns, cfg=cfg,
        date_range=("2024-01-01", END),
    )

    train_loader = build_dataloader(train_ds, cfg, shuffle=True)
    val_loader = build_dataloader(val_ds, cfg, shuffle=False)

    print(f"  Train samples : {len(train_ds)} days")
    print(f"  Val samples   : {len(val_ds)} days")
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

    model = THGNN(cfg)
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

    with torch.no_grad():
        delta_z, rho = model(
            x=sample_batch.x,
            edge_index=sample_batch.edge_index,
            edge_attr=sample_batch.edge_attr,
            edge_type=sample_batch.edge_type,
            baseline_z=sample_batch.baseline_z,
        )
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
        device="cpu",
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
            dz, rh = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_type=batch.edge_type,
                baseline_z=batch.baseline_z,
            )
            all_delta_z.append(dz)
            all_rho.append(rh)
            all_targets.append(batch.target_z_resid)

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


if __name__ == "__main__":
    main()
