"""
THGNN Training Script
======================
Implements the full training procedure from Appendix A.1:

  Optimiser     :  AdamW   lr=3e-4, betas=(0.9, 0.999), wd=2e-4
                   bias & LayerNorm excluded from weight decay
  LR Schedule   :  CosineAnnealingLR   T_max = total optimizer steps
                   eta_min = 1e-6
  Batch          :  physical=3, gradient accumulation=6 → effective=18
  Grad clipping  :  max_norm = 1.0
  Epochs         :  75
  Init           :  Xavier-uniform + zero bias (handled in each module)

Usage:
    from train import Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg)
    trainer.train()

Or run directly (uses synthetic data for a quick integration test):
    python train.py
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from .config import THGNNConfig
    from .models.thgnn import THGNN
    from .losses.loss import THGNNLoss
except ImportError as exc:
    if __package__:
        raise
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from config import THGNNConfig
    from models.thgnn import THGNN
    from losses.loss import THGNNLoss


class Trainer:
    """
    Encapsulates the full THGNN training loop.

    Parameters
    ----------
    model        : THGNN instance
    train_loader : PyG DataLoader (batch_size = cfg.batch_size = 3)
    val_loader   : PyG DataLoader (or None to skip validation)
    cfg          : THGNNConfig
    device       : str or torch.device
    """

    def __init__(
        self,
        model: THGNN,
        train_loader,
        val_loader=None,
        cfg: Optional[THGNNConfig] = None,
        device: str = "cpu",
    ):
        cfg = THGNNConfig() if cfg is None else cfg
        self.cfg = cfg
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ── Loss function ─────────────────────────────────────────────
        self.loss_fn = THGNNLoss(cfg)

        # ── Optimiser (bias + LayerNorm excluded from decay) ──────────
        param_groups = self.model.parameter_groups()
        self.optimizer = AdamW(
            param_groups,
            lr=cfg.lr,                              # 3e-4
            betas=cfg.betas,                        # (0.9, 0.999)
            # weight_decay set per group in parameter_groups()
        )

        # ── Cosine LR schedule ────────────────────────────────────────
        # T_max = total optimizer steps across all epochs
        # One optimizer step per grad_accum_steps physical batches
        steps_per_epoch = math.ceil(
            len(train_loader) / cfg.grad_accum_steps
        )
        total_optim_steps = steps_per_epoch * cfg.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_optim_steps,                # total optimizer steps
            eta_min=cfg.lr_min,                     # 1e-6
        )

        # ── Tracking ──────────────────────────────────────────────────
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []
        self.global_step = 0

    # ══════════════════════════════════════════════════════════════════
    # Training loop
    # ══════════════════════════════════════════════════════════════════
    def train(self) -> Dict[str, List]:
        """
        Run the full training procedure for cfg.epochs epochs.

        Returns
        -------
        dict with keys "train" and "val", each a list of per-epoch log dicts.
        """
        cfg = self.cfg
        print(f"\n{'=' * 68}")
        print(f"  THGNN Training")
        print(f"  Epochs: {cfg.epochs}  |  Physical batch: {cfg.batch_size}  |  "
              f"Accum: {cfg.grad_accum_steps}  |  Effective batch: "
              f"{cfg.batch_size * cfg.grad_accum_steps}")
        print(f"  LR: {cfg.lr}  |  Cosine → {cfg.lr_min}  |  "
              f"Grad clip: {cfg.grad_clip_norm}")
        print(f"{'=' * 68}\n")

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # ── Train one epoch ───────────────────────────────────────
            train_log = self._train_epoch(epoch)
            self.train_history.append(train_log)

            # ── Validate ──────────────────────────────────────────────
            val_log = {}
            if self.val_loader is not None:
                val_log = self._validate(epoch)
                self.val_history.append(val_log)

            elapsed = time.time() - t0

            # ── Print epoch summary ───────────────────────────────────
            lr_now = self.optimizer.param_groups[0]["lr"]
            msg = (
                f"Epoch {epoch:3d}/{cfg.epochs} "
                f"| train_loss {train_log['loss_total']:.5f} "
                f"(edge {train_log['loss_edge']:.5f}, "
                f"hist {train_log['loss_hist_scaled']:.5f})"
            )
            if val_log:
                msg += (
                    f" | val_loss {val_log['loss_total']:.5f}"
                )
            msg += f" | lr {lr_now:.2e} | {elapsed:.1f}s"
            print(msg)

        print(f"\nTraining complete.  Global optimizer steps: {self.global_step}")

        return {"train": self.train_history, "val": self.val_history}

    # ──────────────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one epoch of training with gradient accumulation."""
        cfg = self.cfg
        self.model.train()
        self.optimizer.zero_grad()

        running = {}
        n_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)

            # ── Forward pass ──────────────────────────────────────────
            delta_z_pred, rho_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_type=batch.edge_type,
                baseline_z=batch.baseline_z,
            )

            # ── Loss ──────────────────────────────────────────────────
            loss, log = self.loss_fn(
                delta_z_pred=delta_z_pred,
                rho_pred=rho_pred,
                target_z_resid=batch.target_z_resid,
                baseline_z=batch.baseline_z,
                edge_type=batch.edge_type,
            )

            # Scale loss by accumulation steps so the effective gradient
            # is averaged over the full effective batch
            scaled_loss = loss / cfg.grad_accum_steps
            scaled_loss.backward()

            # ── Accumulate running metrics ────────────────────────────
            for k, v in log.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

            # ── Optimizer step every grad_accum_steps batches ─────────
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                self._optimizer_step()

        # Handle leftover batches at end of epoch
        if n_batches % cfg.grad_accum_steps != 0:
            self._optimizer_step()

        # Average metrics
        return {k: v / max(n_batches, 1) for k, v in running.items()}

    # ──────────────────────────────────────────────────────────────────
    def _optimizer_step(self):
        """Clip gradients → optimizer step → scheduler step → zero grad."""
        cfg = self.cfg

        # Gradient clipping (max norm = 1.0)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=cfg.grad_clip_norm,
        )

        self.optimizer.step()
        self.scheduler.step()          # cosine schedule steps per optimizer step
        self.optimizer.zero_grad()
        self.global_step += 1

    # ══════════════════════════════════════════════════════════════════
    # Validation
    # ══════════════════════════════════════════════════════════════════
    @torch.no_grad()
    def _validate(self, epoch: int) -> Dict[str, float]:
        """Run one pass over the validation set."""
        self.model.eval()
        running = {}
        n_batches = 0

        for batch in self.val_loader:
            batch = batch.to(self.device)

            delta_z_pred, rho_pred = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                edge_type=batch.edge_type,
                baseline_z=batch.baseline_z,
            )

            _, log = self.loss_fn(
                delta_z_pred=delta_z_pred,
                rho_pred=rho_pred,
                target_z_resid=batch.target_z_resid,
                baseline_z=batch.baseline_z,
                edge_type=batch.edge_type,
            )

            for k, v in log.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in running.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Integration smoke test — runs 3 epochs on synthetic data
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import datetime
    import numpy as np
    try:
        from .data.dataset import THGNNDataset, build_dataloader
    except ImportError as exc:
        if __package__:
            raise
        from data.dataset import THGNNDataset, build_dataloader

    print("=" * 68)
    print("  THGNN Training Loop — Integration Test (3 epochs)")
    print("=" * 68)

    # ── Synthetic data ────────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = THGNNConfig()
    cfg.epochs = 3                   # just 3 for the smoke test
    cfg.top_k_corr = 5
    cfg.bot_k_corr = 5
    cfg.rand_mid_k = 5
    cfg.grad_accum_steps = 2         # smaller for test

    n_stocks, n_days = 20, 200
    rng = np.random.default_rng(42)
    base = datetime.date(2018, 1, 1)
    dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]

    features, returns, sector_map, subind_map = {}, {}, {}, {}
    for sid in range(n_stocks):
        features[sid] = rng.standard_normal((n_days, cfg.num_features)).astype(np.float32)
        returns[sid] = rng.standard_normal(n_days).astype(np.float32) * 0.02
        sector_map[sid] = int(rng.integers(0, 11))
        subind_map[sid] = int(rng.integers(0, 50))

    ds = THGNNDataset(features=features, dates=dates, sector_map=sector_map,
                      subind_map=subind_map, returns=returns, cfg=cfg)
    train_loader = build_dataloader(ds, cfg, shuffle=True)

    print(f"\nDataset : {len(ds)} samples")
    print(f"Batches : {len(train_loader)} (batch_size={cfg.batch_size})")
    print(f"Epochs  : {cfg.epochs}")

    # ── Model + Trainer ───────────────────────────────────────────────
    model = THGNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model   : {n_params:,} parameters")

    trainer = Trainer(model=model, train_loader=train_loader, cfg=cfg)

    # ── Verify scheduler is configured correctly ──────────────────────
    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.epochs
    print(f"Optim steps/epoch : {steps_per_epoch}")
    print(f"Total optim steps : {total_steps}")
    print(f"Scheduler T_max   : {trainer.scheduler.T_max}")

    # ── Train ─────────────────────────────────────────────────────────
    history = trainer.train()

    # ── Verify ────────────────────────────────────────────────────────
    print(f"\n── Verification ──")
    assert len(history["train"]) == cfg.epochs
    assert trainer.global_step > 0
    print(f"  Global optimizer steps : {trainer.global_step}")
    print(f"  Final LR              : {trainer.optimizer.param_groups[0]['lr']:.2e}")

    # Check loss is finite across all epochs
    for i, h in enumerate(history["train"]):
        assert math.isfinite(h["loss_total"]), f"Epoch {i+1}: non-finite loss!"
    print(f"  All losses finite     : ✓")

    # Check LR decreased from initial
    final_lr = trainer.optimizer.param_groups[0]["lr"]
    assert final_lr < cfg.lr, "LR should have decreased via cosine schedule"
    print(f"  LR cosine decay       : {cfg.lr:.2e} → {final_lr:.2e} ✓")

    # Check loss trend (loss should generally decrease over 3 epochs on
    # the same data, though not guaranteed with random data)
    l0 = history["train"][0]["loss_total"]
    lf = history["train"][-1]["loss_total"]
    trend = "↓ decreased" if lf < l0 else "~ flat/increased (ok with random data)"
    print(f"  Loss trend            : {l0:.5f} → {lf:.5f}  {trend}")

    print(f"\n{'=' * 68}")
    print(f"  TRAINING INTEGRATION TEST PASSED ✓")
    print(f"{'=' * 68}")
