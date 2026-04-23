"""
Dynamic Regime GNN — Training Engine
======================================
Dual-task optimization for Market Regime Detection & Early Warning.

Loss Function:
──────────────
    L_total = w_regime · L_regime  +  w_transition · L_transition

    L_regime     = Focal Cross-Entropy (4-class, label smoothing=0.05, γ=2)
                   Focal loss down-weights easy examples to handle class
                   imbalance (Bull ≫ Crash/Stress).

    L_transition = BCE with logits (binary, pos_weight for imbalance)
                   Early-warning target: P(stress in 5–20 days).

Optimizer:
──────────
    AdamW  lr=1e-3, betas=(0.9, 0.999), wd=1e-4
    Bias & LayerNorm parameters excluded from weight decay.

LR Schedule:
────────────
    Linear warmup (200 steps) → CosineAnnealingLR → lr_min=1e-6

Training Loop:
──────────────
    Gradient accumulation (4 steps) → effective batch = 4 × 4 = 16
    Gradient clipping: max_norm = 1.0

Metrics:
────────
    Regime: accuracy, per-class accuracy, macro-F1
    Transition: accuracy, ROC-AUC, precision, recall

Usage:
    from train import Trainer
    trainer = Trainer(model, train_loader, val_loader, cfg)
    history = trainer.train()
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.data import HeteroData

try:
    from .config import RegimeConfig
    from .models.dynamic_regime_gnn import DynamicRegimeGNN
except ImportError as exc:
    if __package__:
        raise
    import os
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from config import RegimeConfig
    from models.dynamic_regime_gnn import DynamicRegimeGNN


# ═══════════════════════════════════════════════════════════════════════════
# Focal Loss for class-imbalanced regime classification
# ═══════════════════════════════════════════════════════════════════════════
class FocalCrossEntropyLoss(nn.Module):
    """
    Focal Loss: down-weights well-classified examples to focus on hard ones.

        FL(p_t) = -α_t (1 - p_t)^γ  log(p_t)

    With label smoothing applied to the target distribution before computing
    the focal modulation.

    Parameters
    ----------
    gamma          : focusing parameter (γ=0 → standard CE, γ=2 recommended)
    label_smoothing: smooth targets toward uniform (0.05 default)
    weight         : per-class weight tensor (optional)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C) raw logits from regime head.
        targets : (B,)   integer class labels.

        Returns
        -------
        scalar focal loss.
        """
        C = logits.shape[1]

        # Label smoothing: convert hard labels → soft distribution
        with torch.no_grad():
            smooth = torch.full_like(logits, self.label_smoothing / (C - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        # Log-softmax for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)           # (B, C)
        probs = torch.exp(log_probs)                         # (B, C)

        # Focal modulation: (1 - p_t)^γ
        focal_weight = (1.0 - probs).pow(self.gamma)         # (B, C)

        # Per-class weighting
        if self.weight is not None:
            class_weight = self.weight.to(logits.device)
            focal_weight = focal_weight * class_weight.unsqueeze(0)

        # Focal cross-entropy
        loss = -focal_weight * smooth * log_probs             # (B, C)
        return loss.sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════════════════
# Device transfer utilities for HeteroData
# ═══════════════════════════════════════════════════════════════════════════
def move_hetero_to_device(data: HeteroData, device: torch.device) -> HeteroData:
    """Move all tensors in a HeteroData object to the specified device."""
    return data.to(device)


def move_snapshots_to_device(
    snapshots: List[List[HeteroData]],
    device: torch.device,
) -> List[List[HeteroData]]:
    """
    Compatibility helper for older call sites.

    The model now batches each timestep on CPU and moves the PyG Batch to the
    target device inside its forward pass. Returning the original snapshot grid
    here avoids mutating cached HeteroData objects and prevents redundant
    per-snapshot device transfers.

    Parameters
    ----------
    snapshots : list[list[HeteroData]] — (T, B) from collate_fn.
    device    : target device.

    Returns
    -------
    The original snapshot grid.
    """
    return snapshots


# ═══════════════════════════════════════════════════════════════════════════
# Metrics computation
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(
    regime_logits_all: torch.Tensor,
    regime_labels_all: torch.Tensor,
    trans_logits_all: torch.Tensor,
    trans_labels_all: torch.Tensor,
    num_classes: int = 4,
) -> Dict[str, float]:
    """
    Compute classification metrics for both tasks.

    Parameters
    ----------
    regime_logits_all  : (N_total, 4) concatenated regime logits.
    regime_labels_all  : (N_total,)   ground truth regime labels.
    trans_logits_all   : (N_total,)   concatenated transition logits.
    trans_labels_all   : (N_total,)   ground truth transition labels.

    Returns
    -------
    dict with keys:
        regime_accuracy, regime_per_class_acc, regime_macro_f1,
        transition_accuracy, transition_roc_auc,
        transition_precision, transition_recall
    """
    metrics = {}

    # ── Regime metrics ──────────────────────────────────────────────────
    regime_preds = regime_logits_all.argmax(dim=-1)          # (N,)
    correct = (regime_preds == regime_labels_all).float()
    metrics["regime_accuracy"] = correct.mean().item()

    # Per-class accuracy
    per_class = {}
    for c in range(num_classes):
        mask = regime_labels_all == c
        if mask.sum() > 0:
            per_class[c] = correct[mask].mean().item()
        else:
            per_class[c] = float("nan")
    metrics["regime_per_class_acc"] = per_class

    # Macro F1 (per-class F1, averaged)
    f1_scores = []
    for c in range(num_classes):
        support = (
            (regime_labels_all == c).sum() + (regime_preds == c).sum()
        ).item()
        if support == 0:
            continue
        tp = ((regime_preds == c) & (regime_labels_all == c)).sum().float()
        fp = ((regime_preds == c) & (regime_labels_all != c)).sum().float()
        fn = ((regime_preds != c) & (regime_labels_all == c)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())
    metrics["regime_macro_f1"] = float(np.mean(f1_scores)) if f1_scores else float("nan")

    # ── Transition metrics ──────────────────────────────────────────────
    trans_probs = torch.sigmoid(trans_logits_all)
    trans_preds = (trans_probs >= 0.5).long()
    trans_correct = (trans_preds == trans_labels_all.long()).float()
    metrics["transition_accuracy"] = trans_correct.mean().item()

    # Precision / Recall for positive class (stress ahead)
    tp = ((trans_preds == 1) & (trans_labels_all == 1)).sum().float()
    fp = ((trans_preds == 1) & (trans_labels_all == 0)).sum().float()
    fn = ((trans_preds == 0) & (trans_labels_all == 1)).sum().float()
    metrics["transition_precision"] = (tp / (tp + fp + 1e-8)).item()
    metrics["transition_recall"] = (tp / (tp + fn + 1e-8)).item()

    # ROC-AUC (manual trapezoid implementation — no sklearn dependency)
    try:
        metrics["transition_roc_auc"] = _compute_roc_auc(
            trans_probs.cpu().numpy(),
            trans_labels_all.cpu().numpy(),
        )
    except Exception:
        metrics["transition_roc_auc"] = float("nan")

    return metrics


def _compute_roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute ROC-AUC without sklearn, with correct tie handling.

    This uses the Mann-Whitney / rank-sum formulation, which is equivalent
    to ROC-AUC and correctly assigns average ranks to tied probabilities.
    Returns NaN if only one class is present.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(probs, kind="mergesort")
    sorted_probs = probs[order]
    ranks = np.empty(len(probs), dtype=np.float64)

    i = 0
    while i < len(sorted_probs):
        j = i + 1
        while j < len(sorted_probs) and sorted_probs[j] == sorted_probs[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + j)  # 1-based average rank over [i, j)
        ranks[order[i:j]] = avg_rank
        i = j

    pos_ranks = ranks[labels == 1].sum()
    auc = (pos_ranks - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


# ═══════════════════════════════════════════════════════════════════════════
# Trainer
# ═══════════════════════════════════════════════════════════════════════════
class Trainer:
    """
    Dual-task training engine for DynamicRegimeGNN.

    Parameters
    ----------
    model        : DynamicRegimeGNN instance
    train_loader : DataLoader from build_regime_dataloader
    val_loader   : DataLoader (or None to skip validation)
    cfg          : RegimeConfig
    device       : str or torch.device
    """

    def __init__(
        self,
        model: DynamicRegimeGNN,
        train_loader,
        val_loader=None,
        cfg: Optional[RegimeConfig] = None,
        device: str = "cpu",
    ):
        cfg = RegimeConfig() if cfg is None else cfg
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(device)

        # ── Loss functions ──────────────────────────────────────────────
        self.regime_loss_fn = FocalCrossEntropyLoss(
            gamma=cfg.focal_gamma,
            label_smoothing=cfg.label_smoothing,
        )
        # Transition: BCE with logits. pos_weight adjustable for imbalance.
        self.transition_loss_fn = nn.BCEWithLogitsLoss()

        # ── Optimizer: AdamW with parameter groups ──────────────────────
        param_groups = model.parameter_groups()
        self.optimizer = AdamW(
            param_groups,
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        # ── LR Scheduler: warmup + cosine annealing ────────────────────
        steps_per_epoch = math.ceil(
            len(train_loader) / cfg.grad_accum_steps
        )
        self.total_steps = steps_per_epoch * cfg.epochs
        self.warmup_steps = cfg.warmup_steps

        self.cosine_steps = max(self.total_steps - self.warmup_steps, 1)

        self.global_step = 0

        # Start from zero when warmup is enabled so the first optimizer step
        # uses a small but non-zero LR after warmup adjustment.
        if self.warmup_steps > 0:
            for pg in self.optimizer.param_groups:
                pg["lr"] = 0.0

    def _get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def _warmup_lr(self, step_num: int):
        """Apply linear warmup for the upcoming optimizer step."""
        if step_num <= self.warmup_steps:
            warmup_factor = step_num / max(self.warmup_steps, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.cfg.lr * warmup_factor

    def _step_scheduler(self):
        """Apply the post-warmup cosine learning-rate schedule."""
        if self.global_step > self.warmup_steps:
            post_warmup_step = min(
                self.global_step - self.warmup_steps,
                self.cosine_steps,
            )
            cosine = 0.5 * (
                1.0 + math.cos(math.pi * post_warmup_step / self.cosine_steps)
            )
            lr = self.cfg.lr_min + (self.cfg.lr - self.cfg.lr_min) * cosine
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    # ── Training one epoch ──────────────────────────────────────────────
    def train_one_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch with gradient accumulation.

        Returns
        -------
        dict with average losses: loss_total, loss_regime, loss_transition
        """
        self.model.train()
        cfg = self.cfg

        total_loss_accum = 0.0
        regime_loss_accum = 0.0
        trans_loss_accum = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # ── Unpack batch from regime_collate_fn ─────────────────────
            snapshots = batch["snapshots"]
            regime_labels = batch["regime_label"].to(self.device)       # (B,)
            trans_labels = batch["transition_label"].to(self.device)     # (B,)

            # ── Forward pass ────────────────────────────────────────────
            regime_logits, trans_logit = self.model(snapshots)
            # regime_logits: (B, 4), trans_logit: (B,)

            # ── Dual loss ───────────────────────────────────────────────
            loss_regime = self.regime_loss_fn(regime_logits, regime_labels)
            loss_trans = self.transition_loss_fn(trans_logit, trans_labels)

            loss_total = (
                cfg.regime_loss_weight * loss_regime
                + cfg.transition_loss_weight * loss_trans
            )

            # Scale by accumulation steps
            scaled_loss = loss_total / cfg.grad_accum_steps

            # ── Backward ────────────────────────────────────────────────
            scaled_loss.backward()

            # ── Optimizer step every grad_accum_steps batches ───────────
            if (batch_idx + 1) % cfg.grad_accum_steps == 0 or \
               (batch_idx + 1) == len(self.train_loader):
                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    cfg.grad_clip_norm,
                )

                # Warmup LR adjustment
                if self.global_step < self.warmup_steps:
                    self._warmup_lr(self.global_step + 1)

                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Scheduler step
                self.global_step += 1
                self._step_scheduler()

            # ── Accumulate metrics ──────────────────────────────────────
            total_loss_accum += loss_total.item()
            regime_loss_accum += loss_regime.item()
            trans_loss_accum += loss_trans.item()
            n_batches += 1

        return {
            "loss_total": total_loss_accum / max(n_batches, 1),
            "loss_regime": regime_loss_accum / max(n_batches, 1),
            "loss_transition": trans_loss_accum / max(n_batches, 1),
        }

    # ── Evaluation ──────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader=None) -> Dict[str, Any]:
        """
        Evaluate on validation set.

        Returns
        -------
        dict with losses and classification metrics.
        """
        if loader is None:
            loader = self.val_loader
        if loader is None:
            return {}

        self.model.eval()
        cfg = self.cfg

        total_loss_accum = 0.0
        regime_loss_accum = 0.0
        trans_loss_accum = 0.0
        n_batches = 0

        all_regime_logits = []
        all_regime_labels = []
        all_trans_logits = []
        all_trans_labels = []

        for batch in loader:
            snapshots = batch["snapshots"]
            regime_labels = batch["regime_label"].to(self.device)
            trans_labels = batch["transition_label"].to(self.device)

            regime_logits, trans_logit = self.model(snapshots)

            loss_regime = self.regime_loss_fn(regime_logits, regime_labels)
            loss_trans = self.transition_loss_fn(trans_logit, trans_labels)
            loss_total = (
                cfg.regime_loss_weight * loss_regime
                + cfg.transition_loss_weight * loss_trans
            )

            total_loss_accum += loss_total.item()
            regime_loss_accum += loss_regime.item()
            trans_loss_accum += loss_trans.item()
            n_batches += 1

            all_regime_logits.append(regime_logits.cpu())
            all_regime_labels.append(regime_labels.cpu())
            all_trans_logits.append(trans_logit.cpu())
            all_trans_labels.append(trans_labels.cpu())

        if n_batches == 0:
            return {}

        # Concatenate all predictions
        regime_logits_cat = torch.cat(all_regime_logits, dim=0)
        regime_labels_cat = torch.cat(all_regime_labels, dim=0)
        trans_logits_cat = torch.cat(all_trans_logits, dim=0)
        trans_labels_cat = torch.cat(all_trans_labels, dim=0)

        # Compute metrics
        metrics = compute_metrics(
            regime_logits_cat, regime_labels_cat,
            trans_logits_cat, trans_labels_cat,
            num_classes=cfg.num_regime_classes,
        )

        # Add losses
        metrics["loss_total"] = total_loss_accum / max(n_batches, 1)
        metrics["loss_regime"] = regime_loss_accum / max(n_batches, 1)
        metrics["loss_transition"] = trans_loss_accum / max(n_batches, 1)

        return metrics

    # ── Full training loop ──────────────────────────────────────────────
    def train(self) -> Dict[str, List]:
        """
        Full training loop across all epochs.

        Returns
        -------
        dict with keys "train" and "val", each a list of per-epoch dicts.
        """
        cfg = self.cfg
        history: Dict[str, List] = {"train": [], "val": []}

        n_params = sum(p.numel() for p in self.model.parameters())
        regime_names = cfg.regime_names

        print("\n" + "=" * 74)
        print("  Dynamic Regime GNN — Training")
        print(f"  Epochs: {cfg.epochs}  |  Phys batch: {cfg.batch_size}  |  "
              f"Accum: {cfg.grad_accum_steps}  |  Eff batch: "
              f"{cfg.batch_size * cfg.grad_accum_steps}")
        print(f"  LR: {cfg.lr}  |  Warmup: {cfg.warmup_steps} steps  |  "
              f"Cosine → {cfg.lr_min}")
        print(f"  Focal γ: {cfg.focal_gamma}  |  Label smooth: "
              f"{cfg.label_smoothing}  |  Grad clip: {cfg.grad_clip_norm}")
        print(f"  Parameters: {n_params:,}")
        print("=" * 74)

        for epoch in range(1, cfg.epochs + 1):
            t0 = time.time()

            # ── Train ───────────────────────────────────────────────────
            train_metrics = self.train_one_epoch()
            history["train"].append(train_metrics)

            # ── Validate ────────────────────────────────────────────────
            val_metrics = self.evaluate()
            history["val"].append(val_metrics)

            elapsed = time.time() - t0

            # ── Print epoch summary ─────────────────────────────────────
            line = (
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"train {train_metrics['loss_total']:.4f} "
                f"(reg {train_metrics['loss_regime']:.4f}, "
                f"trans {train_metrics['loss_transition']:.4f})"
            )

            if val_metrics:
                line += (
                    f" | val {val_metrics['loss_total']:.4f} "
                    f"(acc {val_metrics['regime_accuracy']:.3f}, "
                    f"F1 {val_metrics['regime_macro_f1']:.3f}, "
                    f"AUC {val_metrics['transition_roc_auc']:.3f})"
                )

            line += f" | lr {self._get_lr():.2e} | {elapsed:.1f}s"
            print(line)

            # Per-class accuracy every 5 epochs (or final epoch)
            if val_metrics and (epoch % 5 == 0 or epoch == cfg.epochs):
                pca = val_metrics["regime_per_class_acc"]
                class_str = "  Per-class: " + "  ".join(
                    f"{regime_names[c]}={pca[c]:.3f}"
                    if not np.isnan(pca[c]) else f"{regime_names[c]}=N/A"
                    for c in range(cfg.num_regime_classes)
                )
                print(class_str)
                print(
                    f"  Transition: prec={val_metrics['transition_precision']:.3f}  "
                    f"rec={val_metrics['transition_recall']:.3f}  "
                    f"acc={val_metrics['transition_accuracy']:.3f}"
                )

        print(f"\nTraining complete.  Global optimizer steps: {self.global_step}")
        return history


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test with synthetic data
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from torch_geometric.data import HeteroData

    print("=" * 74)
    print("  Training Engine — Smoke Test (synthetic data)")
    print("=" * 74)

    torch.manual_seed(42)
    np.random.seed(42)

    # ── Config (small for fast test) ────────────────────────────────────
    cfg = RegimeConfig()
    cfg.seq_len = 10             # 10 timesteps (faster)
    cfg.epochs = 3               # 3 epochs
    cfg.batch_size = 2
    cfg.grad_accum_steps = 2     # effective batch = 4
    cfg.warmup_steps = 2
    cfg.lr = 5e-4
    cfg.rgcn_layers = 2
    cfg.lstm_layers = 1

    N = 15                       # nodes per graph
    N_TRAIN = 20                 # training samples
    N_VAL = 8                    # validation samples

    # ── Generate synthetic HeteroData sequences ─────────────────────────
    def make_hetero_data(n_nodes: int) -> HeteroData:
        data = HeteroData()
        data["stock"].x = torch.randn(n_nodes, cfg.node_input_dim)
        data["stock"].num_nodes = n_nodes
        for edge_type_name, e_count in [("correlation", 40), ("etf_cohold", 20), ("supply_chain", 10)]:
            data["stock", edge_type_name, "stock"].edge_index = torch.stack([
                torch.randint(0, n_nodes, (e_count,)),
                torch.randint(0, n_nodes, (e_count,)),
            ])
            data["stock", edge_type_name, "stock"].edge_attr = torch.randn(e_count, cfg.edge_attr_dim)
        return data

    def make_sample():
        """One sample: T snapshots + labels."""
        return {
            "snapshots": [make_hetero_data(N) for _ in range(cfg.seq_len)],
            "regime_label": int(np.random.choice(4, p=[0.5, 0.15, 0.2, 0.15])),
            "transition_label": int(np.random.choice(2, p=[0.8, 0.2])),
            "date": "2024-01-01",
        }

    # Build datasets as lists
    train_samples = [make_sample() for _ in range(N_TRAIN)]
    val_samples = [make_sample() for _ in range(N_VAL)]

    # Use the custom collate function
    try:
        from .data.hetero_dataset import regime_collate_fn
    except ImportError as exc:
        if __package__:
            raise
        from data.hetero_dataset import regime_collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_samples,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=regime_collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_samples,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=regime_collate_fn,
    )

    print(f"\nSynthetic dataset:")
    print(f"  Train samples : {N_TRAIN}")
    print(f"  Val samples   : {N_VAL}")
    print(f"  Nodes/graph   : {N}")
    print(f"  Timesteps     : {cfg.seq_len}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")

    # ── Check label distributions ───────────────────────────────────────
    train_regimes = [s["regime_label"] for s in train_samples]
    train_trans = [s["transition_label"] for s in train_samples]
    print(f"\n  Train regime dist : {np.bincount(train_regimes, minlength=4)}")
    print(f"  Train transition  : {np.bincount(train_trans, minlength=2)}")

    # ── Model ───────────────────────────────────────────────────────────
    model = DynamicRegimeGNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model parameters: {n_params:,}")

    # ── Trainer ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        device="cpu",
    )

    print(f"  Total optimizer steps planned: {trainer.total_steps}")
    print(f"  Warmup steps: {trainer.warmup_steps}")

    # ── Train ───────────────────────────────────────────────────────────
    history = trainer.train()

    # ── Verify training dynamics ────────────────────────────────────────
    print(f"\n{'─' * 74}")
    print(f"  Verification")
    print(f"{'─' * 74}")

    t0_loss = history["train"][0]["loss_total"]
    tf_loss = history["train"][-1]["loss_total"]
    print(f"  Train loss: {t0_loss:.4f} → {tf_loss:.4f}  "
          f"({'↓ decreased' if tf_loss < t0_loss else '→ flat/increased'})")

    if history["val"]:
        v_final = history["val"][-1]
        print(f"  Val loss       : {v_final['loss_total']:.4f}")
        print(f"  Regime accuracy: {v_final['regime_accuracy']:.3f}")
        print(f"  Regime macro-F1: {v_final['regime_macro_f1']:.3f}")
        print(f"  Transition AUC : {v_final['transition_roc_auc']:.3f}")
        print(f"  Transition acc : {v_final['transition_accuracy']:.3f}")

    # Check gradient flow on final epoch
    model.train()
    batch = next(iter(train_loader))
    snapshots = move_snapshots_to_device(batch["snapshots"], torch.device("cpu"))
    regime_logits, trans_logit = model(snapshots)
    loss = regime_logits.sum() + trans_logit.sum()
    loss.backward()
    n_graded = sum(1 for _, p in model.named_parameters()
                   if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for _ in model.parameters())
    print(f"\n  Gradient flow: {n_graded}/{n_total} parameters")
    assert n_graded == n_total, f"Missing gradients on {n_total - n_graded} params!"

    # Check LR schedule
    print(f"  Final LR: {trainer._get_lr():.2e}")
    print(f"  Global steps: {trainer.global_step}")

    print(f"\n{'=' * 74}")
    print(f"  TRAINING ENGINE SMOKE TEST PASSED ✓")
    print(f"{'=' * 74}")
