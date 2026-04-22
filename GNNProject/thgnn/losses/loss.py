"""
THGNN Loss Functions
=====================
Implements the combined loss from Section 4.2.5:

  L_total = L_edge  +  s · L_hist

  where s = 7.0 (hist_scale) ensures the two terms are comparable in magnitude.

  ── L_edge (Huber / Smooth-L1) ────────────────────────────────────────

      L_edge = Huber(ẑ_pred, z_target)

      Applied in Fisher-z space.  Quadratic for small residuals, linear
      for large ones → robust to regime-shock outliers.

  ── L_hist (Gaussian Soft-Histogram Matching) ─────────────────────────

      Prevents mode collapse by forcing the predicted correlation
      distribution to match the target distribution.

      Gaussian soft binning (Karandikar et al., 2021):

          w(n, b) = exp( -(x_n - c_b)² / (2σ²) )

          h_b = (1/N)  Σ_n  w(n,b) / Σ_{b'} w(n,b')

          L_hist = Σ_b (h_pred[b] - h_true[b])²

      Applied four times:
        • 3 × per-edge-type histograms  (6 bins each)
        • 1 × global histogram          (15 bins)
      All four are equally weighted, then multiplied by s = 7.

  The total loss gives equal weight to edge and histogram terms.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THGNNConfig


# ═══════════════════════════════════════════════════════════════════════════
# Gaussian Soft-Histogram
# ═══════════════════════════════════════════════════════════════════════════
def soft_histogram(
    values: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
) -> torch.Tensor:
    """
    Compute a differentiable soft histogram via Gaussian kernel binning.

    Parameters
    ----------
    values      : (M,)  — 1-D tensor of scalar values (e.g., predicted ρ)
    bin_centers : (B,)  — 1-D tensor of bin center positions
    sigma       : float — Gaussian bandwidth

    Returns
    -------
    hist : (B,)  — normalised soft histogram (sums to ≈ 1)

    Math (Section 4.2.5):
        w(n, b) = exp( -(x_n - c_b)² / (2σ²) )
        h_b     = (1/N) Σ_n  w(n,b) / Σ_{b'} w(n,b')
    """
    M = values.shape[0]
    if M == 0:
        return torch.zeros_like(bin_centers)

    # (M, B):  distance of each value to each bin center
    diff = values.unsqueeze(1) - bin_centers.unsqueeze(0)      # (M, B)
    weights = torch.exp(-diff.pow(2) / (2 * sigma ** 2))       # (M, B)

    # Normalise per sample: each sample's weights sum to 1 across bins
    weights_norm = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)  # (M, B)

    # Average across samples → soft histogram
    hist = weights_norm.mean(dim=0)                            # (B,)

    return hist


def make_bin_centers(
    targets: torch.Tensor,
    num_bins: int,
) -> torch.Tensor:
    """
    Create evenly spaced bin centers spanning the target range.

    The paper says "bin centers spanning the batch target range"
    (Section 4.2.5), so we set endpoints from the observed target min/max.
    """
    lo = targets.min().item()
    hi = targets.max().item()
    # Small padding to avoid edge effects
    margin = (hi - lo) * 0.05 + 1e-6
    return torch.linspace(lo - margin, hi + margin, num_bins, device=targets.device)


# ═══════════════════════════════════════════════════════════════════════════
# Combined THGNN Loss
# ═══════════════════════════════════════════════════════════════════════════
class THGNNLoss(nn.Module):
    """
    Combined loss:  L_total = L_edge  +  s · L_hist

    Parameters
    ----------
    cfg : THGNNConfig — provides all loss hyperparameters.
    """

    def __init__(self, cfg: THGNNConfig = THGNNConfig()):
        super().__init__()
        self.cfg = cfg
        self.huber = nn.SmoothL1Loss(reduction="mean", beta=cfg.huber_delta)

    def forward(
        self,
        delta_z_pred: torch.Tensor,
        rho_pred: torch.Tensor,
        target_z_resid: torch.Tensor,
        baseline_z: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ----------
        delta_z_pred    : (E,)  — predicted Δẑ from expert heads
        rho_pred        : (E,)  — predicted ρ = tanh(z_base + Δẑ)
        target_z_resid  : (E,)  — target Δz  (z_future - z_base)
        baseline_z      : (E,)  — z_base = atanh(ρ_base)
        edge_type       : (E,)  — {0, 1, 2}

        Returns
        -------
        total_loss : scalar tensor
        log_dict   : dict of scalar tensors for logging
        """
        cfg = self.cfg

        # ══════════════════════════════════════════════════════════════
        # 1.  L_edge  —  Huber loss in Fisher-z space
        # ══════════════════════════════════════════════════════════════
        #
        #   Predicted full z  = z_base + Δẑ_pred
        #   Target full z     = z_base + Δz_target
        #   Huber(pred_z, target_z) = Huber(Δẑ, Δz)  (z_base cancels)
        #
        loss_edge = self.huber(delta_z_pred, target_z_resid)

        # ══════════════════════════════════════════════════════════════
        # 2.  L_hist  —  Soft-histogram distribution matching
        # ══════════════════════════════════════════════════════════════
        #
        #   Operates in *correlation* space (ρ), not z-space.
        #   Reconstructs target correlations from z_base + Δz_target.
        #
        rho_target = torch.tanh(baseline_z + target_z_resid)   # (E,)

        sigma = cfg.hist_sigma
        hist_losses = []

        # ── 2a.  Per-edge-type histograms (6 bins each) ──────────────
        for etype in range(cfg.num_edge_types):                # 0, 1, 2
            mask = (edge_type == etype)
            if mask.sum() < 2:
                # Not enough edges of this type — skip (contributes 0)
                continue

            pred_sub = rho_pred[mask]
            true_sub = rho_target[mask]

            bins = make_bin_centers(true_sub, cfg.hist_bins_per_type)  # (6,)

            h_pred = soft_histogram(pred_sub, bins, sigma)     # (6,)
            h_true = soft_histogram(true_sub.detach(), bins, sigma)  # (6,)

            hist_losses.append((h_pred - h_true).pow(2).sum())

        # ── 2b.  Global histogram (15 bins) ───────────────────────────
        bins_global = make_bin_centers(rho_target, cfg.hist_bins_global)  # (15,)

        h_pred_global = soft_histogram(rho_pred, bins_global, sigma)     # (15,)
        h_true_global = soft_histogram(rho_target.detach(), bins_global, sigma)

        hist_losses.append((h_pred_global - h_true_global).pow(2).sum())

        # ── Average the 4 histogram losses (equal weight), then × s ──
        if len(hist_losses) > 0:
            loss_hist = torch.stack(hist_losses).mean()
        else:
            loss_hist = torch.tensor(0.0, device=delta_z_pred.device)

        loss_hist_scaled = cfg.hist_scale * loss_hist          # s = 7

        # ══════════════════════════════════════════════════════════════
        # 3.  L_total = L_edge + s · L_hist
        # ══════════════════════════════════════════════════════════════
        total_loss = loss_edge + loss_hist_scaled

        # ── Logging dict ──────────────────────────────────────────────
        log_dict = {
            "loss_total": total_loss.detach(),
            "loss_edge": loss_edge.detach(),
            "loss_hist": loss_hist.detach(),
            "loss_hist_scaled": loss_hist_scaled.detach(),
            "rho_pred_mean": rho_pred.detach().mean(),
            "rho_pred_std": rho_pred.detach().std(),
            "rho_target_mean": rho_target.detach().mean(),
            "rho_target_std": rho_target.detach().std(),
        }

        return total_loss, log_dict


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = THGNNConfig()

    loss_fn = THGNNLoss(cfg)
    print(f"THGNNLoss configuration:")
    print(f"  Huber delta        : {cfg.huber_delta}")
    print(f"  Hist bins/type     : {cfg.hist_bins_per_type}")
    print(f"  Hist bins global   : {cfg.hist_bins_global}")
    print(f"  Hist sigma         : {cfg.hist_sigma}")
    print(f"  Hist scale (s)     : {cfg.hist_scale}")

    # ── Synthetic data ────────────────────────────────────────────────
    E = 400
    delta_z_pred = torch.randn(E, requires_grad=True) * 0.3
    baseline_z = torch.randn(E) * 0.5
    rho_pred = torch.tanh(baseline_z + delta_z_pred)
    target_z_resid = torch.randn(E) * 0.3
    edge_type = torch.randint(0, 3, (E,))

    for et in range(3):
        print(f"  edge_type={et} count: {(edge_type == et).sum().item()}")

    # ── Forward ───────────────────────────────────────────────────────
    total_loss, log = loss_fn(delta_z_pred, rho_pred, target_z_resid, baseline_z, edge_type)

    print(f"\n── Loss values ──")
    for k, v in log.items():
        print(f"  {k:20s} : {v.item():.6f}")

    assert total_loss.shape == ()
    assert torch.isfinite(total_loss)
    assert total_loss > 0, "Loss should be > 0 for random predictions"

    # ── Gradient flow ─────────────────────────────────────────────────
    # delta_z_pred is a leaf, but rho_pred is derived from it via tanh,
    # so we retain_grad on rho_pred and check delta_z_pred directly.
    delta_z_pred.retain_grad()
    rho_pred.retain_grad()
    total_loss.backward()
    assert delta_z_pred.grad is not None
    assert delta_z_pred.grad.abs().sum() > 0
    print(f"\n── Gradient flow ──")
    print(f"  ∂L/∂delta_z norm : {delta_z_pred.grad.norm().item():.6f}")
    print(f"  ∂L/∂rho_pred norm: {rho_pred.grad.norm().item():.6f}")

    # ── Verify histogram differentiability ────────────────────────────
    # Make predictions exactly match targets → hist loss should be ~0
    delta_z_perfect = target_z_resid.clone().detach().requires_grad_(True)
    rho_perfect = torch.tanh(baseline_z + delta_z_perfect)
    loss_perfect, log_perfect = loss_fn(
        delta_z_perfect, rho_perfect, target_z_resid, baseline_z, edge_type
    )
    print(f"\n── Perfect prediction sanity check ──")
    print(f"  loss_edge (should be ~0) : {log_perfect['loss_edge'].item():.8f}")
    print(f"  loss_hist (should be ~0) : {log_perfect['loss_hist'].item():.8f}")
    print(f"  loss_total               : {log_perfect['loss_total'].item():.8f}")
    assert log_perfect["loss_edge"].item() < 1e-6, "Edge loss should be ~0 for perfect preds"
    assert log_perfect["loss_hist"].item() < 1e-4, "Hist loss should be ~0 for perfect preds"

    print(f"\n✓ THGNNLoss smoke test passed.")
