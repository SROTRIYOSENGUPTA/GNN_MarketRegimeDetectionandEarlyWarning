"""
THGNN — Master Wrapper Model
==============================
Composes the three stages of the Temporal-Heterogeneous Graph Neural
Network (Fanshawe, Masih & Cameron, 2026 — arXiv:2601.04602):

    Stage 1 — TemporalEncoder
        Input  :  x  (N, L=30, F=37)
        Output :  h0  (N, 512)

    Stage 2 — RelationalEncoder  (edge-aware GAT, 3 layers × 4 heads)
        Input  :  h0, edge_index, edge_attr, edge_type
        Output :  node_embed (N, 512),  edge_state (E, 64)

    Stage 3 — ExpertPredictionHeads  (neg / mid / pos expert MLPs)
        Input  :  node_embed, edge_state, edge_index, edge_type, baseline_z
        Output :  delta_z_pred (E,),  rho_pred (E,)

Overall:
    (N, 30, 37) + graph → (E,) predicted Δẑ  and  (E,) predicted ρ̂

Weight init: Xavier-uniform + zero bias throughout (Appendix A.1).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THGNNConfig
from models.temporal_encoder import TemporalEncoder
from models.relational_encoder import RelationalEncoder
from models.expert_heads import ExpertPredictionHeads


class THGNN(nn.Module):
    """
    Full Temporal-Heterogeneous Graph Neural Network.

    A single forward pass runs:
        x → TemporalEncoder → RelationalEncoder → ExpertHeads → (Δẑ, ρ̂)
    """

    def __init__(self, cfg: THGNNConfig = THGNNConfig()):
        super().__init__()
        self.cfg = cfg

        self.temporal_encoder = TemporalEncoder(cfg)
        self.relational_encoder = RelationalEncoder(cfg)
        self.expert_heads = ExpertPredictionHeads(cfg)

    # ──────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
        baseline_z: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x            : (N, 30, 37)   — per-stock feature sequences
        edge_index   : (2, E)        — COO edges
        edge_attr    : (E, 5)        — [ρ_base, |ρ_base|, sign, same_sector, same_subind]
        edge_type    : (E,)          — {0=neg, 1=mid, 2=pos}
        baseline_z   : (E,)          — atanh(ρ_base)
        padding_mask : (N, 30) bool, optional — True for padded time-steps

        Returns
        -------
        delta_z_pred : (E,)   — predicted Fisher-z residual Δẑ
        rho_pred     : (E,)   — predicted correlation tanh(z_base + Δẑ) ∈ (-1, 1)
        """
        # Stage 1: Temporal Encoder
        h0 = self.temporal_encoder(x, padding_mask)         # (N, 512)

        # Stage 2: Relational Encoder (edge-aware GAT)
        node_embed, edge_state = self.relational_encoder(
            h0, edge_index, edge_attr, edge_type,
        )                                                    # (N, 512), (E, 64)

        # Stage 3: Expert Prediction Heads
        delta_z_pred, rho_pred = self.expert_heads(
            node_embed, edge_state, edge_index,
            edge_type, baseline_z,
        )                                                    # (E,), (E,)

        return delta_z_pred, rho_pred

    # ──────────────────────────────────────────────────────────────────
    def parameter_groups(self) -> list:
        """
        Build parameter groups with bias / LayerNorm excluded from weight
        decay, as specified in Appendix A.1.

        Returns a list of dicts suitable for `torch.optim.AdamW(params=...)`.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude biases and LayerNorm parameters from weight decay
            if "bias" in name or "layer_norm" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = THGNNConfig()

    model = THGNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"THGNN total parameters: {n_params:,}\n")

    # Synthetic graph
    N, E = 50, 400
    x = torch.randn(N, cfg.seq_len, cfg.num_features)     # (50, 30, 37)
    edge_index = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ])
    edge_attr = torch.randn(E, cfg.num_edge_attr)          # (400, 5)
    edge_type = torch.randint(0, 3, (E,))                  # (400,)
    baseline_z = torch.randn(E) * 0.5                       # (400,)

    # Forward
    delta_z, rho = model(x, edge_index, edge_attr, edge_type, baseline_z)
    print(f"Input  x       : {tuple(x.shape)}")
    print(f"Output Δẑ      : {tuple(delta_z.shape)}")
    print(f"Output ρ̂       : {tuple(rho.shape)}")
    print(f"ρ̂ range        : [{rho.min().item():.4f}, {rho.max().item():.4f}]")

    assert delta_z.shape == (E,)
    assert rho.shape == (E,)
    assert torch.isfinite(delta_z).all()

    # Gradient flow
    loss = delta_z.sum()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No grad on {name}"
    print(f"\n✓ All {n_params:,} parameters receive gradients.")

    # Parameter groups
    groups = model.parameter_groups()
    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_nodecay = sum(p.numel() for p in groups[1]["params"])
    print(f"\nParameter groups (for AdamW):")
    print(f"  With weight decay    : {n_decay:>10,}  (wd={groups[0]['weight_decay']})")
    print(f"  Without weight decay : {n_nodecay:>10,}  (bias + LayerNorm)")
    print(f"  Total                : {n_decay + n_nodecay:>10,}")

    print(f"\n✓ THGNN wrapper smoke test passed.")
