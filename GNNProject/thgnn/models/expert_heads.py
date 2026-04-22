"""
Relation-Routed Expert Prediction Heads
=========================================
Implements the final prediction layer from Section 4.2.4:

  After the GAT, pairwise edge embeddings are formed:

      u_edge_ij = [h_i^{L_g} || h_j^{L_g} || e_ij^{L_g}]   ∈ R^{1088}
                   (512)        (512)          (64)

  These are routed by edge_type τ_ij ∈ {0=neg, 1=mid, 2=pos} to one of
  three small expert MLP heads.  Each head outputs a scalar Fisher-z
  residual  Δẑ_ij.  The final predicted correlation is:

      ρ̂_ij = tanh(z_base_ij + Δẑ_ij)

  This routing allows the model to learn distinct dynamics for negative,
  neutral, and positive correlation regimes.

  Weight init: Xavier-uniform + zero bias (Appendix A.1).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from ..config import THGNNConfig
except ImportError as exc:
    if __package__:
        raise
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import THGNNConfig


class ExpertMLP(nn.Module):
    """
    A single expert MLP head for one correlation regime.

    Architecture:
        Linear(input_dim → hidden_dim) → GELU → Dropout →
        Linear(hidden_dim → hidden_dim) → GELU → Dropout →
        Linear(hidden_dim → 1)

    The extra hidden layer gives each expert enough capacity to capture
    the distinct non-linear relationships within its correlation regime.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),               # scalar Δẑ output
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (E_k, input_dim)   — edge embeddings for this regime

        Returns
        -------
        (E_k,)  — scalar Δẑ per edge
        """
        return self.mlp(x).squeeze(-1)


class ExpertPredictionHeads(nn.Module):
    """
    Three relation-routed expert MLPs for neg / mid / pos correlation regimes.

    Input:
        node_embed  : (N, 512)       — final GAT node embeddings
        edge_state  : (E, 64)        — final GAT edge states
        edge_index  : (2, E)         — COO edges
        edge_type   : (E,)           — {0, 1, 2}
        baseline_z  : (E,)           — atanh(ρ_base)

    Output:
        delta_z_pred : (E,)          — predicted Fisher-z residual Δẑ
        rho_pred     : (E,)          — predicted correlation = tanh(z_base + Δẑ)
    """

    def __init__(self, cfg: Optional[THGNNConfig] = None):
        super().__init__()
        cfg = THGNNConfig() if cfg is None else cfg
        self.cfg = cfg

        # Input dim = h_i (512) + h_j (512) + e_ij (64) = 1088
        input_dim = cfg.gat_hidden_dim * 2 + cfg.edge_state_dim

        # Three expert heads — one per edge type
        self.experts = nn.ModuleList([
            ExpertMLP(
                input_dim=input_dim,            # 1088
                hidden_dim=cfg.expert_hidden_dim,  # 256
                dropout=cfg.gat_dropout,
            )
            for _ in range(cfg.num_expert_heads)   # 3
        ])

    def forward(
        self,
        node_embed: torch.Tensor,
        edge_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        baseline_z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        node_embed  : (N, 512)
        edge_state  : (E, 64)
        edge_index  : (2, E)
        edge_type   : (E,)        ∈ {0, 1, 2}
        baseline_z  : (E,)        = atanh(ρ_base)

        Returns
        -------
        delta_z_pred : (E,)   — predicted Δẑ
        rho_pred     : (E,)   — tanh(z_base + Δẑ) ∈ (-1, 1)
        """
        E = edge_index.shape[1]
        src, dst = edge_index                          # (E,), (E,)

        # ── Compose pairwise edge embeddings ──────────────────────────
        u_edge = torch.cat([
            node_embed[src],                           # (E, 512)
            node_embed[dst],                           # (E, 512)
            edge_state,                                # (E, 64)
        ], dim=-1)                                     # (E, 1088)

        # ── Route through expert heads by edge_type ───────────────────
        delta_z_pred = torch.zeros(E, device=u_edge.device, dtype=u_edge.dtype)

        for etype in range(self.cfg.num_expert_heads):
            mask = (edge_type == etype)                # (E,) bool
            if mask.any():
                delta_z_pred[mask] = self.experts[etype](u_edge[mask])

        # ── Map to correlation space ──────────────────────────────────
        z_pred = baseline_z + delta_z_pred             # (E,)
        rho_pred = torch.tanh(z_pred)                  # (E,)  ∈ (-1, 1)

        return delta_z_pred, rho_pred


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = THGNNConfig()

    model = ExpertPredictionHeads(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    input_dim = cfg.gat_hidden_dim * 2 + cfg.edge_state_dim
    print(f"ExpertPredictionHeads parameters: {n_params:,}")
    print(f"  Input dim     : {input_dim}  (512 + 512 + 64)")
    print(f"  Hidden dim    : {cfg.expert_hidden_dim}")
    print(f"  Expert heads  : {cfg.num_expert_heads}")

    # ── Synthetic inputs ──────────────────────────────────────────────
    N, E = 50, 400
    node_embed = torch.randn(N, cfg.gat_hidden_dim)           # (50, 512)
    edge_state = torch.randn(E, cfg.edge_state_dim)           # (400, 64)
    edge_index = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ])                                                         # (2, 400)
    edge_type = torch.randint(0, 3, (E,))                     # (400,)
    baseline_z = torch.randn(E) * 0.5                          # (400,)

    # ── Verify routing coverage ───────────────────────────────────────
    for et in range(3):
        cnt = (edge_type == et).sum().item()
        print(f"  edge_type={et} count: {cnt}")

    # ── Forward pass ──────────────────────────────────────────────────
    delta_z, rho = model(node_embed, edge_state, edge_index, edge_type, baseline_z)

    print(f"\n── Forward pass ──")
    print(f"  delta_z_pred : {tuple(delta_z.shape)}  — expected ({E},)")
    print(f"  rho_pred     : {tuple(rho.shape)}  — expected ({E},)")
    print(f"  rho range    : [{rho.min().item():.4f}, {rho.max().item():.4f}]  (should be in (-1, 1))")

    assert delta_z.shape == (E,)
    assert rho.shape == (E,)
    assert (rho > -1).all() and (rho < 1).all(), "Correlations outside (-1, 1)!"
    assert torch.isfinite(delta_z).all()
    assert torch.isfinite(rho).all()

    # ── Gradient flow ─────────────────────────────────────────────────
    node_embed_g = node_embed.clone().requires_grad_(True)
    edge_state_g = edge_state.clone().requires_grad_(True)
    dz, rp = model(node_embed_g, edge_state_g, edge_index, edge_type, baseline_z)
    loss = dz.sum()
    loss.backward()
    assert node_embed_g.grad is not None and node_embed_g.grad.abs().sum() > 0
    assert edge_state_g.grad is not None and edge_state_g.grad.abs().sum() > 0
    print(f"\n── Gradient flow ──")
    print(f"  ∂loss/∂node_embed norm : {node_embed_g.grad.norm().item():.6f}")
    print(f"  ∂loss/∂edge_state norm : {edge_state_g.grad.norm().item():.6f}")

    print(f"\n✓ ExpertPredictionHeads smoke test passed.")
