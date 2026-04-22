"""
Relational Encoder  (Edge-Aware GAT)
======================================
Implements the Graph Attention Network described in Section 4.2.4:

  Input:
      h_i^(0)   ∈ R^512        — per-stock node embeddings from Temporal Encoder
      edge_index (2, E)         — COO sparse edge list
      edge_attr  (E, 5)         — [ρ_base, |ρ_base|, sign, same_sector, same_subind]
      edge_type  (E,)           — {0=neg, 1=mid, 2=pos} relation class

  Architecture (per layer l = 0..2):
      For each edge (i,j) and head h, compute an edge-conditioned gate:

          m_{ij}^{l,h} = E_type^{h}(τ_ij) + W_f^{h} f_ij + W_a^{h} a_{ij} + W_s^{h} e_{ij}^{l}

      where:
          τ_ij ∈ {0,1,2}     — discrete relation class → learned embedding
          f_ij  ∈ R^2         — [same_sector, same_subind] binary flags
          a_ij  ∈ R^3         — [ρ_base, |ρ_base|, sign_indicator] continuous attrs
          e_ij^{l} ∈ R^d_e   — persistent edge state (propagates across layers)

      Attention coefficients are computed from concatenated projected node
      features and the edge gate via LeakyReLU → softmax over N(i).

      All heads are concatenated → linear projection → residual + LayerNorm.

      Edge states are updated in parallel via a residual MLP:
          e_ij^{l+1} = e_ij^{l} + MLP([h_i^{l+1} || h_j^{l+1} || e_ij^{l}])

  Output:
      h_i^{L_g}  ∈ R^512       — updated node embeddings
      e_ij^{L_g} ∈ R^d_e       — final edge states

  The downstream prediction head forms:
      u_edge_ij = [h_i^{L_g} || h_j^{L_g} || e_ij^{L_g}]   ∈ R^{512+512+d_e}
  and routes by edge_type to expert MLPs (implemented separately).

  Weight init: Xavier-uniform weights, zero biases (Appendix A.1).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax as pyg_softmax
from torch_geometric.nn import MessagePassing

try:
    from ..config import THGNNConfig
except ImportError as exc:
    if __package__:
        raise
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import THGNNConfig


# ═══════════════════════════════════════════════════════════════════════════
# Single Edge-Aware GAT Layer
# ═══════════════════════════════════════════════════════════════════════════
class EdgeAwareGATLayer(MessagePassing):
    """
    One layer of the edge-conditioned Graph Attention Network.

    Implements the full message-passing cycle described in Section 4.2.4:
      1. Per-head node projection
      2. Edge-conditioned gate construction
      3. LeakyReLU attention with softmax normalisation
      4. Weighted neighbour aggregation
      5. Head concatenation → linear projection → residual + LayerNorm
      6. Edge state residual update

    Parameters
    ----------
    embed_dim      : int   — node embedding dimension (512)
    num_heads      : int   — attention heads per layer (4)
    edge_attr_dim  : int   — continuous edge attribute dimension (3: ρ, |ρ|, sign)
    edge_flag_dim  : int   — binary edge flag dimension (2: same_sector, same_subind)
    edge_state_dim : int   — persistent edge state dimension (64)
    num_edge_types : int   — discrete relation types (3: neg/mid/pos)
    dropout        : float — attention & FFN dropout (0.2)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 4,
        edge_attr_dim: int = 3,
        edge_flag_dim: int = 2,
        edge_state_dim: int = 64,
        num_edge_types: int = 3,
        dropout: float = 0.2,
    ):
        # aggregate neighbour messages by summing (attention-weighted)
        # node_dim=0 tells PyG that node features are 2D: (N, F)
        super().__init__(aggr="add", flow="source_to_target", node_dim=0)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads        # 512 / 4 = 128
        self.edge_state_dim = edge_state_dim
        self.dropout = dropout

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        # ── Per-head node projections: Q, K, V ────────────────────────
        # We project source and target nodes independently per head.
        # For attention scoring we use [W_src h_i || W_dst h_j || m_ij]
        self.W_src = nn.Linear(embed_dim, embed_dim, bias=False)    # → (H * d_h)
        self.W_dst = nn.Linear(embed_dim, embed_dim, bias=False)    # → (H * d_h)
        self.W_val = nn.Linear(embed_dim, embed_dim, bias=False)    # value projection

        # ── Edge-conditioned gate components (per-head) ───────────────
        # m_{ij}^{l,h} = E_type(τ) + W_f f_ij + W_a a_ij + W_s e_ij^{l}
        #
        # Each component projects to (num_heads * head_dim) so we can
        # reshape to (E, H, d_h) and add element-wise.

        # E_type: learnable embedding per relation class per head
        self.edge_type_embed = nn.Embedding(num_edge_types, num_heads * self.head_dim)

        # W_f: project 2-dim binary flags → (H * d_h)
        self.W_f = nn.Linear(edge_flag_dim, num_heads * self.head_dim, bias=False)

        # W_a: project 3-dim continuous edge attrs → (H * d_h)
        self.W_a = nn.Linear(edge_attr_dim, num_heads * self.head_dim, bias=False)

        # W_s: project persistent edge state → (H * d_h)
        self.W_s = nn.Linear(edge_state_dim, num_heads * self.head_dim, bias=False)

        # ── Attention scoring vector (per head) ───────────────────────
        # Score = a^T [W_src h_i || W_dst h_j || m_ij]  →  scalar per head
        # Concatenation of 3 head-dim vectors → 3 * head_dim
        self.attn_vec = nn.Parameter(torch.empty(num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_vec)

        # ── Output projection + residual + LayerNorm ──────────────────
        # After concatenating H heads: (H * d_h) = node_dim → node_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # ── Edge state residual MLP ───────────────────────────────────
        # e_ij^{l+1} = e_ij^{l} + MLP([h_i^{l+1} || h_j^{l+1} || e_ij^{l}])
        # Input: 512 + 512 + edge_state_dim → edge_state_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + edge_state_dim, edge_state_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_state_dim * 2, edge_state_dim),
        )
        self.edge_norm = nn.LayerNorm(edge_state_dim)

        self._init_weights()

    def _init_weights(self):
        """Xavier-uniform for Linear weights; zero for biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_cont: torch.Tensor,
        edge_flags: torch.Tensor,
        edge_type: torch.Tensor,
        edge_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x              : (N, 512)     — node embeddings
        edge_index     : (2, E)       — COO edges
        edge_attr_cont : (E, 3)       — [ρ_base, |ρ_base|, sign_indicator]
        edge_flags     : (E, 2)       — [same_sector, same_subind]
        edge_type      : (E,)         — {0, 1, 2}
        edge_state     : (E, d_e)     — persistent edge states from previous layer

        Returns
        -------
        x_new          : (N, 512)     — updated node embeddings
        edge_state_new : (E, d_e)     — updated edge states
        """
        H, d_h = self.num_heads, self.head_dim
        E = edge_index.shape[1]

        # ── 1. Project nodes ──────────────────────────────────────────
        h_src = self.W_src(x).view(-1, H, d_h)     # (N, H, d_h)
        h_dst = self.W_dst(x).view(-1, H, d_h)     # (N, H, d_h)
        h_val = self.W_val(x).view(-1, H, d_h)     # (N, H, d_h)

        # ── 2. Build edge-conditioned gate ────────────────────────────
        # m_{ij} = E_type(τ) + W_f f + W_a a + W_s e
        m_type = self.edge_type_embed(edge_type).view(E, H, d_h)     # (E, H, d_h)
        m_flag = self.W_f(edge_flags).view(E, H, d_h)                # (E, H, d_h)
        m_attr = self.W_a(edge_attr_cont).view(E, H, d_h)            # (E, H, d_h)
        m_state = self.W_s(edge_state).view(E, H, d_h)               # (E, H, d_h)

        edge_gate = m_type + m_flag + m_attr + m_state                # (E, H, d_h)

        # Store pre-computed tensors for message passing
        self._h_src = h_src
        self._h_dst = h_dst
        self._h_val = h_val
        self._edge_gate = edge_gate

        # ── 3. Message passing (computes attention + aggregation) ─────
        out = self.propagate(
            edge_index,
            size=None,
            x=x,
        )
        # out shape: (N, H * d_h) = (N, 512)

        # ── 4. Output projection + residual + LayerNorm ──────────────
        out = self.out_proj(out)                      # (N, 512)
        out = self.attn_dropout(out)
        x_new = self.layer_norm(x + out)              # residual + LN

        # ── 5. Edge state update (residual MLP) ──────────────────────
        src_idx, dst_idx = edge_index                 # (E,), (E,)
        edge_input = torch.cat([
            x_new[src_idx],                           # (E, 512)
            x_new[dst_idx],                           # (E, 512)
            edge_state,                               # (E, d_e)
        ], dim=-1)                                    # (E, 512+512+d_e)

        edge_state_new = self.edge_norm(
            edge_state + self.edge_mlp(edge_input)    # residual
        )                                             # (E, d_e)

        # Clean up cached tensors
        del self._h_src, self._h_dst, self._h_val, self._edge_gate

        return x_new, edge_state_new

    # ── PyG message passing hooks ─────────────────────────────────────
    def message(
        self,
        x_j: torch.Tensor,       # source node features (unused — we use cached)
        x_i: torch.Tensor,       # target node features (unused — we use cached)
        index: torch.Tensor,     # target node indices (for softmax)
        size_i: int,             # number of target nodes
        edge_index_i: torch.Tensor,   # source indices
        edge_index_j: torch.Tensor,   # target indices
    ) -> torch.Tensor:
        """
        Compute attention-weighted messages.

        For each edge (j → i):
          1. Gather projected source h_src[j] and target h_dst[i]
          2. Concatenate with edge gate: [h_src_j || h_dst_i || m_ij]
          3. Score via dot product with attn_vec → LeakyReLU
          4. Softmax over N(i) → attention coefficient α_ij
          5. Return α_ij * h_val[j] (weighted value)
        """
        H, d_h = self.num_heads, self.head_dim
        # Note: in PyG source_to_target flow:
        #   edge_index[0] = source (j), edge_index[1] = target (i)
        #   x_j are source features, x_i are target features
        #   index = edge_index[1] (target indices for softmax grouping)

        # Use edge ordering to index into cached per-head projections
        # edge_index_j = source node indices, edge_index_i = target node indices
        # But PyG provides x_j/x_i which are already gathered. We need the
        # per-head versions, so we re-gather from cached projections.

        # Source and target per-head projections for this edge set
        # We need the original node indices. In source_to_target:
        #   message is called with x_j = x[edge_index[0]], x_i = x[edge_index[1]]
        # But we stored per-head projections. Let's use the raw edge indices.
        h_src_j = self._h_src[edge_index_j]     # (E, H, d_h)
        h_dst_i = self._h_dst[edge_index_i]     # (E, H, d_h)
        h_val_j = self._h_val[edge_index_j]     # (E, H, d_h)
        edge_gate = self._edge_gate              # (E, H, d_h)

        # ── Attention score ───────────────────────────────────────────
        # Concatenate: [h_src_j || h_dst_i || m_ij] → (E, H, 3*d_h)
        attn_input = torch.cat([h_src_j, h_dst_i, edge_gate], dim=-1)

        # Dot product with per-head attention vector: (H, 3*d_h)
        # Result: (E, H)
        e = (attn_input * self.attn_vec.unsqueeze(0)).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=0.2)

        # ── Softmax over neighbourhood of each target node ────────────
        # index = target node indices (E,) — softmax groups by target
        alpha = pyg_softmax(e, index)             # (E, H)
        alpha = self.attn_dropout(alpha)

        # ── Weighted message: α_ij * h_val_j ─────────────────────────
        msg = alpha.unsqueeze(-1) * h_val_j       # (E, H, d_h)

        # Flatten heads: (E, H * d_h) = (E, 512)
        return msg.reshape(-1, self.num_heads * self.head_dim)

    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Identity — residual + norm done in forward()."""
        return aggr_out                            # (N, 512)


# ═══════════════════════════════════════════════════════════════════════════
# Full Relational Encoder (3-layer GAT stack)
# ═══════════════════════════════════════════════════════════════════════════
class RelationalEncoder(nn.Module):
    """
    Complete edge-aware GAT stack (Section 4.2.4).

    Stacks L_g = 3 EdgeAwareGATLayers. Persistent edge states e_ij are
    initialised to zeros and propagated / updated through each layer.

    Input:
        h0          : (N, 512)    — node embeddings from TemporalEncoder
        edge_index  : (2, E)      — COO edges
        edge_attr   : (E, 5)      — [ρ_base, |ρ_base|, sign, same_sector, same_subind]
        edge_type   : (E,)        — {0, 1, 2}

    Output:
        node_embed  : (N, 512)    — final node embeddings h_i^{L_g}
        edge_embed  : (E, d_e)    — final edge states e_ij^{L_g}
    """

    def __init__(self, cfg: Optional[THGNNConfig] = None):
        super().__init__()
        cfg = THGNNConfig() if cfg is None else cfg
        self.cfg = cfg

        # Continuous edge attributes: ρ_base, |ρ_base|, sign → dim 3
        # Binary edge flags: same_sector, same_subind → dim 2
        self.edge_attr_dim = 3
        self.edge_flag_dim = 2

        # ── Stack of GAT layers ───────────────────────────────────────
        self.gat_layers = nn.ModuleList([
            EdgeAwareGATLayer(
                embed_dim=cfg.gat_hidden_dim,       # 512
                num_heads=cfg.gat_heads,             # 4
                edge_attr_dim=self.edge_attr_dim,    # 3
                edge_flag_dim=self.edge_flag_dim,    # 2
                edge_state_dim=cfg.edge_state_dim,   # 64
                num_edge_types=cfg.num_edge_types,   # 3
                dropout=cfg.gat_dropout,             # 0.2
            )
            for _ in range(cfg.gat_layers)           # 3 layers
        ])

    def forward(
        self,
        h0: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h0         : (N, 512)   — from TemporalEncoder
        edge_index : (2, E)
        edge_attr  : (E, 5)     — [ρ_base, |ρ_base|, sign, same_sector, same_subind]
        edge_type  : (E,)       — {0, 1, 2}

        Returns
        -------
        node_embed : (N, 512)   — h_i^{L_g}
        edge_state : (E, d_e)   — e_ij^{L_g}

        Intermediate shapes per layer:
            h          : (N, 512)
            edge_state : (E, 64)
        """
        E = edge_index.shape[1]

        # ── Split edge_attr into continuous attrs and binary flags ────
        # edge_attr columns: [ρ_base, |ρ_base|, sign, same_sector, same_subind]
        edge_attr_cont = edge_attr[:, :3]             # (E, 3)
        edge_flags = edge_attr[:, 3:]                 # (E, 2)

        # ── Initialise persistent edge states to zero ─────────────────
        edge_state = torch.zeros(
            E, self.cfg.edge_state_dim,
            device=h0.device, dtype=h0.dtype,
        )                                             # (E, 64)

        # ── Forward through L_g = 3 GAT layers ───────────────────────
        h = h0
        for layer in self.gat_layers:
            h, edge_state = layer(
                x=h,
                edge_index=edge_index,
                edge_attr_cont=edge_attr_cont,
                edge_flags=edge_flags,
                edge_type=edge_type,
                edge_state=edge_state,
            )
            # h: (N, 512),  edge_state: (E, 64)

        return h, edge_state


# ═══════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    torch.manual_seed(42)
    cfg = THGNNConfig()

    model = RelationalEncoder(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"RelationalEncoder parameters: {n_params:,}")
    print(f"  GAT layers : {cfg.gat_layers}")
    print(f"  GAT heads  : {cfg.gat_heads}")
    print(f"  Head dim   : {cfg.gat_hidden_dim // cfg.gat_heads}")
    print(f"  Edge state : {cfg.edge_state_dim}")

    # ── Synthetic graph ───────────────────────────────────────────────
    N = 50                                        # 50 stocks
    E = 400                                       # 400 edges

    h0 = torch.randn(N, cfg.node_embed_dim)       # (50, 512) — from TemporalEncoder
    edge_index = torch.stack([
        torch.randint(0, N, (E,)),
        torch.randint(0, N, (E,)),
    ])                                             # (2, 400)

    edge_attr = torch.randn(E, cfg.num_edge_attr)  # (400, 5)
    edge_type = torch.randint(0, 3, (E,))          # (400,)

    print(f"\n── Forward pass ──")
    print(f"  h0         : {tuple(h0.shape)}")
    print(f"  edge_index : {tuple(edge_index.shape)}")
    print(f"  edge_attr  : {tuple(edge_attr.shape)}")
    print(f"  edge_type  : {tuple(edge_type.shape)}")

    node_embed, edge_state = model(h0, edge_index, edge_attr, edge_type)

    print(f"\n  node_embed : {tuple(node_embed.shape)}  — expected ({N}, {cfg.gat_hidden_dim})")
    print(f"  edge_state : {tuple(edge_state.shape)}  — expected ({E}, {cfg.edge_state_dim})")

    assert node_embed.shape == (N, cfg.gat_hidden_dim), \
        f"Node embed shape mismatch: {node_embed.shape}"
    assert edge_state.shape == (E, cfg.edge_state_dim), \
        f"Edge state shape mismatch: {edge_state.shape}"
    assert torch.isfinite(node_embed).all(), "Non-finite values in node embeddings!"
    assert torch.isfinite(edge_state).all(), "Non-finite values in edge states!"

    # ── Gradient flow test ────────────────────────────────────────────
    h0_grad = h0.clone().requires_grad_(True)
    node_out, edge_out = model(h0_grad, edge_index, edge_attr, edge_type)
    loss = node_out.sum() + edge_out.sum()
    loss.backward()
    assert h0_grad.grad is not None, "No gradient on h0!"
    assert h0_grad.grad.abs().sum() > 0, "Zero gradients on h0!"
    print(f"\n── Gradient flow ──")
    print(f"  ∂loss/∂h0 norm : {h0_grad.grad.norm().item():.6f}")

    # ── Verify edge embedding composition ─────────────────────────────
    src, dst = edge_index
    u_edge = torch.cat([
        node_out[src],       # (E, 512)
        node_out[dst],       # (E, 512)
        edge_out,            # (E, 64)
    ], dim=-1)               # (E, 512+512+64 = 1088)
    print(f"\n── Pairwise edge embedding (for expert heads) ──")
    print(f"  u_edge     : {tuple(u_edge.shape)}  — [h_i || h_j || e_ij]")
    expected_dim = cfg.gat_hidden_dim * 2 + cfg.edge_state_dim
    assert u_edge.shape == (E, expected_dim), \
        f"Edge embedding shape mismatch: {u_edge.shape}, expected ({E}, {expected_dim})"

    print(f"\n✓ RelationalEncoder smoke test passed.")
    print(f"  Nodes: {N}, Edges: {E}")
    print(f"  Node embed dim : {cfg.gat_hidden_dim}")
    print(f"  Edge state dim : {cfg.edge_state_dim}")
    print(f"  Pairwise dim   : {expected_dim}  (→ fed to expert MLPs)")
