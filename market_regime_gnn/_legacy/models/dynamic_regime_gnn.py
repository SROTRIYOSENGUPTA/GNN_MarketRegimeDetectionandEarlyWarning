"""
Dynamic Regime GNN — Master Model
====================================
Multi-Relational Dynamic Graph Neural Network for Market Regime Detection
and Early Warning.

Architecture (4 stages):
────────────────────────
    Stage 1 — Node Feature Encoder
        Input  :  stock.x  (N, 37)     — raw per-stock daily features
        Output :  h_node   (N, 128)    — projected node embeddings

    Stage 2 — Spatial Encoder (R-GCN with basis decomposition)
        Input  :  h_node (N, 128) + 3 relation edge_index/edge_attr
        Output :  h_spatial (N, 128)   — relation-aware node embeddings

    Stage 3 — Graph Pooling → Temporal Aggregation
        Pool   :  global_mean_pool  h_spatial → g_t (128,)  per snapshot
        Sequence: [g_1, ..., g_T]  →  LSTM  →  context (256,)

    Stage 4 — Dual Prediction Heads
        Head A :  MLP(256) → 4-class regime logits  (Bull/Crash/Liquidity/Stress)
        Head B :  MLP(256) → scalar transition logit (stress in 5–20 days)

Overall:
    T × HeteroData snapshots → (4-class logits, binary logit)

Key tensor shapes (per sample):
───────────────────────────────
    Spatial encoder input : (N, 37) per snapshot, 3 edge types
    Pooled embedding      : (1, 128) per snapshot → (T=30, 128) sequence
    LSTM output           : (256,) context vector
    Regime head output    : (4,) logits
    Transition head output: (1,) logit

Batched shapes (B samples):
────────────────────────────
    Spatial encoder  : (B*N, 37) per timestep (PyG batched)
    Pooled sequence  : (B, T, 128)
    LSTM output      : (B, 256)
    Regime logits    : (B, 4)
    Transition logit : (B,)

Weight init: Xavier-uniform + zero bias throughout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData, Batch
from torch_geometric.nn import (
    RGCNConv,
    global_mean_pool,
    global_max_pool,
)

try:
    from ..config import RegimeConfig
except ImportError as exc:
    if __package__:
        raise
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config import RegimeConfig


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1: Node Feature Encoder
# ═══════════════════════════════════════════════════════════════════════════
class NodeFeatureEncoder(nn.Module):
    """
    Projects raw per-stock features into the GNN embedding space.

    (N, 37) → Linear → ELU → Dropout → Linear → (N, 128)
    """

    def __init__(self, cfg: RegimeConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(cfg.node_input_dim, cfg.node_hidden_dim),
            nn.ELU(),
            nn.Dropout(cfg.rgcn_dropout),
            nn.Linear(cfg.node_hidden_dim, cfg.rgcn_hidden_dim),
            nn.ELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 37) raw node features.

        Returns
        -------
        (N, 128) projected node embeddings.
        """
        return self.encoder(x)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2: Spatial Encoder (R-GCN with basis decomposition)
# ═══════════════════════════════════════════════════════════════════════════
class SpatialRGCN(nn.Module):
    """
    Multi-layer R-GCN for heterogeneous graph message passing.

    Uses basis decomposition to share parameters across relation types,
    which is critical when the number of relations is small (R=3) but
    we still want relation-specific transformations.

    Architecture per layer:
        h^{l+1} = σ( Σ_r  Σ_{j∈N_r(i)}  (1/c_{i,r}) W_r^l h_j^l  +  W_0^l h_i^l )

    where W_r = Σ_b a_{rb} V_b  (basis decomposition with num_bases bases).

    Parameters
    ----------
    cfg : RegimeConfig

    Input:
        x          : (N, 128)           — node embeddings
        edge_index : (2, E_total)       — all edges concatenated
        edge_type  : (E_total,)         — relation ID per edge ∈ {0,1,2}

    Output:
        h          : (N, 128)           — updated node embeddings
    """

    def __init__(self, cfg: RegimeConfig):
        super().__init__()
        self.cfg = cfg

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for layer_idx in range(cfg.rgcn_layers):
            in_dim = cfg.rgcn_hidden_dim if layer_idx > 0 else cfg.rgcn_hidden_dim
            out_dim = cfg.rgcn_out_dim if layer_idx == cfg.rgcn_layers - 1 else cfg.rgcn_hidden_dim

            conv = RGCNConv(
                in_channels=in_dim,
                out_channels=out_dim,
                num_relations=cfg.num_relations,      # 3
                num_bases=cfg.rgcn_num_bases,          # 3 (basis decomposition)
                aggr="mean",                           # mean aggregation
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(cfg.rgcn_dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x          : (N, 128)      — node embeddings from NodeFeatureEncoder
        edge_index : (2, E_total)  — concatenated edges from all 3 relations
        edge_type  : (E_total,)    — relation ID per edge

        Returns
        -------
        h : (N, 128) updated node embeddings.

        Intermediate shapes per layer:
            h : (N, 128) → (N, 128) → (N, 128)
        """
        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_type)      # (N, out_dim)
            h_new = norm(h_new)                          # (N, out_dim)
            if i < len(self.convs) - 1:
                h_new = F.elu(h_new)                     # activation
                h_new = self.dropout(h_new)              # dropout
                h_new = h_new + h                        # residual (same dim)
            else:
                h_new = F.elu(h_new)                     # final layer
                if h.shape[-1] == h_new.shape[-1]:
                    h_new = h_new + h                    # residual if dims match
            h = h_new
        return h


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3: Temporal Aggregator (LSTM or Transformer)
# ═══════════════════════════════════════════════════════════════════════════
class TemporalLSTM(nn.Module):
    """
    Processes a sequence of T=30 graph-level embeddings to capture
    temporal evolution of market structure.

    (B, T, D_graph) → LSTM → (B, D_hidden) from final hidden state.

    Parameters
    ----------
    cfg : RegimeConfig
    """

    def __init__(self, cfg: RegimeConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=cfg.graph_embed_dim,          # 128
            hidden_size=cfg.lstm_hidden_dim,          # 256
            num_layers=cfg.lstm_layers,               # 2
            batch_first=True,                         # (B, T, D)
            dropout=cfg.lstm_dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=cfg.lstm_bidirectional,
        )
        self.layer_norm = nn.LayerNorm(cfg.lstm_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 128) — sequence of graph-level embeddings.

        Returns
        -------
        (B, 256) — LSTM final hidden state (last layer, last timestep).
        """
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, B, hidden_dim)
        # Take last layer's hidden state
        h_last = h_n[-1]                              # (B, 256)
        h_last = self.layer_norm(h_last)              # (B, 256)
        return h_last


class TemporalTransformer(nn.Module):
    """
    Alternative temporal aggregator using a Transformer encoder.

    (B, T, D_graph) → Positional Encoding → Transformer → (B, D_graph)
    from the last timestep's output.

    Parameters
    ----------
    cfg : RegimeConfig
    """

    def __init__(self, cfg: RegimeConfig):
        super().__init__()
        self.d_model = cfg.graph_embed_dim            # 128

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.seq_len, cfg.graph_embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.graph_embed_dim,
            nhead=cfg.temporal_n_heads,
            dim_feedforward=cfg.temporal_ff_dim,
            dropout=cfg.temporal_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,                          # pre-norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.temporal_layers,
        )
        self.layer_norm = nn.LayerNorm(cfg.graph_embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, 128) — sequence of graph-level embeddings.

        Returns
        -------
        (B, 128) — last timestep's transformer output.
        """
        # Causal mask: prevent attending to future timesteps
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        x = x + self.pos_embed[:, :T, :]             # (B, T, 128)
        h = self.transformer(x, mask=mask)            # (B, T, 128)
        h_last = h[:, -1, :]                          # (B, 128)
        return self.layer_norm(h_last)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4: Dual Prediction Heads
# ═══════════════════════════════════════════════════════════════════════════
class RegimeClassifierHead(nn.Module):
    """
    4-class regime classifier: Bull / Crash / Liquidity / Stress.

    (B, D_temporal) → MLP → (B, 4) logits.
    """

    def __init__(self, input_dim: int, cfg: RegimeConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, cfg.regime_head_hidden),
            nn.ELU(),
            nn.Dropout(cfg.regime_head_dropout),
            nn.Linear(cfg.regime_head_hidden, cfg.regime_head_hidden),
            nn.ELU(),
            nn.Dropout(cfg.regime_head_dropout),
            nn.Linear(cfg.regime_head_hidden, cfg.num_regime_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, D_temporal) — temporal context vector.

        Returns
        -------
        (B, 4) raw logits (NOT softmax).
        """
        return self.head(x)


class TransitionLogitHead(nn.Module):
    """
    Binary early-warning head: P(stress onset in 5–20 days).

    (B, D_temporal) → MLP → (B,) logit (NOT sigmoid).
    """

    def __init__(self, input_dim: int, cfg: RegimeConfig):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, cfg.transition_head_hidden),
            nn.ELU(),
            nn.Dropout(cfg.transition_head_dropout),
            nn.Linear(cfg.transition_head_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, D_temporal) — temporal context vector.

        Returns
        -------
        (B,) raw logit (NOT sigmoid — apply BCE with logits loss).
        """
        return self.head(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# Master Model: DynamicRegimeGNN
# ═══════════════════════════════════════════════════════════════════════════
class DynamicRegimeGNN(nn.Module):
    """
    Multi-Relational Dynamic Graph Neural Network for Market Regime
    Detection and Early Warning.

    Complete forward pass:
        1. For each timestep t in [1, ..., T]:
           a) Batch all B graphs at timestep t into a PyG mega-graph
           b) NodeFeatureEncoder: (B*N, 37) → (B*N, 128)
           c) SpatialRGCN: message passing with 3 relation types → (B*N, 128)
           d) global_mean_pool → (B, 128) graph-level embedding g_t
        2. Stack [g_1, ..., g_T] → (B, T, 128)
        3. TemporalLSTM: (B, T, 128) → (B, 256) context vector
        4. RegimeClassifierHead: (B, 256) → (B, 4) regime logits
           TransitionLogitHead: (B, 256) → (B,) transition logit

    Parameters
    ----------
    cfg : RegimeConfig
    """

    # Canonical HeteroData edge type tuples
    EDGE_TYPES = [
        ("stock", "correlation", "stock"),
        ("stock", "etf_cohold", "stock"),
        ("stock", "supply_chain", "stock"),
    ]
    # Map each edge type name to its integer relation ID
    REL_ID = {
        "correlation": 0,
        "etf_cohold": 1,
        "supply_chain": 2,
    }

    def __init__(self, cfg: Optional[RegimeConfig] = None):
        super().__init__()
        cfg = RegimeConfig() if cfg is None else cfg
        self.cfg = cfg

        # Stage 1: Node Feature Encoder
        self.node_encoder = NodeFeatureEncoder(cfg)

        # Stage 2: Spatial R-GCN
        self.spatial_encoder = SpatialRGCN(cfg)

        # Stage 3: Temporal Aggregator
        if cfg.temporal_type == "lstm":
            self.temporal_encoder = TemporalLSTM(cfg)
            temporal_out = cfg.lstm_hidden_dim         # 256
        elif cfg.temporal_type == "transformer":
            self.temporal_encoder = TemporalTransformer(cfg)
            temporal_out = cfg.graph_embed_dim          # 128
        else:
            raise ValueError(f"Unknown temporal_type: {cfg.temporal_type}")

        # Stage 4: Dual Prediction Heads
        self.regime_head = RegimeClassifierHead(temporal_out, cfg)
        self.transition_head = TransitionLogitHead(temporal_out, cfg)

    def _prepare_homogeneous_edges(
        self,
        batched_graph: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the heterogeneous edge types from a batched HeteroData
        into a single (edge_index, edge_type) pair for R-GCN.

        Parameters
        ----------
        batched_graph : PyG Batch of HeteroData graphs.

        Returns
        -------
        edge_index : (2, E_total) — concatenated edges across all relations.
        edge_type  : (E_total,)   — integer relation ID per edge.
        """
        all_edge_index = []
        all_edge_type = []

        for et_tuple in self.EDGE_TYPES:
            rel_name = et_tuple[1]
            rel_id = self.REL_ID[rel_name]

            if et_tuple in batched_graph.edge_types:
                ei = batched_graph[et_tuple].edge_index   # (2, E_r)
                E_r = ei.shape[1]
                all_edge_index.append(ei)
                all_edge_type.append(
                    torch.full((E_r,), rel_id, dtype=torch.long, device=ei.device)
                )

        if len(all_edge_index) == 0:
            device = batched_graph["stock"].x.device
            return (
                torch.zeros(2, 0, dtype=torch.long, device=device),
                torch.zeros(0, dtype=torch.long, device=device),
            )

        edge_index = torch.cat(all_edge_index, dim=1)    # (2, E_total)
        edge_type = torch.cat(all_edge_type, dim=0)       # (E_total,)
        return edge_index, edge_type

    def forward_snapshot(
        self,
        batched_graph: Batch,
    ) -> torch.Tensor:
        """
        Process a single batched timestep through the spatial encoder.

        Parameters
        ----------
        batched_graph : PyG Batch of B HeteroData graphs (one timestep).

        Returns
        -------
        graph_embed : (B, 128) — pooled graph-level embedding for this timestep.
        """
        # Extract node features
        x = batched_graph["stock"].x                     # (B*N, 37)
        batch_vec = batched_graph["stock"].batch          # (B*N,) — graph membership

        # Stage 1: Node feature encoding
        h = self.node_encoder(x)                         # (B*N, 128)

        # Stage 2: Spatial R-GCN
        edge_index, edge_type = self._prepare_homogeneous_edges(batched_graph)
        h = self.spatial_encoder(h, edge_index, edge_type)  # (B*N, 128)

        # Graph pooling → one embedding per graph
        if self.cfg.pool_method == "mean":
            g = global_mean_pool(h, batch_vec)           # (B, 128)
        elif self.cfg.pool_method == "max":
            g = global_max_pool(h, batch_vec)            # (B, 128)
        else:
            g = global_mean_pool(h, batch_vec)           # default: mean

        return g

    def forward(
        self,
        snapshots: List[List[HeteroData]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass over a batch of temporal graph sequences.

        Parameters
        ----------
        snapshots : list[list[HeteroData]] — shape (T, B).
                    snapshots[t] = list of B HeteroData graphs at timestep t.
                    Produced by regime_collate_fn.

        Returns
        -------
        regime_logits     : (B, 4)  — raw logits for 4 regime classes.
        transition_logit  : (B,)    — raw logit for stress transition.
        """
        T = len(snapshots)
        B = len(snapshots[0])

        # Stage 1+2: Process each timestep through spatial encoder
        graph_embeds = []                                 # will be T × (B, 128)
        for t in range(T):
            # Batch B graphs at timestep t into a single PyG mega-graph
            batched_t = Batch.from_data_list(snapshots[t])
            g_t = self.forward_snapshot(batched_t)        # (B, 128)
            graph_embeds.append(g_t)

        # Stack into temporal sequence: (B, T, 128)
        graph_seq = torch.stack(graph_embeds, dim=1)      # (B, T, 128)

        # Stage 3: Temporal aggregation
        context = self.temporal_encoder(graph_seq)         # (B, D_temporal)

        # Stage 4: Dual prediction heads
        regime_logits = self.regime_head(context)          # (B, 4)
        transition_logit = self.transition_head(context)   # (B,)

        return regime_logits, transition_logit

    def parameter_groups(self) -> list:
        """
        Build parameter groups: bias/LayerNorm excluded from weight decay.
        """
        decay_params = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": self.cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]


# ═══════════════════════════════════════════════════════════════════════════
# Comprehensive Smoke Test
# ═══════════════════════════════════════════════════════════════════════════
def test_forward_pass():
    """
    End-to-end smoke test with synthetic HeteroData sequences.

    Verifies:
        1. All tensor dimensions through each stage
        2. Forward pass produces finite outputs
        3. Gradient flow to all parameters
        4. Both prediction heads produce correct shapes
        5. Both LSTM and Transformer temporal encoders
    """
    import time

    print("=" * 70)
    print("  DynamicRegimeGNN — Comprehensive Smoke Test")
    print("=" * 70)

    torch.manual_seed(42)

    # ── Config ───────────────────────────────────────────────────────
    cfg = RegimeConfig()
    cfg.seq_len = 10            # shorter for fast test
    cfg.rgcn_layers = 2
    cfg.lstm_layers = 2

    N = 20                      # nodes per graph
    B = 3                       # batch size
    T = cfg.seq_len             # timesteps

    # ── Generate synthetic HeteroData snapshots ──────────────────────
    def make_hetero_data(n_nodes: int) -> HeteroData:
        """Create a single HeteroData snapshot."""
        data = HeteroData()
        data["stock"].x = torch.randn(n_nodes, cfg.node_input_dim)
        data["stock"].num_nodes = n_nodes

        # Correlation edges (dense-ish)
        e_corr = 60
        data["stock", "correlation", "stock"].edge_index = torch.stack([
            torch.randint(0, n_nodes, (e_corr,)),
            torch.randint(0, n_nodes, (e_corr,)),
        ])
        data["stock", "correlation", "stock"].edge_attr = torch.randn(e_corr, cfg.edge_attr_dim)

        # ETF co-holding edges (medium)
        e_etf = 30
        data["stock", "etf_cohold", "stock"].edge_index = torch.stack([
            torch.randint(0, n_nodes, (e_etf,)),
            torch.randint(0, n_nodes, (e_etf,)),
        ])
        data["stock", "etf_cohold", "stock"].edge_attr = torch.randn(e_etf, cfg.edge_attr_dim)

        # Supply chain edges (sparse)
        e_sc = 15
        data["stock", "supply_chain", "stock"].edge_index = torch.stack([
            torch.randint(0, n_nodes, (e_sc,)),
            torch.randint(0, n_nodes, (e_sc,)),
        ])
        data["stock", "supply_chain", "stock"].edge_attr = torch.randn(e_sc, cfg.edge_attr_dim)

        return data

    # Build (T, B) structure of HeteroData
    snapshots = []
    for t in range(T):
        timestep_graphs = [make_hetero_data(N) for _ in range(B)]
        snapshots.append(timestep_graphs)

    print(f"\nTest configuration:")
    print(f"  Nodes (N)      : {N}")
    print(f"  Batch (B)      : {B}")
    print(f"  Timesteps (T)  : {T}")
    print(f"  Relations      : {cfg.num_relations}")
    print(f"  Features       : {cfg.node_input_dim}")

    # ═════════════════════════════════════════════════════════════════
    # Test 1: LSTM temporal encoder
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  Test 1: DynamicRegimeGNN with LSTM temporal encoder")
    print(f"{'─' * 70}")

    cfg.temporal_type = "lstm"
    model = DynamicRegimeGNN(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # Forward pass
    t0 = time.time()
    regime_logits, transition_logit = model(snapshots)
    fwd_time = time.time() - t0

    print(f"\n  Forward pass ({fwd_time:.3f}s):")
    print(f"    regime_logits    : {tuple(regime_logits.shape)}  — expected ({B}, {cfg.num_regime_classes})")
    print(f"    transition_logit : {tuple(transition_logit.shape)}  — expected ({B},)")

    assert regime_logits.shape == (B, cfg.num_regime_classes), \
        f"Regime logits shape: {regime_logits.shape}"
    assert transition_logit.shape == (B,), \
        f"Transition logit shape: {transition_logit.shape}"
    assert torch.isfinite(regime_logits).all(), "Non-finite regime logits!"
    assert torch.isfinite(transition_logit).all(), "Non-finite transition logit!"
    print(f"    ✓ All outputs finite")

    # Gradient flow
    loss = regime_logits.sum() + transition_logit.sum()
    loss.backward()

    n_graded = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            n_graded += 1
        else:
            print(f"    ⚠ No gradient on: {name}")

    print(f"\n  Gradient flow:")
    print(f"    Parameters with gradients: {n_graded}/{n_total}")
    assert n_graded == n_total, f"Missing gradients on {n_total - n_graded} params!"
    print(f"    ✓ All {n_params:,} parameters receive gradients")

    # Parameter groups
    groups = model.parameter_groups()
    n_decay = sum(p.numel() for p in groups[0]["params"])
    n_nodecay = sum(p.numel() for p in groups[1]["params"])
    print(f"\n  Parameter groups:")
    print(f"    With weight decay    : {n_decay:>10,}")
    print(f"    Without weight decay : {n_nodecay:>10,} (bias + norms)")
    print(f"    Total                : {n_decay + n_nodecay:>10,}")

    # Output value ranges
    print(f"\n  Output statistics:")
    print(f"    Regime logits range  : [{regime_logits.min():.4f}, {regime_logits.max():.4f}]")
    print(f"    Transition logit range: [{transition_logit.min():.4f}, {transition_logit.max():.4f}]")

    # Softmax probabilities
    probs = F.softmax(regime_logits, dim=-1)
    print(f"    Regime probs (sample 0): {probs[0].detach().numpy().round(3)}")

    # ═════════════════════════════════════════════════════════════════
    # Test 2: Transformer temporal encoder
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  Test 2: DynamicRegimeGNN with Transformer temporal encoder")
    print(f"{'─' * 70}")

    cfg.temporal_type = "transformer"
    model_xfmr = DynamicRegimeGNN(cfg)
    n_params_xfmr = sum(p.numel() for p in model_xfmr.parameters())
    print(f"  Total parameters: {n_params_xfmr:,}")

    regime_logits_x, transition_logit_x = model_xfmr(snapshots)

    print(f"  regime_logits    : {tuple(regime_logits_x.shape)}")
    print(f"  transition_logit : {tuple(transition_logit_x.shape)}")

    assert regime_logits_x.shape == (B, cfg.num_regime_classes)
    assert transition_logit_x.shape == (B,)
    assert torch.isfinite(regime_logits_x).all()
    assert torch.isfinite(transition_logit_x).all()
    print(f"  ✓ Transformer variant passes")

    # Gradient check
    loss_x = regime_logits_x.sum() + transition_logit_x.sum()
    loss_x.backward()
    for name, p in model_xfmr.named_parameters():
        assert p.grad is not None, f"No grad on {name}"
    print(f"  ✓ All parameters receive gradients")

    # ═════════════════════════════════════════════════════════════════
    # Test 3: Single snapshot processing
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  Test 3: Single snapshot spatial encoding")
    print(f"{'─' * 70}")

    cfg.temporal_type = "lstm"
    model_single = DynamicRegimeGNN(cfg)
    batched_single = Batch.from_data_list([make_hetero_data(N) for _ in range(B)])
    g = model_single.forward_snapshot(batched_single)
    print(f"  Graph embed: {tuple(g.shape)} — expected ({B}, {cfg.graph_embed_dim})")
    assert g.shape == (B, cfg.graph_embed_dim)
    assert torch.isfinite(g).all()
    print(f"  ✓ Single snapshot encoding passes")

    # ═════════════════════════════════════════════════════════════════
    # Test 4: Variable node counts per graph
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  Test 4: Variable node counts per graph")
    print(f"{'─' * 70}")

    snapshots_var = []
    for t in range(T):
        timestep_graphs = []
        for b in range(B):
            n = 10 + b * 5  # 10, 15, 20 nodes
            timestep_graphs.append(make_hetero_data(n))
        snapshots_var.append(timestep_graphs)

    regime_v, trans_v = model(snapshots_var)
    print(f"  Node counts per graph: [10, 15, 20]")
    print(f"  regime_logits    : {tuple(regime_v.shape)}")
    print(f"  transition_logit : {tuple(trans_v.shape)}")
    assert regime_v.shape == (B, cfg.num_regime_classes)
    assert trans_v.shape == (B,)
    print(f"  ✓ Variable node counts handled correctly")

    # ═════════════════════════════════════════════════════════════════
    # Test 5: Edge count summary
    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print(f"  Test 5: Edge type statistics")
    print(f"{'─' * 70}")

    sample_batch = Batch.from_data_list(snapshots[0])
    edge_index, edge_type = model._prepare_homogeneous_edges(sample_batch)
    total_edges = edge_index.shape[1]
    for rel_name, rel_id in model.REL_ID.items():
        count = (edge_type == rel_id).sum().item()
        print(f"  {rel_name:15s} : {count:4d} edges ({100*count/max(total_edges,1):.1f}%)")
    print(f"  {'total':15s} : {total_edges:4d} edges")

    # ═════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print(f"  ALL SMOKE TESTS PASSED ✓")
    print(f"{'=' * 70}")
    print(f"  LSTM model    : {n_params:,} parameters")
    print(f"  Transformer   : {n_params_xfmr:,} parameters")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    test_forward_pass()
