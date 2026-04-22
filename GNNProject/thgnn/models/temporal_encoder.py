"""
Temporal Encoder  (Transformer)
================================
Implements the per-stock Transformer encoder described in Section 4.2.2:

  Input per stock i at date t:
      X_{i,t} ∈ R^{L × F}  =  (30, 37)

  Pipeline:
      1.  Linear projection  F → d_model = 128           → (30, 128)
      2.  Add sinusoidal positional encoding               → (30, 128)
      3.  4 × Pre-Norm Transformer encoder layers
              • 8 attention heads  (d_k = 16)
              • FFN inner dim = 512   (standard 4×d_model)
              • dropout = 0.2
          Output H_{i,t}                                   → (30, 128)
      4.  Flatten H to vec(H) ∈ R^{3840}                  → (3840,)
      5.  LayerNorm → MLP → 512-d node embedding h_i^(0)  → (512,)

  Weight init: Xavier-uniform weights, zero biases (Appendix A.1).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import THGNNConfig


# ═══════════════════════════════════════════════════════════════════════════
# Sinusoidal Positional Encoding  (Vaswani et al., 2017)
# ═══════════════════════════════════════════════════════════════════════════
class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding added *after* the linear projection.

    pe(pos, 2i)   = sin(pos / 10000^{2i/d_model})
    pe(pos, 2i+1) = cos(pos / 10000^{2i/d_model})

    Registered as a buffer (not a parameter) so it moves to GPU automatically
    but is not updated by the optimiser.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                          # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()    # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                            # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                         # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, d_model)

        Returns
        -------
        (batch, seq_len, d_model)  with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ═══════════════════════════════════════════════════════════════════════════
# Pre-Norm Transformer Encoder Layer
# ═══════════════════════════════════════════════════════════════════════════
class PreNormTransformerEncoderLayer(nn.Module):
    """
    A single Pre-Norm Transformer encoder layer.

    Pre-norm applies LayerNorm *before* attention and *before* FFN,
    as specified in the paper (Section 4.2.2 / Appendix A.1):

        x ← x + MHA(LN(x))
        x ← x + FFN(LN(x))

    This improves gradient flow through deep stacks and stabilises
    training on noisy financial data.
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ── Layer norms (applied *before* sub-layers) ─────────────────
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ── Multi-Head Self-Attention ─────────────────────────────────
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,        # input shape: (B, L, d_model)
        )

        # ── Position-wise Feed-Forward Network ────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),               # standard modern choice
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        src : (B, L, d_model)

        Returns
        -------
        (B, L, d_model)
        """
        # ── Pre-norm self-attention ───────────────────────────────────
        x_norm = self.norm1(src)
        attn_out, _ = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.dropout(attn_out)          # residual connection

        # ── Pre-norm FFN ──────────────────────────────────────────────
        src = src + self.ffn(self.norm2(src))        # residual connection

        return src


# ═══════════════════════════════════════════════════════════════════════════
# Full Temporal Encoder
# ═══════════════════════════════════════════════════════════════════════════
class TemporalEncoder(nn.Module):
    """
    Complete per-stock Temporal Encoder.

    Input  → (N, L=30, F=37)         N stocks, 30-day sequences, 37 features
    Output → (N, 512)                512-d node embeddings for the GAT

    Architecture:
        Linear(37 → 128) → SinusoidalPE → 4× PreNormTransformerLayer →
        Flatten(30×128=3840) → LayerNorm → Linear(3840→512) → GELU →
        Dropout → Linear(512→512)
    """

    def __init__(self, cfg: THGNNConfig = THGNNConfig()):
        super().__init__()
        self.cfg = cfg

        # ── 1. Input projection: F=37 → d_model=128 ──────────────────
        self.input_proj = nn.Linear(cfg.num_features, cfg.d_model)   # (37 → 128)

        # ── 2. Sinusoidal positional encoding ─────────────────────────
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=cfg.d_model,
            max_len=cfg.seq_len + 10,   # small buffer
            dropout=cfg.transformer_dropout,
        )

        # ── 3. Transformer encoder stack (4 pre-norm layers) ─────────
        self.encoder_layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.transformer_dropout,
            )
            for _ in range(cfg.n_encoder_layers)
        ])

        # Final LayerNorm after the last encoder layer (standard pre-norm practice)
        self.final_norm = nn.LayerNorm(cfg.d_model)

        # ── 4–5. Flatten + LayerNorm-MLP → 512-d embedding ───────────
        flatten_dim = cfg.seq_len * cfg.d_model          # 30 × 128 = 3840
        self.embedding_head = nn.Sequential(
            nn.LayerNorm(flatten_dim),
            nn.Linear(flatten_dim, cfg.node_embed_dim),  # 3840 → 512
            nn.GELU(),
            nn.Dropout(cfg.transformer_dropout),
            nn.Linear(cfg.node_embed_dim, cfg.node_embed_dim),  # 512 → 512
        )

        # ── Weight initialisation (Xavier-uniform + zero bias) ────────
        self._init_weights()

    # ── weight init ───────────────────────────────────────────────────
    def _init_weights(self):
        """Xavier-uniform for all Linear weights; zero for all biases."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    # ── forward ───────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, L=30, F=37)
            Per-stock feature sequences (already z-score normalised).
        padding_mask : (N, L) bool, optional
            True for padded / missing time-steps.

        Returns
        -------
        h0 : (N, 512)
            Node embeddings to be fed into the GAT relational encoder.

        Intermediate shapes:
            after input_proj : (N, 30, 128)
            after pos_enc    : (N, 30, 128)
            after encoder    : (N, 30, 128)
            after flatten    : (N, 3840)
            after MLP        : (N, 512)
        """
        N, L, F = x.shape
        assert L == self.cfg.seq_len, f"Expected seq_len={self.cfg.seq_len}, got {L}"
        assert F == self.cfg.num_features, f"Expected F={self.cfg.num_features}, got {F}"

        # 1. Linear projection
        h = self.input_proj(x)                 # (N, 30, 128)

        # 2. Add sinusoidal positional encoding
        h = self.pos_enc(h)                    # (N, 30, 128)

        # 3. Pass through 4 pre-norm Transformer encoder layers
        for layer in self.encoder_layers:
            h = layer(h, src_key_padding_mask=padding_mask)   # (N, 30, 128)

        # Apply final LayerNorm
        h = self.final_norm(h)                 # (N, 30, 128)

        # 4. Flatten: (N, 30, 128) → (N, 3840)
        h = h.reshape(N, -1)                   # (N, 3840)

        # 5. LayerNorm → MLP → 512-d node embedding
        h0 = self.embedding_head(h)            # (N, 512)

        return h0


# ═══════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    cfg = THGNNConfig()
    model = TemporalEncoder(cfg)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TemporalEncoder parameters: {n_params:,}")

    # Synthetic forward pass: 50 stocks, 30 days, 37 features
    x = torch.randn(50, cfg.seq_len, cfg.num_features)
    h0 = model(x)
    print(f"Input  shape : {tuple(x.shape)}")         # (50, 30, 37)
    print(f"Output shape : {tuple(h0.shape)}")         # (50, 512)
    assert h0.shape == (50, cfg.node_embed_dim)
    print("✓ Temporal Encoder smoke test passed.")
