"""
End-to-end integration test: Dataset → TemporalEncoder → RelationalEncoder.
Validates the full forward pass from raw features to final node + edge embeddings.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import datetime
import numpy as np
import torch
from config import THGNNConfig
from data.dataset import THGNNDataset, build_dataloader
from models.temporal_encoder import TemporalEncoder
from models.relational_encoder import RelationalEncoder


def generate_synthetic_data(n_stocks=20, n_days=200, n_features=37, seed=42):
    rng = np.random.default_rng(seed)
    base = datetime.date(2018, 1, 1)
    dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]
    features, returns, sector_map, subind_map = {}, {}, {}, {}
    for sid in range(n_stocks):
        features[sid] = rng.standard_normal((n_days, n_features)).astype(np.float32)
        returns[sid] = rng.standard_normal(n_days).astype(np.float32) * 0.02
        sector_map[sid] = int(rng.integers(0, 11))
        subind_map[sid] = int(rng.integers(0, 50))
    return features, dates, sector_map, subind_map, returns


def run_e2e_step2():
    print("=" * 64)
    print("  THGNN End-to-End Test — Step 2")
    print("  Dataset → TemporalEncoder → RelationalEncoder")
    print("=" * 64 + "\n")

    cfg = THGNNConfig()
    cfg.top_k_corr = 5
    cfg.bot_k_corr = 5
    cfg.rand_mid_k = 5

    features, dates, sector_map, subind_map, returns = generate_synthetic_data()

    ds = THGNNDataset(features=features, dates=dates, sector_map=sector_map,
                      subind_map=subind_map, returns=returns, cfg=cfg)
    loader = build_dataloader(ds, cfg, shuffle=False)

    temporal_enc = TemporalEncoder(cfg)
    relational_enc = RelationalEncoder(cfg)
    temporal_enc.eval()
    relational_enc.eval()

    batch = next(iter(loader))
    N = batch.x.shape[0]
    E = batch.edge_index.shape[1]

    print(f"── Batch from DataLoader ──")
    print(f"  x             : {tuple(batch.x.shape)}")
    print(f"  edge_index    : {tuple(batch.edge_index.shape)}")
    print(f"  edge_attr     : {tuple(batch.edge_attr.shape)}")
    print(f"  edge_type     : {tuple(batch.edge_type.shape)}")
    print(f"  baseline_z    : {tuple(batch.baseline_z.shape)}")
    print(f"  target_z_resid: {tuple(batch.target_z_resid.shape)}")
    print(f"  batch.batch   : {tuple(batch.batch.shape)}")

    with torch.no_grad():
        # Stage 1: Temporal Encoder
        h0 = temporal_enc(batch.x)
        print(f"\n── TemporalEncoder ──")
        print(f"  Input  : {tuple(batch.x.shape)}  → Output : {tuple(h0.shape)}")
        assert h0.shape == (N, cfg.node_embed_dim)

        # Stage 2: Relational Encoder
        node_embed, edge_state = relational_enc(
            h0, batch.edge_index, batch.edge_attr, batch.edge_type
        )
        print(f"\n── RelationalEncoder ──")
        print(f"  node_embed : {tuple(node_embed.shape)}  — expected ({N}, {cfg.gat_hidden_dim})")
        print(f"  edge_state : {tuple(edge_state.shape)}  — expected ({E}, {cfg.edge_state_dim})")
        assert node_embed.shape == (N, cfg.gat_hidden_dim)
        assert edge_state.shape == (E, cfg.edge_state_dim)

        # Stage 3: Compose pairwise edge embeddings (preview for expert heads)
        src, dst = batch.edge_index
        u_edge = torch.cat([node_embed[src], node_embed[dst], edge_state], dim=-1)
        expected_dim = cfg.gat_hidden_dim * 2 + cfg.edge_state_dim
        print(f"\n── Pairwise Edge Embedding ──")
        print(f"  u_edge     : {tuple(u_edge.shape)}  — [h_i || h_j || e_ij]")
        print(f"  dim        : {expected_dim}  (512 + 512 + 64 = 1088)")
        assert u_edge.shape == (E, expected_dim)

    # ── Gradient flow through full pipeline ───────────────────────────
    temporal_enc.train()
    relational_enc.train()
    x_grad = batch.x.clone().requires_grad_(True)
    h0 = temporal_enc(x_grad)
    node_embed, edge_state = relational_enc(
        h0, batch.edge_index, batch.edge_attr, batch.edge_type
    )
    loss = node_embed.sum() + edge_state.sum()
    loss.backward()
    assert x_grad.grad is not None
    assert x_grad.grad.abs().sum() > 0
    print(f"\n── Gradient flow (full pipeline) ──")
    print(f"  ∂loss/∂x norm : {x_grad.grad.norm().item():.6f}")
    print(f"  ✓ Gradients flow from RelationalEncoder back through TemporalEncoder")

    # ── Parameter counts ──────────────────────────────────────────────
    te_params = sum(p.numel() for p in temporal_enc.parameters())
    re_params = sum(p.numel() for p in relational_enc.parameters())
    print(f"\n── Model sizes ──")
    print(f"  TemporalEncoder   : {te_params:>10,} params")
    print(f"  RelationalEncoder : {re_params:>10,} params")
    print(f"  Total (so far)    : {te_params + re_params:>10,} params")

    assert torch.isfinite(node_embed).all()
    assert torch.isfinite(edge_state).all()

    print(f"\n{'=' * 64}")
    print(f"  ALL E2E TESTS PASSED ✓")
    print(f"{'=' * 64}")


def test_e2e_step2():
    run_e2e_step2()


if __name__ == "__main__":
    run_e2e_step2()
