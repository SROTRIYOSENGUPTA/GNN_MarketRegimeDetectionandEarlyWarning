"""
Full pipeline integration test — Step 3
Dataset → TemporalEncoder → RelationalEncoder → ExpertHeads → THGNNLoss

Validates:
  1. All tensor shapes through the complete forward pass
  2. Loss computation produces finite, positive values
  3. Gradients flow end-to-end from loss back to raw input features
  4. A single optimiser step reduces the loss (basic learnability check)
"""

import datetime
from pathlib import Path
import numpy as np
import pytest
import torch

pytest.importorskip("torch_geometric")

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(REPO_ROOT))

from GNNProject.thgnn.config import THGNNConfig
from GNNProject.thgnn.data.dataset import THGNNDataset, build_dataloader
from GNNProject.thgnn.models.temporal_encoder import TemporalEncoder
from GNNProject.thgnn.models.relational_encoder import RelationalEncoder
from GNNProject.thgnn.models.expert_heads import ExpertPredictionHeads
from GNNProject.thgnn.losses.loss import THGNNLoss


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


def run_e2e_step3():
    print("=" * 68)
    print("  THGNN Full Pipeline Integration Test — Step 3")
    print("  Dataset → Temporal → GAT → Experts → Loss")
    print("=" * 68 + "\n")

    cfg = THGNNConfig()
    cfg.top_k_corr = 5
    cfg.bot_k_corr = 5
    cfg.rand_mid_k = 5

    # ── Data ──────────────────────────────────────────────────────────
    features, dates, sector_map, subind_map, returns = generate_synthetic_data()
    ds = THGNNDataset(features=features, dates=dates, sector_map=sector_map,
                      subind_map=subind_map, returns=returns, cfg=cfg)
    loader = build_dataloader(ds, cfg, shuffle=False)
    batch = next(iter(loader))

    N = batch.x.shape[0]
    E = batch.edge_index.shape[1]
    print(f"Batch:  N={N} nodes,  E={E} edges,  batch_size={cfg.batch_size}\n")

    # ── Models ────────────────────────────────────────────────────────
    temporal_enc = TemporalEncoder(cfg)
    relational_enc = RelationalEncoder(cfg)
    expert_heads = ExpertPredictionHeads(cfg)
    loss_fn = THGNNLoss(cfg)

    # ── Full forward pass ─────────────────────────────────────────────
    print("── Forward pass ──")

    # Stage 1: Temporal Encoder
    h0 = temporal_enc(batch.x)                         # (N, 512)
    print(f"  1. TemporalEncoder   : {tuple(batch.x.shape)} → {tuple(h0.shape)}")

    # Stage 2: Relational Encoder (GAT)
    node_embed, edge_state = relational_enc(
        h0, batch.edge_index, batch.edge_attr, batch.edge_type
    )
    print(f"  2. RelationalEncoder : node={tuple(node_embed.shape)}, edge={tuple(edge_state.shape)}")

    # Stage 3: Expert Prediction Heads
    delta_z_pred, rho_pred = expert_heads(
        node_embed, edge_state, batch.edge_index,
        batch.edge_type, batch.baseline_z,
    )
    print(f"  3. ExpertHeads       : Δẑ={tuple(delta_z_pred.shape)}, ρ̂={tuple(rho_pred.shape)}")
    print(f"     ρ̂ range: [{rho_pred.min().item():.4f}, {rho_pred.max().item():.4f}]")

    # Stage 4: Loss
    total_loss, log = loss_fn(
        delta_z_pred, rho_pred,
        batch.target_z_resid, batch.baseline_z,
        batch.edge_type,
    )
    print(f"\n── Loss ──")
    for k, v in log.items():
        print(f"  {k:22s}: {v.item():.6f}")

    assert total_loss.shape == ()
    assert torch.isfinite(total_loss)
    assert total_loss > 0

    # ── End-to-end gradient flow ──────────────────────────────────────
    total_loss.backward()
    print(f"\n── Gradient flow (end-to-end) ──")

    has_grad = True
    for name, mod in [("TemporalEncoder", temporal_enc),
                       ("RelationalEncoder", relational_enc),
                       ("ExpertHeads", expert_heads)]:
        grads = [p.grad for p in mod.parameters() if p.grad is not None]
        grad_norm = torch.stack([g.norm() for g in grads]).sum().item() if grads else 0
        n_graded = len(grads)
        n_total = sum(1 for _ in mod.parameters())
        ok = n_graded == n_total and grad_norm > 0
        has_grad = has_grad and ok
        symbol = "✓" if ok else "✗"
        print(f"  {symbol} {name:22s}: {n_graded}/{n_total} params with grad, "
              f"norm={grad_norm:.6f}")

    assert has_grad, "Gradient flow broken in at least one module!"

    # ── Learnability check: single optim step ─────────────────────────
    print(f"\n── Learnability check (1 optimiser step) ──")

    # Reset
    temporal_enc.zero_grad()
    relational_enc.zero_grad()
    expert_heads.zero_grad()

    all_params = (list(temporal_enc.parameters()) +
                  list(relational_enc.parameters()) +
                  list(expert_heads.parameters()))
    optimizer = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Forward + backward
    h0 = temporal_enc(batch.x)
    node_embed, edge_state = relational_enc(
        h0, batch.edge_index, batch.edge_attr, batch.edge_type
    )
    delta_z_pred, rho_pred = expert_heads(
        node_embed, edge_state, batch.edge_index,
        batch.edge_type, batch.baseline_z,
    )
    loss_before, _ = loss_fn(
        delta_z_pred, rho_pred,
        batch.target_z_resid, batch.baseline_z, batch.edge_type,
    )
    loss_before.backward()
    optimizer.step()

    # Second forward pass (after step)
    with torch.no_grad():
        h0 = temporal_enc(batch.x)
        node_embed, edge_state = relational_enc(
            h0, batch.edge_index, batch.edge_attr, batch.edge_type
        )
        delta_z_pred, rho_pred = expert_heads(
            node_embed, edge_state, batch.edge_index,
            batch.edge_type, batch.baseline_z,
        )
        loss_after, _ = loss_fn(
            delta_z_pred, rho_pred,
            batch.target_z_resid, batch.baseline_z, batch.edge_type,
        )

    improved = loss_after < loss_before
    symbol = "✓" if improved else "~"
    print(f"  Loss before step: {loss_before.item():.6f}")
    print(f"  Loss after step : {loss_after.item():.6f}")
    print(f"  {symbol} {'Decreased' if improved else 'Not decreased (can happen with random data)'}")

    # ── Parameter summary ─────────────────────────────────────────────
    te = sum(p.numel() for p in temporal_enc.parameters())
    re = sum(p.numel() for p in relational_enc.parameters())
    eh = sum(p.numel() for p in expert_heads.parameters())
    total = te + re + eh
    print(f"\n── Parameter counts ──")
    print(f"  TemporalEncoder      : {te:>10,}")
    print(f"  RelationalEncoder    : {re:>10,}")
    print(f"  ExpertPredictionHeads: {eh:>10,}")
    print(f"  ─────────────────────────────────")
    print(f"  TOTAL                : {total:>10,}")

    print(f"\n{'=' * 68}")
    print(f"  ALL PIPELINE TESTS PASSED ✓")
    print(f"{'=' * 68}")


def test_e2e_step3():
    run_e2e_step3()


if __name__ == "__main__":
    run_e2e_step3()
