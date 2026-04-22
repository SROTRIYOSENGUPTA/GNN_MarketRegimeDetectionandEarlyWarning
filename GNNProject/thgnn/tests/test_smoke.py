"""
End-to-end smoke test: synthetic data → Dataset → DataLoader → TemporalEncoder.
Validates that all tensor shapes are correct and the pipeline runs without error.
"""

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


def generate_synthetic_data(
    n_stocks: int = 20,
    n_days: int = 200,
    n_features: int = 37,
    seed: int = 42,
):
    """Create minimal synthetic data to exercise the full pipeline."""
    rng = np.random.default_rng(seed)
    import datetime
    base = datetime.date(2018, 1, 1)
    dates = [(base + datetime.timedelta(days=i)).isoformat() for i in range(n_days)]
    stock_ids = list(range(n_stocks))

    features = {}
    returns = {}
    sector_map = {}
    subind_map = {}

    for sid in stock_ids:
        features[sid] = rng.standard_normal((n_days, n_features)).astype(np.float32)
        returns[sid] = rng.standard_normal(n_days).astype(np.float32) * 0.02
        sector_map[sid] = int(rng.integers(0, 11))
        subind_map[sid] = int(rng.integers(0, 50))

    return features, dates, sector_map, subind_map, returns, stock_ids


def make_cfg() -> THGNNConfig:
    """Build a small config suitable for fast synthetic tests."""
    cfg = THGNNConfig()
    cfg.top_k_corr = 5
    cfg.bot_k_corr = 5
    cfg.rand_mid_k = 5
    return cfg


def build_dataset(cfg: THGNNConfig) -> THGNNDataset:
    """Create a small synthetic dataset for smoke tests."""
    features, dates, sector_map, subind_map, returns, _ = generate_synthetic_data()
    return THGNNDataset(
        features=features,
        dates=dates,
        sector_map=sector_map,
        subind_map=subind_map,
        returns=returns,
        cfg=cfg,
    )


def build_batch(ds: THGNNDataset, cfg: THGNNConfig):
    """Fetch the first batch from the PyG dataloader."""
    loader = build_dataloader(ds, cfg, shuffle=False)
    return next(iter(loader))


@pytest.fixture
def cfg():
    return make_cfg()


@pytest.fixture
def ds(cfg):
    return build_dataset(cfg)


@pytest.fixture
def batch(ds, cfg):
    return build_batch(ds, cfg)


def test_dataset_shapes(ds, cfg):
    """Verify shapes of a single Data object from the dataset."""

    print(f"Dataset length: {len(ds)} trading-day snapshots")
    assert len(ds) > 0

    data = ds[0]
    N = data.num_nodes
    E = data.edge_index.shape[1]

    print(f"\n── Single sample (day = {data.date}) ──")
    print(f"  x              : {tuple(data.x.shape)}          (N={N}, L=30, F=37)")
    print(f"  edge_index     : {tuple(data.edge_index.shape)}       (2, E={E})")
    print(f"  edge_attr      : {tuple(data.edge_attr.shape)}      (E, 5)")
    print(f"  edge_type      : {tuple(data.edge_type.shape)}          (E,)")
    print(f"  edge_weight    : {tuple(data.edge_weight.shape)}          (E,)")
    print(f"  baseline_z     : {tuple(data.baseline_z.shape)}          (E,)")
    print(f"  target_z_resid : {tuple(data.target_z_resid.shape)}          (E,)")

    assert data.x.shape[1] == cfg.seq_len
    assert data.x.shape[2] == cfg.num_features
    assert data.edge_attr.shape[1] == cfg.num_edge_attr
    assert data.edge_type.max() <= 2
    assert data.edge_type.min() >= 0

    print("  ✓ All single-sample shapes correct.\n")


def test_dataloader_batching(batch, cfg):
    """Verify PyG batching produces correct concatenated shapes."""
    N_batch = batch.x.shape[0]  # sum of N_t across batch items
    E_batch = batch.edge_index.shape[1]

    print(f"── Batch (batch_size={cfg.batch_size}) ──")
    print(f"  batch.x             : {tuple(batch.x.shape)}      (ΣN, 30, 37)")
    print(f"  batch.edge_index    : {tuple(batch.edge_index.shape)}    (2, ΣE={E_batch})")
    print(f"  batch.edge_attr     : {tuple(batch.edge_attr.shape)}   (ΣE, 5)")
    print(f"  batch.edge_type     : {tuple(batch.edge_type.shape)}       (ΣE,)")
    print(f"  batch.baseline_z    : {tuple(batch.baseline_z.shape)}       (ΣE,)")
    print(f"  batch.target_z_resid: {tuple(batch.target_z_resid.shape)}       (ΣE,)")
    print(f"  batch.batch         : {tuple(batch.batch.shape)}       (ΣN,)  — graph assignment")

    assert batch.x.shape[1] == cfg.seq_len
    assert batch.x.shape[2] == cfg.num_features
    print("  ✓ Batched shapes correct.\n")


def test_temporal_encoder_on_batch(batch, cfg):
    """Run the Temporal Encoder on a real batched input."""
    encoder = TemporalEncoder(cfg)
    encoder.eval()

    with torch.no_grad():
        h0 = encoder(batch.x)    # (ΣN, 512)

    print(f"── TemporalEncoder on batch ──")
    print(f"  Input  : {tuple(batch.x.shape)}")
    print(f"  Output : {tuple(h0.shape)}  — expected (ΣN={batch.x.shape[0]}, 512)")
    assert h0.shape == (batch.x.shape[0], cfg.node_embed_dim)
    assert torch.isfinite(h0).all(), "Non-finite values detected in encoder output!"
    print("  ✓ Temporal Encoder forward pass correct.\n")


def test_gradient_flow(cfg):
    """Verify gradients flow back through the entire encoder."""
    encoder = TemporalEncoder(cfg)
    x = torch.randn(10, cfg.seq_len, cfg.num_features, requires_grad=True)
    h0 = encoder(x)
    loss = h0.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input!"
    assert x.grad.abs().sum() > 0, "Zero gradients!"
    print("── Gradient flow test ──")
    print(f"  ∂loss/∂x norm : {x.grad.norm().item():.6f}")
    print("  ✓ Gradients flow correctly.\n")


if __name__ == "__main__":
    print("=" * 64)
    print("  THGNN Smoke Tests — Step 1")
    print("=" * 64 + "\n")

    cfg = make_cfg()
    ds = build_dataset(cfg)
    batch = build_batch(ds, cfg)

    test_dataset_shapes(ds, cfg)
    test_dataloader_batching(batch, cfg)
    test_temporal_encoder_on_batch(batch, cfg)
    test_gradient_flow(cfg)

    print("=" * 64)
    print("  ALL TESTS PASSED ✓")
    print("=" * 64)
