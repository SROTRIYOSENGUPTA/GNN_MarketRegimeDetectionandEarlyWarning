from pathlib import Path

import pytest

pytest.importorskip("torch_geometric")
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn import RegimeConfig
from market_regime_gnn.data.hetero_dataset import RegimeDataset, generate_synthetic_data


def test_regime_dataset_keeps_earliest_valid_target_day():
    cfg = RegimeConfig()
    data = generate_synthetic_data(n_stocks=8, n_days=200, seed=42)

    ds = RegimeDataset(
        features=data[0],
        dates=data[1],
        sector_map=data[2],
        subind_map=data[3],
        returns=data[4],
        regime_labels=data[5],
        transition_labels=data[6],
        cfg=cfg,
    )

    expected_first = cfg.rolling_zscore_window - 1 + (cfg.seq_len - 1)
    expected_len = len(data[1]) - cfg.transition_horizon_max - expected_first

    assert ds.date_indices[0] == expected_first
    assert len(ds) == expected_len
