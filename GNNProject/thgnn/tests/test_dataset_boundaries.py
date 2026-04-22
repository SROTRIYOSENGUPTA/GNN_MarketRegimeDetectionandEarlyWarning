"""
Boundary tests for THGNNDataset sample eligibility.
"""

import datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch_geometric")

REPO_ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(REPO_ROOT))

from GNNProject.thgnn.config import THGNNConfig
from GNNProject.thgnn.data.dataset import THGNNDataset


def test_thgnn_dataset_keeps_earliest_valid_target_day():
    cfg = THGNNConfig()
    rng = np.random.default_rng(42)
    n_days = 200
    n_stocks = 8

    dates = [
        (datetime.date(2018, 1, 1) + datetime.timedelta(days=i)).isoformat()
        for i in range(n_days)
    ]
    features = {
        sid: rng.standard_normal((n_days, cfg.num_features)).astype(np.float32)
        for sid in range(n_stocks)
    }
    returns = {
        sid: (rng.standard_normal(n_days) * 0.02).astype(np.float32)
        for sid in range(n_stocks)
    }
    sector_map = {sid: int(rng.integers(0, 11)) for sid in range(n_stocks)}
    subind_map = {sid: int(rng.integers(0, 50)) for sid in range(n_stocks)}

    ds = THGNNDataset(
        features=features,
        dates=dates,
        sector_map=sector_map,
        subind_map=subind_map,
        returns=returns,
        cfg=cfg,
    )

    expected_first = cfg.rolling_zscore_window - 1 + (cfg.min_trading_days - 1)
    expected_len = len(dates) - cfg.forecast_horizon - expected_first

    assert ds.date_indices[0] == expected_first
    assert len(ds) == expected_len
