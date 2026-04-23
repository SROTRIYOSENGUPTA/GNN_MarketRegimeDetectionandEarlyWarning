from __future__ import annotations

from market_regime_gnn.config import RegimeConfig
from market_regime_gnn._legacy.data.hetero_dataset import (
    RegimeDataset,
    generate_synthetic_data,
)


def test_regime_dataset_reuses_cached_snapshots(monkeypatch):
    cfg = RegimeConfig()
    cfg.seq_len = 5
    cfg.rolling_zscore_window = 10
    cfg.corr_top_k = 2
    cfg.corr_bot_k = 1

    (
        features,
        dates,
        sector_map,
        subind_map,
        returns_dict,
        regime_labels,
        transition_labels,
    ) = generate_synthetic_data(n_stocks=8, n_days=80, seed=7)

    dataset = RegimeDataset(
        features=features,
        dates=dates,
        sector_map=sector_map,
        subind_map=subind_map,
        returns=returns_dict,
        regime_labels=regime_labels,
        transition_labels=transition_labels,
        cfg=cfg,
    )

    build_calls = {"count": 0}
    original_build_snapshot = dataset._build_snapshot

    def counting_build_snapshot(t_idx, stock_ids):
        build_calls["count"] += 1
        return original_build_snapshot(t_idx, stock_ids)

    monkeypatch.setattr(dataset, "_build_snapshot", counting_build_snapshot)

    first = dataset[0]
    first_count = build_calls["count"]
    second = dataset[0]

    assert first_count == cfg.seq_len
    assert build_calls["count"] == first_count
    assert first["date"] == second["date"]
    assert len(first["snapshots"]) == cfg.seq_len
    assert len(second["snapshots"]) == cfg.seq_len
    for first_snapshot, second_snapshot in zip(first["snapshots"], second["snapshots"]):
        assert first_snapshot is second_snapshot
