from __future__ import annotations

from .._loader import load_legacy_module

_hetero_dataset = load_legacy_module("data.hetero_dataset")

EDGE_TYPES = _hetero_dataset.EDGE_TYPES
RegimeDataset = _hetero_dataset.RegimeDataset
build_regime_dataloader = _hetero_dataset.build_regime_dataloader
regime_collate_fn = _hetero_dataset.regime_collate_fn
generate_synthetic_data = _hetero_dataset.generate_synthetic_data
rolling_zscore = _hetero_dataset.rolling_zscore
compute_rolling_corr = _hetero_dataset.compute_rolling_corr

__all__ = [
    "EDGE_TYPES",
    "RegimeDataset",
    "build_regime_dataloader",
    "regime_collate_fn",
    "generate_synthetic_data",
    "rolling_zscore",
    "compute_rolling_corr",
]
