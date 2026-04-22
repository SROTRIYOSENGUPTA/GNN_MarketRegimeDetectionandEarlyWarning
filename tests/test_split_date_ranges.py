import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from GNNProject.thgnn.run_real_data import build_split_date_ranges as build_thgnn_split_date_ranges
from GNNProject.thgnn.run_real_data import validate_split_sample_counts as validate_thgnn_split_sample_counts
from market_regime_gnn.run_real_data import build_split_date_ranges as build_regime_split_date_ranges
from market_regime_gnn.run_real_data import validate_split_sample_counts as validate_regime_split_sample_counts


def test_split_date_ranges_follow_train_cutoff():
    train_range, val_range = build_regime_split_date_ranges(
        "2020-01-01",
        "2024-12-31",
        "2023-06-30",
    )

    assert train_range == ("2020-01-01", "2023-06-30")
    assert val_range == ("2023-07-01", "2024-12-31")

    train_range, val_range = build_thgnn_split_date_ranges(
        "2021-01-01",
        "2024-06-30",
        "2023-12-31",
    )
    assert train_range == ("2021-01-01", "2023-12-31")
    assert val_range == ("2024-01-01", "2024-06-30")


def test_split_date_ranges_allow_empty_validation_suffix():
    _, val_range = build_thgnn_split_date_ranges(
        "2021-01-01",
        "2021-01-31",
        "2021-01-31",
    )
    assert val_range == ("2021-02-01", "2021-01-31")


def test_split_date_ranges_validate_bounds():
    with pytest.raises(ValueError, match="start must be on or before end"):
        build_regime_split_date_ranges("2024-01-02", "2024-01-01", "2024-01-01")

    with pytest.raises(ValueError, match="train_cutoff must fall within"):
        build_thgnn_split_date_ranges("2024-01-01", "2024-12-31", "2025-01-01")


def test_split_count_validation_requires_nonempty_training_split():
    with pytest.raises(ValueError, match="Training split is empty"):
        validate_regime_split_sample_counts(
            0,
            3,
            ("2020-01-01", "2020-06-30"),
            ("2020-07-01", "2020-12-31"),
        )

    with pytest.raises(ValueError, match="Training split is empty"):
        validate_thgnn_split_sample_counts(
            0,
            0,
            ("2021-01-01", "2021-03-31"),
            ("2021-04-01", "2021-06-30"),
        )
