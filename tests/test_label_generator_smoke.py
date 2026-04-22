import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn.data import label_generator


def test_label_generation_smoke():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    spy_close = pd.Series(
        np.linspace(100.0, 125.0, len(dates)) + np.sin(np.arange(len(dates))),
        index=dates,
    )
    returns_df = pd.DataFrame(
        {
            "A": np.sin(np.arange(len(dates)) / 7.0) * 0.01,
            "B": np.cos(np.arange(len(dates)) / 9.0) * 0.012,
            "C": np.sin(np.arange(len(dates)) / 11.0) * 0.008,
        },
        index=dates,
    )

    avg_corr = label_generator.compute_avg_cross_correlation(returns_df, window=30)
    labels = label_generator.generate_market_labels(
        pd.DataFrame({"Close": spy_close}),
        avg_corr,
        min_history=60,
    )

    assert len(labels) == len(dates)
    assert set(["regime_label", "transition_label", "regime_name"]).issubset(labels.columns)
    assert labels["regime_label"].between(0, 3).all()
    assert labels["transition_label"].isin([0, 1]).all()
