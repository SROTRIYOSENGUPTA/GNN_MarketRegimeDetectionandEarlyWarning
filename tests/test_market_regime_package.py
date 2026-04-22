import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn import RegimeConfig
from market_regime_gnn._loader import load_legacy_module
from market_regime_gnn.data import label_generator


def test_market_regime_wrapper_imports_do_not_pollute_bare_module_names():
    assert RegimeConfig.__name__ == "RegimeConfig"
    assert label_generator.__name__ == "market_regime_gnn.data.label_generator"
    assert "config" not in sys.modules
    assert "data" not in sys.modules
    assert "models" not in sys.modules


def test_market_regime_wrapper_label_generator_runs():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    spy = pd.DataFrame(
        {
            "Close": np.linspace(100.0, 130.0, len(dates))
            + np.cos(np.arange(len(dates)) / 4.0)
        },
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
    labels = label_generator.generate_market_labels(spy, avg_corr, min_history=60)

    assert len(labels) == len(dates)
    assert labels["regime_label"].between(0, 3).all()
    assert labels["transition_label"].isin([0, 1]).all()


def test_market_regime_loader_prefers_packaged_legacy_modules():
    config_module = load_legacy_module("config")
    label_module = load_legacy_module("data.label_generator")

    assert Path(config_module.__file__).resolve() == (
        REPO_ROOT / "market_regime_gnn" / "_legacy" / "config.py"
    )
    assert Path(label_module.__file__).resolve() == (
        REPO_ROOT
        / "market_regime_gnn"
        / "_legacy"
        / "data"
        / "label_generator.py"
    )
