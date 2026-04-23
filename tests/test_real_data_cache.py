from __future__ import annotations

import numpy as np
import pandas as pd

from market_regime_gnn import run_real_data as regime_run_real_data


def _dummy_real_data_payload():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    index = pd.to_datetime(dates)
    features = {
        0: np.ones((3, 37), dtype=np.float32),
        1: np.full((3, 37), 2.0, dtype=np.float32),
    }
    returns_dict = {
        0: np.array([0.0, 0.1, -0.1], dtype=np.float32),
        1: np.array([0.0, -0.1, 0.1], dtype=np.float32),
    }
    spy_df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=index)
    returns_df = pd.DataFrame(
        {
            "AAA": returns_dict[0],
            "BBB": returns_dict[1],
        },
        index=index,
    )
    return (
        features,
        dates,
        {0: 0, 1: 1},
        {0: 0, 1: 1},
        returns_dict,
        spy_df,
        returns_df,
        ["AAA", "BBB"],
        {"AAA": 0, "BBB": 1},
    )


def test_fetch_real_data_reuses_cached_payload(tmp_path, monkeypatch):
    legacy_module = regime_run_real_data._legacy()
    payload = _dummy_real_data_payload()
    calls = {"count": 0}

    def fake_uncached(*args, **kwargs):
        calls["count"] += 1
        return payload

    monkeypatch.setattr(legacy_module, "_fetch_real_data_uncached", fake_uncached)

    first = regime_run_real_data.fetch_real_data(
        start="2020-01-01",
        end="2020-01-31",
        verbose=False,
        cache_dir=tmp_path,
    )
    second = regime_run_real_data.fetch_real_data(
        start="2020-01-01",
        end="2020-01-31",
        verbose=False,
        cache_dir=tmp_path,
    )

    assert calls["count"] == 1
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    assert first[1] == second[1] == payload[1]
    assert np.array_equal(first[0][0], payload[0][0])
    assert np.array_equal(second[0][1], payload[0][1])


def test_fetch_real_data_refresh_cache_forces_rebuild(tmp_path, monkeypatch):
    legacy_module = regime_run_real_data._legacy()
    calls = {"count": 0}

    def fake_uncached(*args, **kwargs):
        calls["count"] += 1
        return _dummy_real_data_payload()

    monkeypatch.setattr(legacy_module, "_fetch_real_data_uncached", fake_uncached)

    regime_run_real_data.fetch_real_data(
        start="2020-01-01",
        end="2020-01-31",
        verbose=False,
        cache_dir=tmp_path,
    )
    regime_run_real_data.fetch_real_data(
        start="2020-01-01",
        end="2020-01-31",
        verbose=False,
        cache_dir=tmp_path,
        refresh_cache=True,
    )

    assert calls["count"] == 2
