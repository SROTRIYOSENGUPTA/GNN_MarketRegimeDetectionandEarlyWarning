import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

SCRIPT_PATH = REPO_ROOT / "scripts" / "run_mpt_backtest.py"


def _load_backtest_module():
    spec = importlib.util.spec_from_file_location("run_mpt_backtest", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_bundle(path, seed, n_stocks=6, n_hist=300, n_val=10):
    rng = np.random.default_rng(seed)
    tickers = np.array([f"T{i}" for i in range(n_stocks)], dtype=object)
    hist_dates = np.array(
        [f"2023-{1 + (i // 28):02d}-{1 + (i % 28):02d}" for i in range(n_hist)]
    )
    rets_hist = rng.normal(scale=0.01, size=(n_hist, n_stocks)).astype(np.float32)

    val_dates = hist_dates[-n_val * 5 :: 5][:n_val]
    logits = rng.normal(size=(n_val, n_stocks, 3))
    val_probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    val_fwd_ret = rng.normal(scale=0.02, size=(n_val, n_stocks)).astype(np.float32)

    np.savez_compressed(
        path,
        edge_mode="bloomberg" if seed == 42 else "none",
        seed=seed,
        tickers=tickers,
        val_dates=val_dates,
        val_probs=val_probs.astype(np.float32),
        val_fwd_ret=val_fwd_ret,
        rets_hist=rets_hist,
        hist_dates=hist_dates,
    )


def test_cli_help():
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    assert "--predictions" in result.stdout
    assert "--none-predictions" in result.stdout


def test_score_mu_bounded():
    mod = _load_backtest_module()
    probs = np.array([[[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]]])
    mu = mod.score_mu(probs)
    assert mu.shape == (1, 2)
    assert np.all(mu >= -1.0) and np.all(mu <= 1.0)
    np.testing.assert_allclose(mu[0, 0], 0.6)  # P(up) - P(down) = 0.7 - 0.1
    np.testing.assert_allclose(mu[0, 1], -0.7)


@pytest.mark.parametrize("sizing", ["mean_variance", "equal_weight_picks", "equal_weight_universe"])
def test_run_config_end_to_end(tmp_path, sizing):
    mod = _load_backtest_module()
    bundle_path = tmp_path / "bundle.npz"
    _synthetic_bundle(bundle_path, seed=42)
    bundle = mod.load_bundle(bundle_path)
    mu = mod.score_mu(bundle["val_probs"]) if sizing != "equal_weight_universe" else None

    result = mod.run_config(
        name=f"test_{sizing}",
        mu_all=mu,
        cov_window=mod.GNN_COV_WINDOW,
        sizing=sizing,
        rets_hist=bundle["rets_hist"],
        hist_dates=bundle["hist_dates"],
        val_dates=bundle["val_dates"],
        val_fwd_ret=bundle["val_fwd_ret"],
        shrinkage=0.3,
        gross_cap=2.0,
        dollar_neutral=True,
        cost_bps=5.0,
    )

    assert result["n_periods"] == len(bundle["val_dates"])
    assert len(result["history"]) == result["n_periods"]
    assert np.isfinite(result["total_return"])
    assert result["max_drawdown"] <= 0.0


def test_main_writes_output_with_both_bundles(tmp_path):
    bloomberg_path = tmp_path / "bloomberg.npz"
    none_path = tmp_path / "none.npz"
    _synthetic_bundle(bloomberg_path, seed=42)
    _synthetic_bundle(none_path, seed=7)
    out_path = tmp_path / "backtest.json"

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT_PATH),
            "--predictions", str(bloomberg_path),
            "--none-predictions", str(none_path),
            "--output", str(out_path),
        ],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()

    import json
    data = json.loads(out_path.read_text())
    assert set(data["configs"].keys()) == {
        "A_bloomberg_gnn_cov_mv", "B_bloomberg_sample_cov_mv",
        "E_bloomberg_equal_weight_picks", "F_equal_weight_universe",
        "C_none_gnn_cov_mv", "D_none_sample_cov_mv",
    }
