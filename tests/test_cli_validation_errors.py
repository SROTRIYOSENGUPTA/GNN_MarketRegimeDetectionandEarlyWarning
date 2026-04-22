import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_module(module_name: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module_name, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_market_regime_cli_reports_validation_errors_without_traceback():
    result = _run_module("market_regime_gnn.run_real_data", "--epochs", "0")
    assert result.returncode == 2
    assert "error: epochs must be a positive integer" in result.stderr
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr


def test_thgnn_cli_reports_validation_errors_without_traceback():
    result = _run_module("GNNProject.thgnn.run_real_data", "--n-stocks", "1")
    assert result.returncode == 2
    assert "error: n_stocks must be an integer >= 2" in result.stderr
    assert "usage:" in result.stderr
    assert "Traceback" not in result.stderr


def test_market_regime_cli_propagates_runtime_value_error(monkeypatch):
    from market_regime_gnn import run_real_data as regime_run_real_data

    legacy_module = regime_run_real_data._legacy()

    def boom(*args, **kwargs):
        raise ValueError("runtime boom")

    monkeypatch.setattr(legacy_module, "main", boom)

    with pytest.raises(ValueError, match="runtime boom"):
        regime_run_real_data.cli([])


def test_thgnn_cli_propagates_runtime_value_error(monkeypatch):
    from GNNProject.thgnn import run_real_data as thgnn_run_real_data

    def boom(*args, **kwargs):
        raise ValueError("runtime boom")

    monkeypatch.setattr(thgnn_run_real_data, "main", boom)

    with pytest.raises(ValueError, match="runtime boom"):
        thgnn_run_real_data.cli([])
