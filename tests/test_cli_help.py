import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_help(module_name: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_market_regime_cli_help():
    result = _run_help("market_regime_gnn.run_real_data")
    assert result.returncode == 0
    assert "Run the Dynamic Regime GNN real-data pipeline." in result.stdout
    assert "--epochs" in result.stdout
    assert "--train-cutoff" in result.stdout
    assert "cuda:N, or mps" in result.stdout


def test_thgnn_cli_help():
    result = _run_help("GNNProject.thgnn.run_real_data")
    assert result.returncode == 0
    assert "Run the THGNN real-data pipeline." in result.stdout
    assert "--n-stocks" in result.stdout
    assert "--top-k-corr" in result.stdout
    assert "cuda:N, or mps" in result.stdout
