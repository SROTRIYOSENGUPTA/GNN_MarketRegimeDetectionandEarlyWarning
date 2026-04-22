import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn._legacy.run_real_data import SP500_SAMPLE


def test_curated_market_regime_universe_shape():
    assert len(SP500_SAMPLE) == 10
    assert sum(len(tickers) for tickers in SP500_SAMPLE.values()) == 30
