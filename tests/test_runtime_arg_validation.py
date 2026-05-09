import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn.run_real_data import validate_runtime_args as validate_regime_runtime_args


def test_regime_runtime_args_accept_valid_defaults():
    validate_regime_runtime_args(
        epochs=5,
        batch_size=2,
        grad_accum_steps=2,
        corr_top_k=10,
        corr_bot_k=5,
        seq_len=30,
        rolling_zscore_window=60,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            dict(
                epochs=0,
                batch_size=2,
                grad_accum_steps=2,
                corr_top_k=10,
                corr_bot_k=5,
                seq_len=30,
                rolling_zscore_window=60,
            ),
            "epochs must be a positive integer",
        ),
        (
            dict(
                epochs=5,
                batch_size=0,
                grad_accum_steps=2,
                corr_top_k=10,
                corr_bot_k=5,
                seq_len=30,
                rolling_zscore_window=60,
            ),
            "batch_size must be a positive integer",
        ),
        (
            dict(
                epochs=5,
                batch_size=2,
                grad_accum_steps=2,
                corr_top_k=-1,
                corr_bot_k=5,
                seq_len=30,
                rolling_zscore_window=60,
            ),
            "corr_top_k must be a non-negative integer",
        ),
        (
            dict(
                epochs=5,
                batch_size=2,
                grad_accum_steps=2,
                corr_top_k=10,
                corr_bot_k=5,
                seq_len=0,
                rolling_zscore_window=60,
            ),
            "seq_len must be a positive integer",
        ),
    ],
)
def test_regime_runtime_args_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        validate_regime_runtime_args(**kwargs)
