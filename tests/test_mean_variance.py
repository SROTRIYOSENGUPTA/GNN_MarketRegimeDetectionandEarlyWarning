import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from market_regime_gnn.portfolio.mean_variance import (
    estimate_covariance,
    optimize_weights,
    quintile_direction,
    shrink_signal,
)


def test_estimate_covariance_matches_manual_calc():
    rng = np.random.default_rng(0)
    rets = rng.normal(scale=0.01, size=(200, 4))
    cov = estimate_covariance(rets, ridge=0.0)
    expected = np.cov(rets, rowvar=False)
    np.testing.assert_allclose(cov, expected, atol=1e-8)


def test_estimate_covariance_handles_degenerate_window():
    cov = estimate_covariance(np.zeros((1, 3)), ridge=1e-6)
    assert cov.shape == (3, 3)
    np.testing.assert_allclose(np.diag(cov), [1e-6] * 3)


def test_shrink_signal_bounds():
    mu = np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(shrink_signal(mu, 0.0), mu)
    np.testing.assert_allclose(shrink_signal(mu, 1.0), np.zeros_like(mu))
    half = shrink_signal(mu, 0.5)
    np.testing.assert_allclose(half, mu * 0.5)


def test_shrink_signal_rejects_out_of_range():
    with pytest.raises(ValueError):
        shrink_signal(np.array([1.0]), 1.5)


def test_quintile_direction_extremes_are_long_short():
    mu = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    direction = quintile_direction(mu, low_q=0.2, high_q=0.8)
    assert direction[0] == -1  # most negative -> short
    assert direction[-1] == 1  # most positive -> long
    assert direction[2] == 0   # middle -> excluded


def test_optimize_weights_unconstrained_matches_closed_form():
    mu = np.array([0.02, 0.01, -0.005])
    cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.03, 0.0], [0.0, 0.0, 0.02]])
    w = optimize_weights(mu, cov, risk_aversion=2.0)
    expected = np.linalg.solve(cov, mu) / 2.0
    np.testing.assert_allclose(w, expected, atol=1e-8)


def test_optimize_weights_dollar_neutral_sums_to_zero():
    mu = np.array([0.02, -0.01, 0.015, -0.02])
    cov = np.eye(4) * 0.01
    w = optimize_weights(mu, cov, risk_aversion=1.0, dollar_neutral=True, gross_cap=2.0)
    assert abs(w.sum()) < 1e-6
    assert np.abs(w).sum() <= 2.0 + 1e-6


def test_optimize_weights_respects_allowed_sign():
    mu = np.array([0.02, 0.02, -0.02, -0.02])
    cov = np.eye(4) * 0.01
    direction = np.array([1, 0, -1, 0])
    w = optimize_weights(mu, cov, risk_aversion=1.0, gross_cap=2.0, allowed_sign=direction)
    assert w[0] >= -1e-8
    assert abs(w[1]) < 1e-6
    assert w[2] <= 1e-8
    assert abs(w[3]) < 1e-6
