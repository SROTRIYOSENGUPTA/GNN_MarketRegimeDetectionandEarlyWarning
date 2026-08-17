"""
Markowitz mean-variance portfolio construction on top of GNN-derived
return and covariance signals.

Intentionally decoupled from any GNN implementation: this module only
consumes plain numpy arrays (expected returns, a covariance matrix, and
optional per-asset direction constraints), so it works the same whether
`mu`/`cov` come from the cross-sectional rank model, a naive baseline, or
raw sample statistics.
"""
from __future__ import annotations

import numpy as np


def estimate_covariance(rets_window: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Rolling-window covariance as correlation * outer(std, std) + ridge*I.

    `rets_window` is (W, N) daily returns. This is the same correlation
    estimator the GNN's correlation edges are built from (see
    `correlation_edges_topk` in scripts/run_xsec_rank.py), so it is a
    GNN-derived covariance rather than an unrelated estimator. The ridge
    term keeps the matrix well-conditioned for inversion when W < N or
    returns are highly collinear.
    """
    W, N = rets_window.shape
    if W < 2:
        return np.eye(N, dtype=np.float64) * ridge
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(rets_window.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)
    std = np.nan_to_num(rets_window.std(axis=0, ddof=1), nan=0.0)
    cov = corr * np.outer(std, std)
    cov = cov + ridge * np.eye(N)
    return cov


def shrink_signal(
    mu: np.ndarray, shrinkage: float, prior: np.ndarray | None = None
) -> np.ndarray:
    """Convex-shrink a noisy return signal toward a prior (default: no view).

    shrinkage=0 trusts `mu` fully; shrinkage=1 ignores it and returns the
    prior. Guards against Markowitz's well-documented sensitivity to noisy
    inputs (Michaud 1989's "error maximization") given this model's modest
    out-of-sample accuracy (macro-F1 ~0.37 on a 3-class task).
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")
    if prior is None:
        prior = np.zeros_like(mu)
    return (1.0 - shrinkage) * mu + shrinkage * prior


def quintile_direction(mu: np.ndarray, low_q: float = 0.2, high_q: float = 0.8) -> np.ndarray:
    """+1 for top-quintile mu (eligible long), -1 for bottom-quintile (eligible short), 0 otherwise."""
    ranks = np.argsort(np.argsort(mu)) / max(len(mu) - 1, 1)
    direction = np.zeros(len(mu), dtype=np.int64)
    direction[ranks >= high_q] = 1
    direction[ranks < low_q] = -1
    return direction


def optimize_weights(
    mu: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float = 1.0,
    gross_cap: float | None = None,
    dollar_neutral: bool = False,
    allowed_sign: np.ndarray | None = None,
    turnover_penalty: float = 0.0,
    w_prev: np.ndarray | None = None,
) -> np.ndarray:
    """Solve max_w  mu.w - (risk_aversion/2) w.cov.w - lambda ||w - w_prev||_1.

    With no constraints and no turnover penalty this reduces to the
    closed-form classic Markowitz solution w* = cov^-1 mu / risk_aversion
    (used directly, and as the reference solution in tests). Any constraint
    (gross exposure cap, dollar-neutrality, or long/short/exclude direction
    bounds — e.g. from `quintile_direction`) or a nonzero turnover penalty
    routes to a convex solve via cvxpy.

    The turnover term is transaction-cost-aware Markowitz: with lambda set
    to the per-unit-turnover cost (e.g. 5bps -> 0.0005), the optimizer only
    trades when the expected mean-variance improvement exceeds the cost of
    getting there from `w_prev`.
    """
    n = mu.shape[0]
    penalized = turnover_penalty > 0.0 and w_prev is not None
    unconstrained = (gross_cap is None and not dollar_neutral
                     and allowed_sign is None and not penalized)
    if unconstrained:
        return np.linalg.solve(cov, mu) / risk_aversion

    import cvxpy as cp

    w = cp.Variable(n)
    obj = mu @ w - (risk_aversion / 2) * cp.quad_form(w, cp.psd_wrap(cov))
    if penalized:
        obj = obj - turnover_penalty * cp.norm(w - w_prev, 1)
    objective = cp.Maximize(obj)
    constraints = []
    if dollar_neutral:
        constraints.append(cp.sum(w) == 0)
    if gross_cap is not None:
        constraints.append(cp.norm(w, 1) <= gross_cap)
    if allowed_sign is not None:
        allowed_sign = np.asarray(allowed_sign)
        if np.any(allowed_sign == 0):
            constraints.append(w[allowed_sign == 0] == 0)
        if np.any(allowed_sign == 1):
            constraints.append(w[allowed_sign == 1] >= 0)
        if np.any(allowed_sign == -1):
            constraints.append(w[allowed_sign == -1] <= 0)

    problem = cp.Problem(objective, constraints)
    problem.solve()
    if w.value is None:
        raise RuntimeError(f"Portfolio optimization failed to converge (status={problem.status})")
    return np.asarray(w.value)
