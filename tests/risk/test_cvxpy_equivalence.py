"""Equivalence tests against cvxpy reference formulations.

These tests require cvxpy (available via ``uv sync --group benchmark``) and
are skipped when it is not installed. They check that the hand-built Clarabel
conic programs match what cvxpy produces for the same mathematical problem,
including the base-portfolio (tracking error) variant.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvx.core import Variable
from cvx.risk.cvar import CVar
from cvx.risk.portfolio import minrisk_problem
from cvx.risk.sample import SampleCovariance

cp = pytest.importorskip("cvxpy")


def random_covariance(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return a random strictly positive definite covariance matrix."""
    g = rng.standard_normal((n, n))
    return g.T @ g / n + 0.1 * np.eye(n)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("use_base", [False, True])
def test_sample_minrisk_matches_cvxpy(seed, use_base):
    """The direct SOC program must match the cvxpy formulation of min ||chol (w - base)||."""
    n = 6
    rng = np.random.default_rng(seed)
    cov = random_covariance(rng, n)
    base = rng.dirichlet(np.ones(n)) if use_base else np.zeros(n)

    model = SampleCovariance(num=n)
    model.update(cov=cov, lower_assets=np.zeros(n), upper_assets=np.ones(n))
    weights = Variable(n)
    problem = minrisk_problem(model, weights, base=base if use_base else 0.0)
    problem.solve()
    assert "Solved" in problem.status

    chol = np.linalg.cholesky(cov).T
    w = cp.Variable(n)
    reference = cp.Problem(
        cp.Minimize(cp.norm(chol @ (w - base), 2)),
        [cp.sum(w) == 1, w >= 0, w <= 1],
    )
    reference.solve(solver="CLARABEL")

    assert np.isclose(problem.value, reference.value, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.array(weights.value), w.value, atol=1e-5)


@pytest.mark.parametrize("seed", range(3))
def test_cvar_minrisk_matches_cvxpy(seed):
    """The direct CVaR LP must match the cvxpy Rockafellar-Uryasev formulation."""
    n_scenarios, m, alpha = 50, 5, 0.9
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((n_scenarios, m)) * 0.02

    model = CVar(alpha=alpha, n=n_scenarios, m=m)
    model.update(returns=returns, lower_assets=np.zeros(m), upper_assets=np.ones(m))
    weights = Variable(m)
    problem = minrisk_problem(model, weights)
    problem.solve()
    assert "Solved" in problem.status

    k = model.k
    w = cp.Variable(m)
    gamma = cp.Variable()
    u = cp.Variable(n_scenarios)
    reference = cp.Problem(
        cp.Minimize(gamma + cp.sum(u) / k),
        [u >= -returns @ w - gamma, u >= 0, cp.sum(w) == 1, w >= 0, w <= 1],
    )
    reference.solve(solver="CLARABEL")

    assert np.isclose(problem.value, reference.value, rtol=1e-6, atol=1e-8)
    assert np.isclose(model.estimate(np.array(weights.value)), reference.value, rtol=1e-5, atol=1e-6)
