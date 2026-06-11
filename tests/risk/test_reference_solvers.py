"""Cross-solver equivalence tests against independent reference implementations.

The direct Clarabel formulations are checked against solvers with completely
independent code paths:

- SampleCovariance against scipy's SLSQP on the raw quadratic form.
- CVar against scipy's HiGHS LP solver on the Rockafellar-Uryasev LP.
- FactorModel against SampleCovariance on the reconstructed full covariance
  matrix beta' cov_f beta + diag(idio^2).

Agreement across these pairs would only happen by accident if either side
built the wrong conic program.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import LinearConstraint, linprog, minimize

from cvx.core import Variable
from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio import minrisk_problem
from cvx.risk.sample import SampleCovariance


def random_covariance(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return a random strictly positive definite covariance matrix."""
    g = rng.standard_normal((n, n))
    return g.T @ g / n + 0.1 * np.eye(n)


def reference_min_volatility(
    cov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    extra: list[tuple[np.ndarray, float | None, float | None]] | None = None,
) -> float:
    """Solve the minimum-volatility problem with SLSQP and return the optimal volatility."""
    n = cov.shape[0]
    constraints = [LinearConstraint(np.ones(n), 1.0, 1.0)]
    for a, lb, ub in extra or []:
        constraints.append(
            LinearConstraint(a, -np.inf if lb is None else lb, np.inf if ub is None else ub),
        )
    result = minimize(
        lambda w: w @ cov @ w,
        x0=np.full(n, 1.0 / n),
        bounds=list(zip(lower, upper, strict=True)),
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    assert result.success, result.message
    return float(np.sqrt(result.fun))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("n", [3, 8])
def test_sample_minrisk_matches_slsqp(seed, n):
    """The Clarabel SOC solution must match scipy SLSQP on the raw quadratic form."""
    rng = np.random.default_rng(seed)
    cov = random_covariance(rng, n)
    lower = np.zeros(n)
    upper = rng.uniform(0.5, 1.0, n)

    model = SampleCovariance(num=n)
    model.update(cov=cov, lower_assets=lower, upper_assets=upper)
    weights = Variable(n)
    problem = minrisk_problem(model, weights)
    problem.solve()

    assert "Solved" in problem.status
    expected = reference_min_volatility(cov, lower, upper)
    assert np.isclose(problem.value, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_sample_minrisk_with_extra_constraints_matches_slsqp(seed):
    """User-supplied linear constraints must be translated into the conic program correctly."""
    n = 5
    rng = np.random.default_rng(seed)
    cov = random_covariance(rng, n)
    lower = np.zeros(n)
    upper = np.ones(n)
    extra = [
        (np.eye(n)[0], 0.15, None),  # at least 15% in the first asset
        (rng.uniform(0, 1, n), None, 0.8),  # random exposure capped at 0.8
        (np.eye(n)[1] - np.eye(n)[2], 0.0, 0.0),  # equal weight in assets 2 and 3
    ]

    model = SampleCovariance(num=n)
    model.update(cov=cov, lower_assets=lower, upper_assets=upper)
    weights = Variable(n)
    problem = minrisk_problem(model, weights, constraints=extra)
    problem.solve()

    assert "Solved" in problem.status
    w = np.array(weights.value)
    assert w[0] >= 0.15 - 1e-6
    assert np.isclose(w[1], w[2], atol=1e-6)

    expected = reference_min_volatility(cov, lower, upper, extra=extra)
    assert np.isclose(problem.value, expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_cvar_minrisk_matches_linprog(seed):
    """The Clarabel CVaR solution must match scipy's HiGHS on the Rockafellar-Uryasev LP."""
    n_scenarios, m, alpha = 40, 6, 0.9
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((n_scenarios, m)) * 0.02

    model = CVar(alpha=alpha, n=n_scenarios, m=m)
    model.update(returns=returns, lower_assets=np.zeros(m), upper_assets=np.ones(m))
    weights = Variable(m)
    problem = minrisk_problem(model, weights)
    problem.solve()

    assert "Solved" in problem.status

    # Reference LP over x = [w, gamma, u]: min gamma + sum(u)/k
    # s.t. -R w - gamma - u <= 0, sum(w) = 1, 0 <= w <= 1, u >= 0.
    k = model.k
    c = np.concatenate([np.zeros(m), [1.0], np.full(n_scenarios, 1.0 / k)])
    a_ub = np.hstack([-returns, -np.ones((n_scenarios, 1)), -np.eye(n_scenarios)])
    a_eq = np.concatenate([np.ones(m), [0.0], np.zeros(n_scenarios)]).reshape(1, -1)
    bounds = [(0.0, 1.0)] * m + [(None, None)] + [(0.0, None)] * n_scenarios
    reference = linprog(c, A_ub=a_ub, b_ub=np.zeros(n_scenarios), A_eq=a_eq, b_eq=[1.0], bounds=bounds)
    assert reference.success, reference.message

    assert np.isclose(problem.value, reference.fun, rtol=1e-6, atol=1e-8)
    # The optimal weights must achieve the optimal CVaR (solutions may be non-unique).
    assert np.isclose(model.estimate(np.array(weights.value)), reference.fun, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("seed", range(3))
def test_sample_tracking_error_matches_slsqp(seed):
    """With a base portfolio, the problem must minimize the risk of (w - base)."""
    n = 5
    rng = np.random.default_rng(seed)
    cov = random_covariance(rng, n)
    base = rng.dirichlet(np.ones(n))

    model = SampleCovariance(num=n)
    model.update(cov=cov, lower_assets=np.zeros(n), upper_assets=np.ones(n))
    weights = Variable(n)
    problem = minrisk_problem(model, weights, base=base)
    problem.solve()
    assert "Solved" in problem.status

    constraints = [LinearConstraint(np.ones(n), 1.0, 1.0)]
    result = minimize(
        lambda w: (w - base) @ cov @ (w - base),
        x0=np.full(n, 1.0 / n),
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    assert result.success, result.message

    assert np.isclose(problem.value, np.sqrt(result.fun), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("seed", range(3))
def test_factor_tracking_error_matches_slsqp(seed):
    """With a base portfolio, the factor problem must minimize the full tracking error.

    Both the systematic and the idiosyncratic term must be evaluated on the
    active position (w - base).
    """
    n, k = 5, 2
    rng = np.random.default_rng(seed)
    cov_f = random_covariance(rng, k)
    exposure = rng.standard_normal((k, n))
    idio = rng.uniform(0.05, 0.3, n)
    base = rng.dirichlet(np.ones(n))

    model = FactorModel(assets=n, k=k)
    model.update(
        exposure=exposure,
        cov=cov_f,
        idiosyncratic_risk=idio,
        lower_assets=np.zeros(n),
        upper_assets=np.ones(n),
        lower_factors=-100 * np.ones(k),
        upper_factors=100 * np.ones(k),
    )
    weights = Variable(n)
    problem = minrisk_problem(model, weights, base=base)
    problem.solve()
    assert "Solved" in problem.status

    def objective(w):
        y = exposure @ (w - base)
        return y @ cov_f @ y + np.sum((idio * (w - base)) ** 2)

    constraints = [LinearConstraint(np.ones(n), 1.0, 1.0)]
    result = minimize(
        objective,
        x0=np.full(n, 1.0 / n),
        bounds=[(0.0, 1.0)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    assert result.success, result.message

    assert np.isclose(problem.value, np.sqrt(result.fun), rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("use_base", [False, True])
def test_factor_minrisk_matches_sample_on_reconstructed_cov(seed, use_base):
    """With non-binding factor bounds, FactorModel must agree with SampleCovariance.

    The factor model on (exposure, cov_f, idio) and the sample model on the
    reconstructed covariance beta' cov_f beta + diag(idio^2) describe the same
    risk, so their minimum-risk problems must have identical optima — with and
    without a base portfolio.
    """
    n, k = 8, 3
    rng = np.random.default_rng(seed)
    cov_f = random_covariance(rng, k)
    exposure = rng.standard_normal((k, n))
    idio = rng.uniform(0.05, 0.3, n)
    base = rng.dirichlet(np.ones(n)) if use_base else 0.0

    factor_model = FactorModel(assets=n, k=k)
    factor_model.update(
        exposure=exposure,
        cov=cov_f,
        idiosyncratic_risk=idio,
        lower_assets=np.zeros(n),
        upper_assets=np.ones(n),
        lower_factors=-100 * np.ones(k),
        upper_factors=100 * np.ones(k),
    )
    w_factor = Variable(n)
    factor_problem = minrisk_problem(factor_model, w_factor, base=base)
    factor_problem.solve()
    assert "Solved" in factor_problem.status

    sample_model = SampleCovariance(num=n)
    sample_model.update(
        cov=exposure.T @ cov_f @ exposure + np.diag(idio**2),
        lower_assets=np.zeros(n),
        upper_assets=np.ones(n),
    )
    w_sample = Variable(n)
    sample_problem = minrisk_problem(sample_model, w_sample, base=base)
    sample_problem.solve()
    assert "Solved" in sample_problem.status

    assert np.isclose(factor_problem.value, sample_problem.value, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.array(w_factor.value), np.array(w_sample.value), atol=1e-4)
