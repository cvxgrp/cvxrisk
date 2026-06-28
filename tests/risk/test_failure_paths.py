"""Tests pinning down the behavior of infeasible problems and invalid inputs.

The documented contract on failure is: ``problem.status`` reports the solver
status, ``problem.value`` stays ``None``, and ``weights.value`` is not
populated. These tests make that contract explicit for each model, and pin
the current behavior for degenerate covariance input.
"""

from __future__ import annotations

import numpy as np
import pytest

from cvx.core import Variable
from cvx.risk.cvar import CVar
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio import minrisk_problem
from cvx.risk.sample import SampleCovariance


def assert_solve_failed(problem, weights):
    """Assert the documented failure contract: no value, no weights, informative status."""
    assert problem.status is not None
    assert "Solved" not in problem.status
    assert problem.value is None
    assert weights.value is None


def test_sample_infeasible_bounds():
    """Bounds that cannot sum to one must yield an infeasible status, not garbage weights."""
    n = 3
    model = SampleCovariance(num=n)
    model.update(
        cov=np.eye(n),
        lower_assets=np.zeros(n),
        upper_assets=np.full(n, 0.2),  # sum of upper bounds < 1
    )
    weights = Variable(n)
    problem = minrisk_problem(model, weights)
    problem.solve()
    assert_solve_failed(problem, weights)


def test_sample_contradictory_extra_constraint():
    """An extra constraint contradicting sum(w) == 1 must be reported as infeasible."""
    n = 3
    model = SampleCovariance(num=n)
    model.update(cov=np.eye(n), lower_assets=np.zeros(n), upper_assets=np.ones(n))
    weights = Variable(n)
    # sum(w) == 2 contradicts the built-in budget constraint sum(w) == 1.
    problem = minrisk_problem(model, weights, constraints=[(np.ones(n), 2.0, 2.0)])
    problem.solve()
    assert_solve_failed(problem, weights)


def test_cvar_infeasible_bounds():
    """The CVaR LP must report infeasibility for contradictory weight bounds."""
    n_scenarios, m = 30, 3
    model = CVar(alpha=0.9, n=n_scenarios, m=m)
    model.update(
        returns=np.random.default_rng(0).standard_normal((n_scenarios, m)),
        lower_assets=np.full(m, 0.6),  # sum of lower bounds > 1
        upper_assets=np.ones(m),
    )
    weights = Variable(m)
    problem = minrisk_problem(model, weights)
    problem.solve()
    assert_solve_failed(problem, weights)


def test_factor_infeasible_factor_bounds():
    """Factor exposure bounds that exclude all simplex portfolios must be infeasible."""
    n, k = 3, 2
    model = FactorModel(assets=n, k=k)
    model.update(
        exposure=np.ones((k, n)),
        cov=np.eye(k),
        idiosyncratic_risk=0.1 * np.ones(n),
        lower_assets=np.zeros(n),
        upper_assets=np.ones(n),
        # Any w on the simplex has exposure y = 1 per factor; force y <= 0.5.
        lower_factors=-np.ones(k),
        upper_factors=0.5 * np.ones(k),
    )
    weights = Variable(n)
    problem = minrisk_problem(model, weights)
    problem.solve()
    assert_solve_failed(problem, weights)


def test_singular_covariance_raises():
    """A rank-deficient covariance matrix raises LinAlgError from the Cholesky factorization.

    This pins the current behavior so any future change (e.g. a regularized
    fallback) is a conscious decision.
    """
    n = 2
    model = SampleCovariance(num=n)
    with pytest.raises(np.linalg.LinAlgError):
        model.update(
            cov=np.ones((n, n)),  # rank 1
            lower_assets=np.zeros(n),
            upper_assets=np.ones(n),
        )


def test_update_validation_messages():
    """Invalid update() inputs must raise ValueError with actionable messages."""
    sample = SampleCovariance(num=3)
    with pytest.raises(ValueError, match="requires a 'cov'"):
        sample.update(lower_assets=np.zeros(3), upper_assets=np.ones(3))
    with pytest.raises(ValueError, match="square"):
        sample.update(cov=np.ones((2, 3)), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    with pytest.raises(ValueError, match="Too many assets"):
        sample.update(cov=np.eye(4), lower_assets=np.zeros(4), upper_assets=np.ones(4))
    with pytest.raises(ValueError, match="requires a 'lower_assets'"):
        sample.update(cov=np.eye(3), lower_asset=np.zeros(3), upper_assets=np.ones(3))  # typo

    factor = FactorModel(assets=3, k=2)
    with pytest.raises(ValueError, match="missing required arguments: cov"):
        factor.update(
            exposure=np.ones((2, 3)),
            idiosyncratic_risk=np.ones(3),
            lower_assets=np.zeros(3),
            upper_assets=np.ones(3),
            lower_factors=-np.ones(2),
            upper_factors=np.ones(2),
        )
    with pytest.raises(ValueError, match="cov must have shape"):
        factor.update(
            exposure=np.ones((2, 3)),
            cov=np.eye(3),  # should be 2x2
            idiosyncratic_risk=np.ones(3),
            lower_assets=np.zeros(3),
            upper_assets=np.ones(3),
            lower_factors=-np.ones(2),
            upper_factors=np.ones(2),
        )
    with pytest.raises(ValueError, match="idiosyncratic_risk must have shape"):
        factor.update(
            exposure=np.ones((2, 3)),
            cov=np.eye(2),
            idiosyncratic_risk=np.ones(2),  # should have length 3
            lower_assets=np.zeros(3),
            upper_assets=np.ones(3),
            lower_factors=-np.ones(2),
            upper_factors=np.ones(2),
        )

    cvar = CVar(alpha=0.9, n=10, m=3)
    with pytest.raises(ValueError, match="requires a 'returns'"):
        cvar.update(lower_assets=np.zeros(3), upper_assets=np.ones(3))
    with pytest.raises(ValueError, match="2d matrix"):
        cvar.update(returns=np.zeros((10,)), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    with pytest.raises(ValueError, match=r"expects n=10"):
        cvar.update(returns=np.zeros((5, 3)), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    with pytest.raises(ValueError, match="Too many assets"):
        cvar.update(returns=np.zeros((10, 4)), lower_assets=np.zeros(4), upper_assets=np.ones(4))

    # Bounds.update() length guard: an over-length bound array (length 4 > m=3)
    # reaches Bounds.update() via CVar.update() and must be rejected.
    with pytest.raises(ValueError, match="maximum is"):
        cvar.update(returns=np.zeros((10, 3)), lower_assets=np.zeros(4), upper_assets=np.ones(3))


def test_solve_minrisk_dimension_mismatch():
    """A weights variable that does not fit the model must raise a clear error."""
    sample = SampleCovariance(num=3)
    sample.update(cov=np.eye(3), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    problem = minrisk_problem(sample, Variable(2))
    with pytest.raises(ValueError, match="capacity is num=3"):
        problem.solve()

    factor = FactorModel(assets=3, k=2)
    factor.update(
        exposure=np.ones((2, 3)),
        cov=np.eye(2),
        idiosyncratic_risk=np.ones(3),
        lower_assets=np.zeros(3),
        upper_assets=np.ones(3),
        lower_factors=-np.ones(2),
        upper_factors=np.ones(2),
    )
    problem = minrisk_problem(factor, Variable(4))
    with pytest.raises(ValueError, match="capacity is assets=3"):
        problem.solve()

    cvar = CVar(alpha=0.9, n=10, m=3)
    cvar.update(returns=np.zeros((10, 3)), lower_assets=np.zeros(3), upper_assets=np.ones(3))
    problem = minrisk_problem(cvar, Variable(4))
    with pytest.raises(ValueError, match="capacity is m=3"):
        problem.solve()


def test_resolve_after_infeasible_recovers():
    """After an infeasible solve, fixing the bounds and re-solving must succeed."""
    n = 3
    model = SampleCovariance(num=n)
    model.update(cov=np.eye(n), lower_assets=np.zeros(n), upper_assets=np.full(n, 0.2))
    weights = Variable(n)
    problem = minrisk_problem(model, weights)
    problem.solve()
    assert "Solved" not in problem.status

    model.update(cov=np.eye(n), lower_assets=np.zeros(n), upper_assets=np.ones(n))
    problem.solve()
    assert "Solved" in problem.status
    assert np.isclose(np.sum(np.array(weights.value)), 1.0, atol=1e-6)
