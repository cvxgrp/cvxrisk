"""Tests for the Conditional Value at Risk (CVaR) implementation."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.core.variable import Variable
from cvx.risk.cvar import CVar
from cvx.risk.portfolio import minrisk_problem


def test_extra_constraints() -> None:
    """Extra constraints (equality, lb-only, ub-only) are passed through solve_minrisk."""
    model = CVar(alpha=0.95, n=20, m=3)
    rng = np.random.default_rng(0)
    weights = Variable(3)

    constraints = [
        (np.array([1.0, 0.0, 0.0]), 0.4, 0.4),  # equality: w[0] == 0.4
        (np.array([0.0, 1.0, 0.0]), 0.1, None),  # lb-only: w[1] >= 0.1
        (np.array([0.0, 0.0, 1.0]), None, 0.7),  # ub-only: w[2] <= 0.7
    ]

    problem = minrisk_problem(model, weights, constraints=constraints)
    model.update(
        returns=rng.standard_normal((20, 3)),
        lower_assets=np.zeros(3),
        upper_assets=np.ones(3),
    )
    problem.solve()
    assert "Solved" in problem.status
    assert np.isclose(weights.value[0], 0.4, atol=1e-4)
    assert weights.value[1] >= 0.1 - 1e-4
    assert weights.value[2] <= 0.7 + 1e-4


def test_infeasible() -> None:
    """Infeasible bounds cause solve_minrisk to return None without raising."""
    model = CVar(alpha=0.95, n=20, m=2)
    rng = np.random.default_rng(0)
    weights = Variable(2)
    problem = minrisk_problem(model, weights)
    # lower bounds sum > 1 contradicts the sum=1 equality → infeasible
    model.update(
        returns=rng.standard_normal((20, 2)),
        lower_assets=np.array([0.7, 0.7]),
        upper_assets=np.ones(2),
    )
    problem.solve()
    assert problem.value is None
    assert "Solved" not in problem.status


def test_estimate_risk() -> None:
    """Test the estimate() method of the CVar class.

    This test verifies that:
    1. The CVar model can be initialized with specified parameters
    2. A minimum risk problem using the CVar model can be created
    3. The model can be updated with new returns data
    4. The problem can be solved and produces the expected optimal value
    5. The model can be updated again and the problem can be re-solved
    """
    model = CVar(alpha=0.95, n=50, m=14)

    rng = np.random.default_rng(42)

    # define the problem
    weights = Variable(14)
    prob = minrisk_problem(model, weights)

    model.update(
        returns=rng.standard_normal((50, 10)),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    prob.solve()
    # rel=1e-4 reflects Clarabel's default solver tolerances when called directly
    assert prob.value == pytest.approx(0.37293694583777964, rel=1e-4)

    # it's enough to only update the R value...
    model.update(
        returns=rng.standard_normal((50, 10)),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    prob.solve()
    assert prob.value == pytest.approx(0.40960097904559756, rel=1e-4)
