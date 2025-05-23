"""Tests for the Conditional Value at Risk (CVaR) implementation"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.portfolio import minrisk_problem
from cvx.risk.cvar import CVar


def test_estimate_risk() -> None:
    """
    Test the estimate() method of the CVar class.

    This test verifies that:
    1. The CVar model can be initialized with specified parameters
    2. A minimum risk problem using the CVar model can be created and is DPP
    3. The model can be updated with new returns data
    4. The problem can be solved and produces the expected optimal value
    5. The model can be updated again and the problem can be re-solved
    """
    model = CVar(alpha=0.95, n=50, m=14)

    np.random.seed(42)

    # define the problem
    weights = cp.Variable(14)
    prob = minrisk_problem(model, weights)
    assert prob.is_dpp()

    model.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    prob.solve(solver="CLARABEL")
    assert prob.value == pytest.approx(0.5058720677762698)

    # it's enough to only update the R value...
    model.update(
        returns=np.random.randn(50, 10),
        lower_assets=np.zeros(10),
        upper_assets=np.ones(10),
    )
    prob.solve(solver="CLARABEL")
    assert prob.value == pytest.approx(0.43559171295408616)
