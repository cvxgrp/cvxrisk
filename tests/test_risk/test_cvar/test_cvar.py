# -*- coding: utf-8 -*-
# Import necessary libraries
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.risk.cvar import CVar


def test_estimate_risk():
    """Test the estimate_risk() method"""
    model = CVar(alpha=0.95, n=50, m=10)

    np.random.seed(42)

    # define the problem
    weights = cp.Variable(10)
    risk = model.estimate_risk(weights)
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(weights) == 1, weights >= 0])
    assert prob.is_dpp()

    model.update_data(returns=np.random.randn(50, 10))
    prob.solve()
    assert prob.value == pytest.approx(0.5058720677762698)

    # it's enough to only update the R value...
    model.update_data(returns=np.random.randn(50, 10))
    prob.solve()
    assert prob.value == pytest.approx(0.43559171295408616)


def test_minvar():
    weights = cp.Variable(10)
    model = CVar(alpha=0.95, n=50, m=10)

    problem = cp.Problem(
        cp.Minimize(model.estimate_risk(weights)),
        [cp.sum(weights) == 1.0, weights >= 0],
    )

    assert problem.is_dpp()
