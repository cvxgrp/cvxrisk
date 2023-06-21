# -*- coding: utf-8 -*-
# Import necessary libraries
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.risk.cvar import CVar
from tests.test_risk.minvar import minvar_problem


def test_estimate_risk():
    """Test the estimate() method"""
    model = CVar(alpha=0.95, n=50, m=14)

    np.random.seed(42)

    # define the problem
    weights = cp.Variable(14)
    prob = minvar_problem(model, weights)
    assert prob.is_dpp()

    model.update(returns=np.random.randn(50, 10), lower=np.zeros(10), upper=np.ones(10))
    prob.solve()
    assert prob.value == pytest.approx(0.5058720677762698)

    # it's enough to only update the R value...
    model.update(returns=np.random.randn(50, 10), lower=np.zeros(10), upper=np.ones(10))
    prob.solve()
    assert prob.value == pytest.approx(0.43559171295408616)
