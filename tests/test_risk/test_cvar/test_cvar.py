# -*- coding: utf-8 -*-
# Import necessary libraries
from __future__ import annotations

import cvxpy as cvx
import numpy as np
import pytest

from cvx.risk.cvar import CVar


def test_estimate_risk():
    """Test the estimate_risk() method"""
    model = CVar(alpha=0.95)
    np.random.seed(42)
    model.R = np.random.randn(50, 10)
    weights = cvx.Variable(10)
    risk = model.estimate_risk(weights)
    prob = cvx.Problem(cvx.Minimize(risk), [cvx.sum(weights) == 1, weights >= 0])
    prob.solve()
    assert prob.value == pytest.approx(0.5058720677762698)
