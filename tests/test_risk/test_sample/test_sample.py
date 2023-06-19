# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np

from cvx.risk.sample import SampleCovariance


def test_sample_product():
    riskmodel = SampleCovariance(num=2)
    riskmodel.update_data(cov=np.array([[1.0, 0.5], [0.5, 2.0]]))
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_min_variance():
    weights = cp.Variable(2)
    riskmodel = SampleCovariance(num=2)

    #
    problem = cp.Problem(
        cp.Minimize(riskmodel.estimate_risk(weights)),
        [cp.sum(weights) == 1.0, weights >= 0],
    )

    riskmodel.update_data(cov=np.array([[1.0, 0.5], [0.5, 2.0]]))
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.75, 0.25]))

    # It's enough to only update the value for the cholesky decomposition
    riskmodel.update_data(cov=np.array([[1.0, 0.5], [0.5, 4.0]]))
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.875, 0.125]))
