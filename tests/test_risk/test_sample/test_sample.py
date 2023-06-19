# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np

from cvx.risk.sample import SampleCovariance


def test_sample():
    riskmodel = SampleCovariance(num=2)
    riskmodel.update_data(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]), lower=np.zeros(2), upper=np.ones(2)
    )
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_sample_large():
    riskmodel = SampleCovariance(num=4)
    riskmodel.update_data(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]), lower=np.zeros(2), upper=np.ones(2)
    )
    var = riskmodel.estimate_risk(np.array([1.0, 1.0, 0.0, 0.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_min_variance():
    weights = cp.Variable(4)
    riskmodel = SampleCovariance(num=4)

    #
    problem = cp.Problem(
        cp.Minimize(riskmodel.estimate_risk(weights)),
        [
            cp.sum(weights) == 1.0,
            weights >= 0,
            riskmodel.lower <= weights,
            weights <= riskmodel.upper,
        ],
    )

    riskmodel.update_data(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]), lower=np.zeros(2), upper=np.ones(2)
    )
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.75, 0.25, 0.0, 0.0]))

    # It's enough to only update the value for the cholesky decomposition
    riskmodel.update_data(
        cov=np.array([[1.0, 0.5], [0.5, 4.0]]), lower=np.zeros(2), upper=np.ones(2)
    )
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.875, 0.125, 0.0, 0.0]))
