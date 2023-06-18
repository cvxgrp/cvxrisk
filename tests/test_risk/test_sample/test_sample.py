# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.risk.linalg import cholesky
from cvx.risk.sample import SampleCovariance_Product

# def test_sample():
#    riskmodel = SampleCovariance(num=2)
#    riskmodel.cov.value = np.array([[1.0, 0.5], [0.5, 2.0]])
#    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
#    np.testing.assert_almost_equal(var, 4.0)


# def test_sample_not_psd():
#    riskmodel = SampleCovariance(num=2)
#    with pytest.raises(ValueError):
#        riskmodel.cov.value = np.array([[-1.0, 0.5], [0.5, -2.0]])
# from cvx.risk.sample.sample import cholesky


def test_sample_product():
    riskmodel = SampleCovariance_Product(num=2)
    riskmodel.chol.value = cholesky(np.array([[1.0, 0.5], [0.5, 2.0]]))
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_min_variance():
    weights = cp.Variable(2)
    riskmodel = SampleCovariance_Product(num=2)

    #
    problem = cp.Problem(
        cp.Minimize(riskmodel.estimate_risk(weights)),
        [cp.sum(weights) == 1.0, weights >= 0],
    )

    riskmodel.chol.value = cholesky(np.array([[1.0, 0.5], [0.5, 2.0]]))
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.75, 0.25]))

    # It's enough to only update the value for the cholesky decomposition
    riskmodel.chol.value = cholesky(np.array([[1.0, 0.5], [0.5, 4.0]]))
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.875, 0.125]))


# def test_optimize_update():
#     weights = cp.Variable(2)
#     riskmodel = SampleCovariance(num=2)
#
#     problem = cp.Problem(
#         cp.Minimize(riskmodel.estimate_risk(weights)),
#         [cp.sum(weights) == 1.0, weights >= 0],
#     )
#     assert problem.is_dpp()
#
#     riskmodel.cov.value = np.array([[1.0, 0.5], [0.5, 2.0]])
#     problem.solve()
#     print(weights.value)
#
#     # this is updating the problem!
#     riskmodel.cov.value = np.array([[1.0, 0.5], [0.5, 4.0]])
#     problem.solve()
#     print(weights.value)
