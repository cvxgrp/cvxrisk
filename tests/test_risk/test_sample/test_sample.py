from __future__ import annotations

import cvxpy as cp
import numpy as np

from cvx.portfolio.min_risk import minrisk_problem
from cvx.risk.sample import SampleCovariance


def test_sample():
    """
    Test the SampleCovariance class with a small covariance matrix.

    This test verifies that:
    1. A SampleCovariance model can be initialized with specified dimensions
    2. The model can be updated with a covariance matrix and bounds
    3. The estimate method calculates the correct portfolio volatility
    """
    riskmodel = SampleCovariance(num=2)
    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    vola = riskmodel.estimate(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_sample_large():
    """
    Test the SampleCovariance class with a larger covariance matrix.

    This test verifies that:
    1. A SampleCovariance model can be initialized with dimensions larger than the data
    2. The model can be updated with a smaller covariance matrix
    3. The estimate method correctly handles portfolios with zero weights
    4. The calculated volatility is correct
    """
    riskmodel = SampleCovariance(num=4)
    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    vola = riskmodel.estimate(np.array([1.0, 1.0, 0.0, 0.0])).value
    np.testing.assert_almost_equal(vola, 2.0)


def test_min_variance():
    """
    Test the minimum variance problem with a sample covariance model.

    This test verifies that:
    1. A minimum risk problem can be created with a SampleCovariance model
    2. The problem is disciplined parametrized programming (DPP) compliant
    3. The problem can be solved and produces the expected optimal weights
    4. The model can be updated with a different covariance matrix
    5. The problem can be re-solved and produces new optimal weights
    6. Unused assets have zero weights in the optimal solution
    """
    weights = cp.Variable(4)
    riskmodel = SampleCovariance(num=4)
    problem = minrisk_problem(riskmodel, weights)
    assert problem.is_dpp()

    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.75, 0.25, 0.0, 0.0]), decimal=5)

    # It's enough to only update the value for the cholesky decomposition
    riskmodel.update(
        cov=np.array([[1.0, 0.5], [0.5, 4.0]]),
        lower_assets=np.zeros(2),
        upper_assets=np.ones(2),
    )
    problem.solve()
    np.testing.assert_almost_equal(weights.value, np.array([0.875, 0.125, 0.0, 0.0]), decimal=5)
