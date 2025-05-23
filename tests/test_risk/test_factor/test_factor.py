"""Tests for the factor risk model implementation"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from cvx.portfolio.min_risk import minrisk_problem
from cvx.random import rand_cov
from cvx.risk.factor import FactorModel
from cvx.risk.linalg import pca as principal_components


@pytest.fixture()
def returns(resource_dir) -> pd.DataFrame:
    """
    Pytest fixture that provides stock return data for testing.

    This fixture loads stock price data from a CSV file, calculates returns
    using percentage change, and fills any NaN values with zeros.

    Args:

        resource_dir: Pytest fixture providing the path to the test resources directory

    Returns:

        pandas.DataFrame: DataFrame containing stock returns
    """
    prices = pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True)
    return prices.pct_change().fillna(0.0)


def test_timeseries_model(returns: pd.DataFrame) -> None:
    """
    Test the FactorModel with time series data.

    This test verifies that:
    1. Principal components can be computed from returns data
    2. A FactorModel can be initialized and updated with the PCA results
    3. The model's estimate method calculates the expected volatility for a given portfolio

    Args:

        returns: Pytest fixture providing stock return data
    """
    # Here we compute the factors and regress the returns on them
    factors = principal_components(returns=returns, n_components=10)

    model = FactorModel(assets=25, k=10)

    model.update(
        cov=factors.cov.values,
        exposure=factors.exposure.values,
        idiosyncratic_risk=factors.idiosyncratic.std().values,
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )

    w = np.zeros(25)
    w[:20] = 0.05

    vola = model.estimate(w).value
    np.testing.assert_almost_equal(vola, 0.00923407730537884)


def test_minvar(returns: pd.DataFrame) -> None:
    """
    Test the minimum variance problem with a factor model.

    This test verifies that:
    1. A minimum risk problem can be created with a FactorModel
    2. The problem is disciplined parametrized programming (DPP) compliant

    Args:

        returns: Pytest fixture providing stock return data
    """
    weights = cp.Variable(20)
    y = cp.Variable(10)

    model = FactorModel(assets=20, k=10)

    problem = minrisk_problem(model, weights, y=y)

    assert problem.is_dpp()


def test_estimate_risk() -> None:
    """Test the estimate() method"""
    model = FactorModel(assets=25, k=12)

    np.random.seed(42)

    # define the problem
    weights = cp.Variable(25)
    y = cp.Variable(12)

    prob = minrisk_problem(model, weights, y=y)
    assert prob.is_dpp()

    model.update(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )
    prob.solve(solver="CLARABEL")
    assert prob.value == pytest.approx(0.14138117837204583)
    assert np.array(weights.value[20:]) == pytest.approx(np.zeros(5), abs=1e-6)

    model.update(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=-0.1 * np.ones(10),
        upper_factors=0.1 * np.ones(10),
    )
    prob.solve(solver="CLARABEL")
    assert prob.value == pytest.approx(0.5454593844618784)
    assert np.array(weights.value[20:]) == pytest.approx(np.zeros(5), abs=1e-6)

    # test that the exposure is correct, e.g. the factor weights match the exposure * asset weights
    assert model.parameter["exposure"].value @ weights.value == pytest.approx(y.value, abs=1e-6)

    # test all entries of y are smaller than 0.1
    assert np.all([y.value <= 0.1 + 1e-6])
    # test all entries of y are larger than -0.1
    assert np.all([y.value >= -(0.1 + 1e-6)])


def test_dynamic_exposure() -> None:
    """
    Test the dynamic exposure update functionality of the FactorModel.

    This test verifies that:
    1. The FactorModel can be updated with different exposure matrices
    2. The exposure parameter is correctly updated with the new values
    3. The shape of the exposure parameter is maintained
    """
    model = FactorModel(assets=3, k=2)
    model.update(
        exposure=np.array([[1.0, 2.0]]),
        idiosyncratic_risk=np.array([1.0, 1.0]),
        cov=np.array([[1.0]]),
        lower_assets=np.array([0.0]),
        upper_assets=np.array([1.0]),
        lower_factors=np.array([0.0]),
        upper_factors=np.array([1.0]),
    )

    model.update(
        exposure=np.array([[1.0]]),
        idiosyncratic_risk=np.array([1.0]),
        cov=np.array([[1.0]]),
        lower_assets=np.array([0.0]),
        upper_assets=np.array([1.0]),
        lower_factors=np.array([0.0]),
        upper_factors=np.array([1.0]),
    )

    np.testing.assert_array_equal(model.parameter["exposure"].value, np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
