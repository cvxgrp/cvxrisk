"""Tests for the factor risk model implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
import pytest
from cvx.linalg import pca as principal_components
from cvx.linalg import rand_cov

from cvx.core.variable import Variable
from cvx.risk.factor import FactorModel
from cvx.risk.portfolio import minrisk_problem


@pytest.fixture
def returns(resource_dir) -> pl.DataFrame:
    """Pytest fixture that provides stock return data for testing.

    This fixture loads stock price data from a CSV file, calculates returns
    using percentage change, and fills any NaN values with zeros.

    Args:
        resource_dir: Pytest fixture providing the path to the test resources directory

    Returns:
        pandas.DataFrame: DataFrame containing stock returns

    """
    prices = pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True)
    return pl.from_pandas(prices.pct_change().fillna(0.0).reset_index(drop=True))


def test_timeseries_model(returns: pl.DataFrame) -> None:
    """Test the FactorModel with time series data.

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
        cov=factors.cov.to_numpy(),
        exposure=factors.exposure.to_numpy(),
        idiosyncratic_risk=factors.idiosyncratic.std().to_numpy().ravel(),
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )

    w = np.zeros(25)
    w[:20] = 0.05

    vola = model.estimate(w)
    np.testing.assert_almost_equal(vola, 0.00923407730537884)


def test_minvar(returns: pl.DataFrame) -> None:
    """Test the minimum variance problem with a factor model.

    This test verifies that a minimum risk problem can be created with a FactorModel.

    Args:
        returns: Pytest fixture providing stock return data

    """
    weights = Variable(20)
    y = Variable(10)

    model = FactorModel(assets=20, k=10)

    problem = minrisk_problem(model, weights, y=y)

    assert problem is not None


def test_estimate_risk() -> None:
    """Test the estimate() method."""
    model = FactorModel(assets=25, k=12)

    rng = np.random.default_rng(42)

    # define the problem
    weights = Variable(25)
    y = Variable(12)

    prob = minrisk_problem(model, weights, y=y)

    model.update(
        cov=rand_cov(10, seed=42),
        exposure=rng.standard_normal((10, 20)),
        idiosyncratic_risk=rng.standard_normal(20),
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=np.zeros(10),
        upper_factors=np.ones(10),
    )
    prob.solve()
    w = np.array(weights.value)

    # rel=1e-4 reflects Clarabel's default solver tolerances when called directly
    assert prob.value == pytest.approx(0.19926997253968454, rel=1e-4)
    assert w[20:] == pytest.approx(np.zeros(5), abs=1e-6)

    model.update(
        cov=rand_cov(10, seed=42),
        exposure=rng.standard_normal((10, 20)),
        idiosyncratic_risk=rng.standard_normal(20),
        lower_assets=np.zeros(20),
        upper_assets=np.ones(20),
        lower_factors=-0.1 * np.ones(10),
        upper_factors=0.1 * np.ones(10),
    )
    prob.solve()
    w = np.array(weights.value)
    # rel=1e-4 reflects Clarabel's default solver tolerances when called directly
    assert prob.value == pytest.approx(0.18811759576078277, rel=1e-4)
    assert w[20:] == pytest.approx(np.zeros(5), abs=1e-6)

    # test that the exposure is correct, e.g. the factor weights match the exposure * asset weights
    assert model.parameter["exposure"].value @ weights.value == pytest.approx(y.value, abs=1e-6)

    # test all entries of y are smaller than 0.1
    assert np.all([y.value <= 0.1 + 1e-6])
    # test all entries of y are larger than -0.1
    assert np.all([y.value >= -(0.1 + 1e-6)])


def test_too_many_factors() -> None:
    """update() raises ValueError when exposure has more factors than k."""
    model = FactorModel(assets=5, k=2)
    with pytest.raises(ValueError, match="Too many factors"):
        model.update(
            exposure=np.ones((3, 3)),
            idiosyncratic_risk=np.ones(3),
            cov=np.eye(3),
            lower_assets=np.zeros(3),
            upper_assets=np.ones(3),
            lower_factors=np.zeros(3),
            upper_factors=np.ones(3),
        )


def test_too_many_assets() -> None:
    """update() raises ValueError when exposure has more assets than self.assets."""
    model = FactorModel(assets=3, k=2)
    with pytest.raises(ValueError, match="Too many assets"):
        model.update(
            exposure=np.ones((2, 4)),
            idiosyncratic_risk=np.ones(4),
            cov=np.eye(2),
            lower_assets=np.zeros(4),
            upper_assets=np.ones(4),
            lower_factors=np.zeros(2),
            upper_factors=np.ones(2),
        )


def test_extra_constraints() -> None:
    """Extra constraints (equality, lb-only, ub-only) are passed through solve_minrisk."""
    model = FactorModel(assets=4, k=2)
    weights = Variable(4)
    y = Variable(2)

    constraints = [
        (np.array([1.0, 0.0, 0.0, 0.0]), 0.3, 0.3),  # equality: w[0] == 0.3
        (np.array([0.0, 1.0, 0.0, 0.0]), 0.1, None),  # lb-only: w[1] >= 0.1
        (np.array([0.0, 0.0, 1.0, 0.0]), None, 0.6),  # ub-only: w[2] <= 0.6
    ]

    problem = minrisk_problem(model, weights, y=y, constraints=constraints)

    rng = np.random.default_rng(42)
    model.update(
        cov=rand_cov(2, seed=42),
        exposure=rng.standard_normal((2, 4)),
        idiosyncratic_risk=np.abs(rng.standard_normal(4)),
        lower_assets=np.zeros(4),
        upper_assets=np.ones(4),
        lower_factors=-np.ones(2),
        upper_factors=np.ones(2),
    )
    problem.solve()
    assert "Solved" in problem.status
    assert np.isclose(weights.value[0], 0.3, atol=1e-4)
    assert weights.value[1] >= 0.1 - 1e-4
    assert weights.value[2] <= 0.6 + 1e-4


def test_infeasible() -> None:
    """Infeasible bounds cause solve_minrisk to return None without raising."""
    model = FactorModel(assets=2, k=1)
    weights = Variable(2)
    y = Variable(1)
    problem = minrisk_problem(model, weights, y=y)
    # lower bounds sum > 1 contradicts the sum=1 equality → infeasible
    model.update(
        cov=np.array([[1.0]]),
        exposure=np.ones((1, 2)),
        idiosyncratic_risk=np.ones(2),
        lower_assets=np.array([0.7, 0.7]),
        upper_assets=np.ones(2),
        lower_factors=np.zeros(1),
        upper_factors=np.ones(1),
    )
    problem.solve()
    assert problem.value is None
    assert "Solved" not in problem.status


def test_dynamic_exposure() -> None:
    """Test the dynamic exposure update functionality of the FactorModel.

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
