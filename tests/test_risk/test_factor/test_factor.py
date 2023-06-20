# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from cvx.risk.factor import FactorModel
from cvx.risk.linalg import pca as principal_components
from cvx.risk.random import rand_cov


@pytest.fixture()
def returns(resource_dir):
    prices = pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    return prices.pct_change().fillna(0.0)


def test_timeseries_model(returns):
    # Here we compute the factors and regress the returns on them
    factors = principal_components(returns=returns, n_components=10)

    model = FactorModel(assets=25, k=10)

    model.update_data(
        cov=factors.cov.values,
        exposure=factors.exposure.values,
        idiosyncratic_risk=factors.idiosyncratic.std().values,
        lower=np.zeros(20),
        upper=np.ones(20),
    )

    w = np.zeros(25)
    w[:20] = 0.05

    vola = model.estimate_risk(w).value
    np.testing.assert_almost_equal(vola, 0.00923407730537884)


def test_minvar(returns):
    weights = cp.Variable(20)
    y = cp.Variable(10)

    model = FactorModel(assets=20, k=10)

    problem = cp.Problem(
        cp.Minimize(model.estimate_risk(weights, y=y)),
        [cp.sum(weights) == 1.0, weights >= 0, y == model.exposure @ weights],
    )

    assert problem.is_dpp()


def test_estimate_risk():
    """Test the estimate_risk() method"""
    model = FactorModel(assets=25, k=12)

    np.random.seed(42)

    # define the problem
    weights = cp.Variable(25)
    y = cp.Variable(12)

    risk = model.estimate_risk(weights, y=y)

    prob = cp.Problem(
        cp.Minimize(risk),
        [
            cp.sum(weights) == 1,
            weights >= 0,
            y == model.exposure @ weights,
            model.lower <= weights,
            weights <= model.upper,
        ],
    )
    assert prob.is_dpp()

    model.update_data(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        lower=np.zeros(20),
        upper=np.ones(20),
    )
    prob.solve()
    assert prob.value == pytest.approx(0.13625197847921858)
    assert np.array(weights.value[20:]) == pytest.approx(np.zeros(5), abs=1e-6)

    model.update_data(
        cov=rand_cov(10),
        exposure=np.random.randn(10, 20),
        idiosyncratic_risk=np.random.randn(20),
        lower=np.zeros(20),
        upper=np.ones(20),
    )
    prob.solve()
    assert prob.value == pytest.approx(0.40835167515605786)
    assert np.array(weights.value[20:]) == pytest.approx(np.zeros(5), abs=1e-6)
