# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from cvx.risk.factor import FactorModel
from cvx.risk.linalg import pca as principal_components


@pytest.fixture()
def returns(resource_dir):
    prices = pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    return prices.pct_change().fillna(0.0)


def test_timeseries_model(returns):
    weights = pd.Series(index=returns.columns, data=0.05).values

    # Here we compute the factors and regress the returns on them
    factors = principal_components(returns=returns, n_components=10)

    model = FactorModel(assets=20, k=10)

    model.update_data(
        cov=factors.cov.values,
        exposure=factors.exposure.values,
        idiosyncratic_risk=factors.idiosyncratic.std().values,
    )

    var = model.estimate_risk(weights).value
    np.testing.assert_almost_equal(var, 8.527444810470023e-05)


def test_minvar(returns):
    weights = cp.Variable(20)
    y = cp.Variable(10)

    model = FactorModel(assets=20, k=10)

    problem = cp.Problem(
        cp.Minimize(model.estimate_risk(weights, y=y)),
        [cp.sum(weights) == 1.0, weights >= 0, y == model.exposure @ weights],
    )

    assert problem.is_dpp()
