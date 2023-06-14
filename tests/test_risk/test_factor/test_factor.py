# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.risk.factor import FundamentalFactorRiskModel
from cvx.risk.factor import TimeseriesFactorRiskModel
from cvx.risk.factor.linalg.pca import pca as principal_components


@pytest.fixture()
def returns(resource_dir):
    prices = pd.read_csv(
        resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True
    )
    return prices.pct_change().fillna(0.0)


def test_timeseries_model(returns):
    weights = pd.Series(index=returns.columns, data=0.05).values

    model = TimeseriesFactorRiskModel(
        returns=returns,
        factors=principal_components(returns=returns, n_components=10).returns,
    )
    var = model.estimate_risk(weights).value
    np.testing.assert_almost_equal(var, 8.527444810470023e-05)


def test_fundamental_model(returns):
    model = TimeseriesFactorRiskModel(
        returns=returns,
        factors=principal_components(returns=returns, n_components=10).returns,
    )

    model = FundamentalFactorRiskModel(
        factor_covariance=model.factors.cov(),
        exposure=model.exposure,
        idiosyncratic_risk=model.idiosyncratic_returns.std(),
    )

    weights = pd.Series(index=returns.columns, data=0.05).values
    var = model.estimate_risk(weights).value
    np.testing.assert_almost_equal(var, 8.527444810470023e-05)


def test_with_covariance(returns):
    factors = principal_components(returns=returns, n_components=10).returns
    cov = factors.cov()
    weights = pd.Series(index=returns.columns, data=0.05).values

    model = TimeseriesFactorRiskModel(cov=cov, factors=factors, returns=returns)
    var = model.estimate_risk(weights).value
    np.testing.assert_almost_equal(var, 8.527444810470023e-05)
