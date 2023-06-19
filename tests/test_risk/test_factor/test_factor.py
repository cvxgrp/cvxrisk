# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.risk.factor import FactorModel
from cvx.risk.linalg import cholesky
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

    model.exposure.value = factors.exposure
    model.chol.value = cholesky(factors.cov.values)
    model.idiosyncratic_risk.value = factors.idiosyncratic.std().values

    var = model.estimate_risk(weights).value
    np.testing.assert_almost_equal(var, 8.527444810470023e-05)
