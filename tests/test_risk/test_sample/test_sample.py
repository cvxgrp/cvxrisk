# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from cvx.risk.factor import FundamentalFactorRiskModel
from cvx.risk.sample import SampleCovariance
from cvx.risk.sample import SampleCovariance_Product


def test_sample():
    riskmodel = SampleCovariance(num=2)
    riskmodel.cov.value = np.array([[1.0, 0.5], [0.5, 2.0]])
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_sample_product():
    riskmodel = SampleCovariance_Product(num=2)
    riskmodel.cov.value = np.array([[1.0, 0.5], [0.5, 2.0]])
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_fundamental_factor_risk_model():
    riskmodel = FundamentalFactorRiskModel()

    riskmodel.factor_covariance = pd.DataFrame(data=np.array([[1.0, 0.5], [0.5, 2.0]]))
    riskmodel.exposure = pd.DataFrame(data=np.eye(2))
    riskmodel.idiosyncratic_risk = pd.Series(data=[0.0, 0.0])

    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)
