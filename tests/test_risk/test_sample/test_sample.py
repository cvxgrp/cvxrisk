# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd

from cvx.risk.factor.fundamental import FundamentalFactorRiskModel
from cvx.risk.factor.fundamental import FundamentalFactorRiskModel_Product
from cvx.risk.sample.sample import SampleCovariance
from cvx.risk.sample.sample import SampleCovarianceCholesky


def test_sample():
    riskmodel = SampleCovariance(num=2)
    riskmodel.cov.value = np.array([[1.0, 0.5], [0.5, 2.0]])
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_sample_cholesky():
    riskmodel = SampleCovarianceCholesky(num=2)
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


def test_fundamental_factor_risk_model_product():
    riskmodel = FundamentalFactorRiskModel_Product()

    riskmodel.factor_covariance = pd.DataFrame(data=np.array([[1.0, 0.5], [0.5, 2.0]]))
    riskmodel.exposure = pd.DataFrame(data=np.eye(2))
    riskmodel.idiosyncratic_risk = pd.Series(data=[0.0, 0.0])

    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)
