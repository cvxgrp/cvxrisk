# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np

from cvx.risk.sample.sample import SampleCovariance
from cvx.risk.sample.sample import SampleCovarianceCholesky


def test_sample():
    riskmodel = SampleCovariance(np.array([[1.0, 0.5], [0.5, 2.0]]))
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)


def test_sample_cholesky():
    riskmodel = SampleCovarianceCholesky(np.array([[1.0, 0.5], [0.5, 2.0]]))
    var = riskmodel.estimate_risk(np.array([1.0, 1.0])).value
    np.testing.assert_almost_equal(var, 4.0)
