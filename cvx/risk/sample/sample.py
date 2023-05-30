# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

import cvxpy as cvx
import numpy as np
import scipy as sc

from cvx.risk.model import RiskModel


class SampleCovariance(RiskModel):
    """Sample covariance model"""

    def __init__(self, cov):
        num = cov.shape[0]
        self.cov = cvx.Parameter(
            shape=(num, num), name="covariance", PSD=True, value=np.identity(num)
        )
        self.cov.value = cov

    def estimate_risk(self, weights):
        return cvx.quad_form(weights, self.cov)


class SampleCovarianceCholesky(RiskModel):
    """Risk model based on Cholesky decomposition of the sample cov matrix"""

    def __init__(self, cov):
        num = cov.shape[0]
        self.__root = cvx.Parameter(shape=(num, num))
        self.__root.value = sc.linalg.sqrtm(cov)

    def estimate_risk(self, weights):
        return cvx.sum_squares(self.__root @ weights)
