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

    def __init__(self, num):
        self.cov = cvx.Parameter(
            shape=(num, num), name="covariance", PSD=True, value=np.identity(num)
        )

    # is not DPP
    def estimate_risk(self, weights, **kwargs):
        return cvx.quad_form(weights, self.cov)


class SampleCovariance_Product(SampleCovariance):
    """Risk model based on Cholesky decomposition of the sample cov matrix"""

    def estimate_risk(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        root = sc.linalg.cholesky(self.cov.value)
        return cvx.sum_squares(root @ weights)
