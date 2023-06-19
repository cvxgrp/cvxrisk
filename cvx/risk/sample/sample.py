# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.linalg import cholesky
from cvx.risk.model import RiskModel


@dataclass
class SampleCovariance(RiskModel):
    """Risk model based on Cholesky decomposition of the sample cov matrix"""

    num: int = 0

    def __post_init__(self):
        self.chol = cvx.Parameter(
            shape=(self.num, self.num),
            name="cholesky of covariance",
            value=np.identity(self.num),
        )

    def estimate_risk(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        return cvx.sum_squares(self.chol @ weights)

    def update_data(self, **kwargs):
        self.chol.value = cholesky(kwargs["cov"])
