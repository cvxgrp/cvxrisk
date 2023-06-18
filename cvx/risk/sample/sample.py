# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.model import RiskModel


# @dataclass
# class SampleCovariance(RiskModel):
#     """Sample covariance model"""
#
#     num: int = 0
#
#     def __post_init__(self):
#         self.cov = cvx.Parameter(
#             shape=(self.num, self.num),
#             name="covariance",
#             PSD=True,
#             value=np.identity(self.num),
#         )
#
#     # is not DPP
#     def estimate_risk(self, weights, **kwargs):
#         return cvx.quad_form(weights, self.cov)


@dataclass
class SampleCovariance_Product(RiskModel):
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
