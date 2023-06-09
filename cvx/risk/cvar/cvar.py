# -*- coding: utf-8 -*-
from __future__ import annotations

import cvxpy as cvx

from cvx.risk.model import RiskModel


class CVar(RiskModel):
    """Conditional value at risk risk model"""

    def __init__(self, alpha=0.95):
        self.alpha = alpha

    def estimate_risk(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        R = kwargs["R"]
        # R is a matrix of returns, n is the number of rows in R
        n = R.shape[0]
        # k is the number of returns in the left tail
        k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        return -cvx.sum_smallest(R @ weights, k=k) / k
