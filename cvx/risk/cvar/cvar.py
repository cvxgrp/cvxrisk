# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.model import RiskModel


@dataclass
class CVar(RiskModel):
    """Conditional value at risk risk model"""

    alpha: float = 0.95
    R: np.ndarray = None

    def estimate_risk(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        # R is a matrix of returns, n is the number of rows in R
        n = self.R.shape[0]
        # k is the number of returns in the left tail
        k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        return -cvx.sum_smallest(self.R @ weights, k=k) / k
