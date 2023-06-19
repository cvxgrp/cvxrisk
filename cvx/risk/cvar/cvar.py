# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.model import RiskModel


@dataclass
class CVar(RiskModel):
    """Conditional value at risk model"""

    alpha: float = 0.95
    n: int = 0
    m: int = 0

    def __post_init__(self):
        self.k = int(self.n * (1 - self.alpha))
        self.R = cvx.Parameter(
            shape=(self.n, self.m), name="returns", value=np.zeros((self.n, self.m))
        )
        self.lower = cvx.Parameter(
            shape=self.m,
            name="lower bound",
            value=np.zeros(self.m),
        )
        self.upper = cvx.Parameter(
            shape=self.m,
            name="upper bound",
            value=np.ones(self.m),
        )

    # R: np.ndarray = None

    def estimate_risk(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        # R is a matrix of returns, n is the number of rows in R
        # n = self.R.shape[0]
        # k is the number of returns in the left tail
        # k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        return -cvx.sum_smallest(self.R @ weights, k=self.k) / self.k

    def update_data(self, **kwargs):
        ret = kwargs["returns"]
        m = ret.shape[1]

        self.R.value[:, :m] = kwargs["returns"]

        self.lower.value = np.zeros(self.m)
        self.lower.value[:m] = kwargs["lower"]

        self.upper.value = np.zeros(self.m)
        self.upper.value[:m] = kwargs["upper"]
