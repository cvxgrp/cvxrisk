# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.bounds import Bounds
from cvx.risk.model import Model


@dataclass
class CVar(Model):
    """Conditional value at risk model"""

    alpha: float = 0.95
    n: int = 0
    m: int = 0

    def __post_init__(self):
        self.k = int(self.n * (1 - self.alpha))
        self.parameter["R"] = cvx.Parameter(
            shape=(self.n, self.m), name="returns", value=np.zeros((self.n, self.m))
        )
        self.bounds = Bounds(m=self.m, name="assets")

    def estimate(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        # R is a matrix of returns, n is the number of rows in R
        # n = self.R.shape[0]
        # k is the number of returns in the left tail
        # k = int(n * (1 - self.alpha))
        # average value of the k elements in the left tail
        return -cvx.sum_smallest(self.parameter["R"] @ weights, k=self.k) / self.k

    def update(self, **kwargs):
        ret = kwargs["returns"]
        m = ret.shape[1]

        self.parameter["R"].value[:, :m] = kwargs["returns"]
        self.bounds.update(**kwargs)

    def constraints(self, weights, **kwargs):
        return self.bounds.constraints(weights)
