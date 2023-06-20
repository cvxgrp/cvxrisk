# -*- coding: utf-8 -*-
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.linalg import cholesky
from cvx.risk.model import RiskModel
from cvx.risk.random import rand_cov


@dataclass
class SampleCovariance(RiskModel):
    """Risk model based on the Cholesky decomposition of the sample cov matrix"""

    num: int = 0

    def __post_init__(self):
        self._chol = cvx.Parameter(
            shape=(self.num, self.num),
            name="cholesky of covariance",
            value=np.zeros((self.num, self.num)),
        )
        self.lower = cvx.Parameter(
            shape=self.num,
            name="lower bound",
            value=np.zeros(self.num),
        )
        self.upper = cvx.Parameter(
            shape=self.num,
            name="upper bound",
            value=np.ones(self.num),
        )

    def estimate_risk(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        return cvx.norm2(self._chol @ weights)

    def update_data(self, **kwargs):
        cov = kwargs["cov"]
        n = cov.shape[0]

        self._chol.value[:n, :n] = cholesky(cov)
        self.lower.value = np.zeros(self.num)
        self.lower.value[:n] = kwargs["lower"]

        self.upper.value = np.zeros(self.num)
        self.upper.value[:n] = kwargs["upper"]


if __name__ == "__main__":
    s = SampleCovariance(num=5)
    cov = rand_cov(4)
    s.update_data(cov=cov, lower=np.zeros(4), upper=np.ones(4))

    print(s.lower.value)
    print(s.upper.value)

    # z = np.zeros((5,5))
    # z[:4,:4] = cholesky(cov)
    # print(z
