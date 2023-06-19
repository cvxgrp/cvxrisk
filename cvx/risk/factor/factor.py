# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.model import RiskModel


@dataclass
class FactorModel(RiskModel):
    """Factor risk model"""

    assets: int = 0
    k: int = 0

    def __post_init__(self):
        self.exposure = cvx.Parameter(
            shape=(self.k, self.assets),
            name="exposure",
            value=np.zeros((self.k, self.assets)),
        )

        self.idiosyncratic_risk = cvx.Parameter(
            shape=(self.assets,), name="idiosyncratic risk", value=np.zeros(self.assets)
        )

        self.chol = cvx.Parameter(
            shape=(self.k, self.k),
            name="cholesky of covariance",
            value=np.zeros((self.k, self.k)),
        )

        self.factor_position = cvx.Parameter(
            shape=(self.k,), name="factor position", value=np.zeros(self.k)
        )

    def estimate_risk(self, weights, **kwargs):
        """
        Compute the total variance
        """
        var_residual = cvx.sum_squares(cvx.multiply(self.idiosyncratic_risk, weights))

        y = kwargs.get("y", self.exposure @ weights)

        return cvx.sum_squares(self.chol @ y) + var_residual
