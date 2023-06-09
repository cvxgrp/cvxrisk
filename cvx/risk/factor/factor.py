# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.linalg import cholesky
from cvx.risk.bounds import Bounds
from cvx.risk.model import Model


@dataclass
class FactorModel(Model):
    """Factor risk model"""

    assets: int = 0
    k: int = 0

    def __post_init__(self):
        self.parameter["exposure"] = cvx.Parameter(
            shape=(self.k, self.assets),
            name="exposure",
            value=np.zeros((self.k, self.assets)),
        )

        self.parameter["idiosyncratic_risk"] = cvx.Parameter(
            shape=self.assets, name="idiosyncratic risk", value=np.zeros(self.assets)
        )

        self.parameter["chol"] = cvx.Parameter(
            shape=(self.k, self.k),
            name="cholesky of covariance",
            value=np.zeros((self.k, self.k)),
        )

        self.bounds_assets = Bounds(m=self.assets, name="assets")
        self.bounds_factors = Bounds(m=self.k, name="factors")

    def estimate(self, weights, **kwargs):
        """
        Compute the total variance
        """
        var_residual = cvx.norm2(
            cvx.multiply(self.parameter["idiosyncratic_risk"], weights)
        )

        y = kwargs.get("y", self.parameter["exposure"] @ weights)

        return cvx.norm2(
            cvx.vstack([cvx.norm2(self.parameter["chol"] @ y), var_residual])
        )

    def update(self, **kwargs):
        exposure = kwargs["exposure"]
        k, assets = exposure.shape

        self.parameter["exposure"].value[:k, :assets] = kwargs["exposure"]
        self.parameter["idiosyncratic_risk"].value[:assets] = kwargs[
            "idiosyncratic_risk"
        ]
        self.parameter["chol"].value[:k, :k] = cholesky(kwargs["cov"])
        self.bounds_assets.update(**kwargs)
        self.bounds_factors.update(**kwargs)

    def constraints(self, weights, **kwargs):
        y = kwargs.get("y", self.parameter["exposure"] @ weights)

        return (
            self.bounds_assets.constraints(weights)
            + self.bounds_factors.constraints(y)
            + [y == self.parameter["exposure"] @ weights]
        )
