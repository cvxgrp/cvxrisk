# -*- coding: utf-8 -*-
"""Fundamental factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cvx.risk.factor._model import FactorModel


@dataclass
class FundamentalFactorRiskModel(FactorModel):
    """Fundamental factor risk model"""

    factor_covariance: pd.DataFrame = None

    def variance(self, weights):
        """
        Estimates variance for a given (cvxpy or numpy) vector

        Args:
            weights: the aforementioned vector
        """
        return super()._variance(weights, cov=self.factor_covariance)

    @property
    def variance_matrix(self):
        """
        Constructs the matrix G such that
        w'G'G w = variance
        """
        return super()._variance_matrix(cov=self.factor_covariance)

    def estimate_risk(self, weights):
        return self.variance(weights)
