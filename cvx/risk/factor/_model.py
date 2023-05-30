# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np
import pandas as pd
from scipy import sparse

from cvx.risk.model import RiskModel


@dataclass
class FactorModel(RiskModel):
    """Factor risk model"""

    exposure: pd.DataFrame = None
    idiosyncratic_risk: pd.Series = None

    def _project_on_factors(self, weights):
        """
        Project the weights (in asset space) down into factor space
        """
        return self.exposure.values @ weights

    def _variance_residual(self, weights):
        """
        Compute the contribution to variance from the idiosyncratic risks
        """
        return cvx.sum_squares(cvx.multiply(self.idiosyncratic_risk.values, weights))

    @staticmethod
    def _cholesky_t(cov):
        return np.transpose(np.linalg.cholesky(cov))

    def _variance_factor(self, weights, cov):
        """
        Compute the contribution to variance from the factor covariance matrix
        """
        return cvx.sum_squares(
            self._cholesky_t(cov) @ self._project_on_factors(weights)
        )

    def _variance(self, weights, cov):
        return self._variance_factor(weights, cov=cov) + self._variance_residual(
            weights
        )

    def _variance_matrix(self, cov):
        """
        Computes the matrix G such that weight'G'G*weight = variance

        Args:
            cov: factor covariance matrix

        Returns:
            The matrix G
        """
        return sparse.vstack(
            (
                sparse.diags(self.idiosyncratic_risk.values, 0),
                sparse.csr_matrix(self._cholesky_t(cov) @ self.exposure.values),
            )
        )
