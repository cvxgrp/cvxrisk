# -*- coding: utf-8 -*-
"""Time series factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np
import pandas as pd

from cvx.risk.factor._model import FactorModel


@dataclass
class TimeseriesFactorRiskModel(FactorModel):
    """Time series factor risk model"""

    factors: pd.DataFrame = None
    returns: pd.DataFrame = None
    cov: pd.DataFrame = None

    def __post_init__(self):
        # B = A * exposure + idiosyncratic_returns
        exposure, _, _, _ = np.linalg.lstsq(
            a=self.factors.values, b=self.returns.values, rcond=None
        )

        exposure = pd.DataFrame(
            index=self.factors.columns, columns=self.returns.columns, data=exposure
        )

        self.exposure = exposure
        self.idiosyncratic_risk = self.idiosyncratic_returns.std()

    @property
    def systematic_returns(self):
        """
        The systematic returns are the factor return * exposure
        """
        return self.factors @ self.exposure

    @property
    def idiosyncratic_returns(self):
        """
        The idiosyncratic_returns are the returns - systematic returns. Hence:

        Returns = systematic returns + idiosyncratic returns
        """
        return self.returns - self.systematic_returns

    #def variance(self, weights):
    #    """
    #    Estimates variance for a given (cvxpy or numpy) vector
    #
    #    Args:
    #        weights: the aforementioned vector
    #    """
    #    return super()._variance(weights, cov=self._covariance)

    @property
    def _covariance(self):
        """
        Either use the covariance matrix provided by the user or compute one
        """
        if self.cov is None:
            return self.factors.cov().values

        return self.cov

    #@property
    #def variance_matrix(self):
    #    """
    #    Constructs the matrix G such that
    #    w'G'G w = variance
    #    """
    #    return super()._variance_matrix(self._covariance)

    def estimate_risk(self, weights, **kwargs):
        return super()._variance(weights, cov=self._covariance)

    def estimate_risk2(self, weights, **kwargs):
        """Estimate the risk by computing a matrix G such that variance = w'G'G w"""
        g = super()._variance_matrix(cov=self._covariance)
        return cvx.sum_squares(g @ weights)