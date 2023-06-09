# -*- coding: utf-8 -*-
"""Factor risk model
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np
import pandas as pd

from cvx.risk.model import RiskModel


@dataclass
class FactorModel(RiskModel):
    """Factor risk model"""

    exposure: pd.DataFrame = None
    idiosyncratic_risk: pd.Series = None

    def _variance(self, weights, cov):
        """
        Compute the total variance
        """
        var_factor = cvx.sum_squares(
            np.transpose(np.linalg.cholesky(cov)) @ (self.exposure.values @ weights)
        )

        var_residual = cvx.sum_squares(
            cvx.multiply(self.idiosyncratic_risk.values, weights)
        )

        return var_factor + var_residual
