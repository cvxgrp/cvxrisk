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

    # let's find out what's faster
    def estimate_risk(self, weights, **kwargs):
        return super()._variance(weights, cov=self.factor_covariance)
