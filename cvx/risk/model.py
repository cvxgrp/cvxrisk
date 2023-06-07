# -*- coding: utf-8 -*-
"""Abstract risk model
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod


class RiskModel(ABC):
    """Abstract risk model"""

    @abstractmethod
    def estimate_risk(self, weights, **kwargs):
        """
        Estimate the variance given the portfolio weights
        """
