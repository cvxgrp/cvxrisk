# -*- coding: utf-8 -*-
"""Abstract risk model
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class RiskModel(ABC):
    """Abstract risk model"""

    @abstractmethod
    def estimate_risk(self, weights, **kwargs):
        """
        Estimate the variance given the portfolio weights
        """

    def update_data(self, **kwargs):
        """
        Update the data in the risk model
        """
