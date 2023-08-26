# -*- coding: utf-8 -*-
"""Abstract risk model
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict

import cvxpy as cp


@dataclass
class Model(ABC):
    """Abstract risk model"""

    parameter: Dict[str, cp.Parameter] = field(default_factory=dict)

    @abstractmethod
    def estimate(self, weights, **kwargs):
        """
        Estimate the variance given the portfolio weights
        """

    @abstractmethod
    def update(self, **kwargs):
        """
        Update the data in the risk model
        """

    @abstractmethod
    def constraints(self, weights, **kwargs):
        """
        Return the constraints for the risk model
        """
