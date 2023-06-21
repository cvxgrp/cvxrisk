# -*- coding: utf-8 -*-
"""Abstract risk model
"""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

import cvxpy as cp
import numpy as np


@dataclass
class RiskModel(ABC):
    """Abstract risk model"""

    parameter: Dict[str, cp.Parameter] = field(default_factory=dict)

    @abstractmethod
    def estimate(self, weights, **kwargs):
        """
        Estimate the variance given the portfolio weights
        """

    def update(self, **kwargs):
        """
        Update the data in the risk model
        """

    def constraints(self, weights, **kwargs):
        """
        Return the constraints for the risk model
        """


@dataclass
class Bounds(RiskModel):
    def estimate(self, weights, **kwargs):
        """No estimation for bounds"""
        raise NotImplementedError("No estimation for bounds")

    m: int = 0

    def __post_init__(self):
        self.parameter["lower"] = cp.Parameter(
            shape=self.m,
            name="lower bound",
            value=np.zeros(self.m),
        )
        self.parameter["upper"] = cp.Parameter(
            shape=self.m,
            name="upper bound",
            value=np.ones(self.m),
        )

    def update(self, **kwargs):
        lower = kwargs.get("lower", np.zeros(self.m))
        self.parameter["lower"].value = np.zeros(self.m)
        self.parameter["lower"].value[: len(lower)] = lower

        upper = kwargs.get("upper", np.ones(self.m))
        self.parameter["upper"].value = np.zeros(self.m)
        self.parameter["upper"].value[
            : len(upper)
        ] = upper  # kwargs.get("upper", np.ones(m))

    def constraints(self, weights, **kwargs):
        return [
            weights >= self.parameter["lower"],
            weights <= self.parameter["upper"],
        ]
