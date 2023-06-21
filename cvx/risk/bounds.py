# -*- coding: utf-8 -*-
"""Bounds"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from cvx.risk import Model


@dataclass
class Bounds(Model):
    m: int = 0

    def estimate(self, weights, **kwargs):
        """No estimation for bounds"""
        raise NotImplementedError("No estimation for bounds")

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
        # lower = kwargs.get("lower", np.zeros(self.m))
        lower = kwargs["lower"]
        self.parameter["lower"].value = np.zeros(self.m)
        self.parameter["lower"].value[: len(lower)] = lower

        upper = kwargs["upper"]  # .get("upper", np.ones(self.m))
        self.parameter["upper"].value = np.zeros(self.m)
        self.parameter["upper"].value[
            : len(upper)
        ] = upper  # kwargs.get("upper", np.ones(m))

    def constraints(self, weights, **kwargs):
        return [
            weights >= self.parameter["lower"],
            weights <= self.parameter["upper"],
        ]
