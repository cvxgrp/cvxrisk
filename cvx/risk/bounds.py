#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Bounds"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .model import Model


@dataclass
class Bounds(Model):
    m: int = 0
    name: str = ""

    def estimate(self, weights, **kwargs):
        """No estimation for bounds"""
        raise NotImplementedError("No estimation for bounds")

    def _f(self, str):
        return f"{str}_{self.name}"

    def __post_init__(self):
        self.parameter[self._f("lower")] = cp.Parameter(
            shape=self.m,
            name="lower bound",
            value=np.zeros(self.m),
        )
        self.parameter[self._f("upper")] = cp.Parameter(
            shape=self.m,
            name="upper bound",
            value=np.ones(self.m),
        )

    def update(self, **kwargs):
        # lower = kwargs.get("lower", np.zeros(self.m))
        lower = kwargs[self._f("lower")]
        self.parameter[self._f("lower")].value = np.zeros(self.m)
        self.parameter[self._f("lower")].value[: len(lower)] = lower

        upper = kwargs[self._f("upper")]  # .get("upper", np.ones(self.m))
        self.parameter[self._f("upper")].value = np.zeros(self.m)
        self.parameter[self._f("upper")].value[
            : len(upper)
        ] = upper  # kwargs.get("upper", np.ones(m))

    def constraints(self, weights, **kwargs):
        return [
            weights >= self.parameter[self._f("lower")],
            weights <= self.parameter[self._f("upper")],
        ]
