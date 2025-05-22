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
    """Maximal number of bounds"""

    name: str = ""
    """Name for the bounds, e.g. assets or factors"""

    def estimate(self, weights: cp.Variable, **kwargs) -> cp.Expression:
        """
        No estimation for bounds.

        Args:

            weights: CVXPY variable representing portfolio weights
            **kwargs: Additional keyword arguments

        Raises:
            NotImplementedError: This method is not implemented for Bounds
        """
        raise NotImplementedError("No estimation for bounds")

    def _f(self, str_prefix: str) -> str:
        """
        Create a parameter name by appending the name attribute.

        Args:

            str_prefix: Base string for the parameter name

        Returns:

            Combined parameter name in the format "{str_prefix}_{self.name}"
        """
        return f"{str_prefix}_{self.name}"

    def __post_init__(self):
        """
        Initialize the parameters after the class is instantiated.

        Creates lower and upper bound parameters with appropriate shapes and default values.
        """
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

    def update(self, **kwargs) -> None:
        """
        Update the lower and upper bound parameters.

        Args:

            **kwargs: Keyword arguments containing lower and upper bounds

                      with keys formatted as "{lower/upper}_{self.name}"
        """
        lower = kwargs[self._f("lower")]
        self.parameter[self._f("lower")].value = np.zeros(self.m)
        self.parameter[self._f("lower")].value[: len(lower)] = lower

        upper = kwargs[self._f("upper")]
        self.parameter[self._f("upper")].value = np.zeros(self.m)
        self.parameter[self._f("upper")].value[: len(upper)] = upper

    def constraints(self, weights: cp.Variable, **kwargs) -> list[cp.Constraint]:
        """
        Return constraints that enforce the bounds on weights.

        Args:

            weights: CVXPY variable representing portfolio weights

            **kwargs: Additional keyword arguments (not used)

        Returns:

            List of CVXPY constraints enforcing lower and upper bounds
        """
        return [
            weights >= self.parameter[self._f("lower")],
            weights <= self.parameter[self._f("upper")],
        ]
