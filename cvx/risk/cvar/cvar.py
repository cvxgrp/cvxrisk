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
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from ..bounds import Bounds
from ..model import Model


@dataclass
class CVar(Model):
    """Conditional value at risk model"""

    alpha: float = 0.95
    """alpha parameter to determine the size of the tail"""

    n: int = 0
    """number of samples"""

    m: int = 0
    """number of assets"""

    def __post_init__(self):
        """
        Initialize the parameters after the class is instantiated.

        Calculates the number of samples in the tail (k) based on alpha,
        creates the returns parameter matrix, and initializes the bounds.
        """
        self.k = int(self.n * (1 - self.alpha))
        self.parameter["R"] = cvx.Parameter(shape=(self.n, self.m), name="returns", value=np.zeros((self.n, self.m)))
        self.bounds = Bounds(m=self.m, name="assets")

    def estimate(self, weights, **kwargs):
        """
        Estimate the Conditional Value at Risk (CVaR) for the given weights.

        Computes the negative average of the k smallest returns in the portfolio,
        where k is determined by the alpha parameter.

        Args:
            weights: CVXPY variable representing portfolio weights
            **kwargs: Additional keyword arguments (not used)

        Returns:
            CVXPY expression: The negative average of the k smallest returns
        """
        # R is a matrix of returns, n is the number of rows in R
        # k is the number of returns in the left tail
        # average value of the k elements in the left tail
        return -cvx.sum_smallest(self.parameter["R"] @ weights, k=self.k) / self.k

    def update(self, **kwargs):
        """
        Update the returns data and bounds parameters.

        Args:
            **kwargs: Keyword arguments containing:
                - returns: Matrix of returns data
                - Other parameters passed to bounds.update()
        """
        ret = kwargs["returns"]
        m = ret.shape[1]

        self.parameter["R"].value[:, :m] = kwargs["returns"]
        self.bounds.update(**kwargs)

    def constraints(self, weights, **kwargs):
        """
        Return constraints for the CVaR model.

        Args:
            weights: CVXPY variable representing portfolio weights
            **kwargs: Additional keyword arguments passed to bounds.constraints()

        Returns:
            list: List of CVXPY constraints from the bounds object
        """
        return self.bounds.constraints(weights)
