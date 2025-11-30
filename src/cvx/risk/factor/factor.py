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
"""Factor risk model."""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.bounds import Bounds
from cvx.risk.linalg import cholesky
from cvx.risk.model import Model


@dataclass
class FactorModel(Model):
    """Factor risk model."""

    assets: int = 0
    """Maximal number of assets"""

    k: int = 0
    """Maximal number of factors"""

    def __post_init__(self):
        """Initialize the parameters after the class is instantiated.

        Creates parameters for factor exposure, idiosyncratic risk, and the Cholesky
        decomposition of the factor covariance matrix. Also initializes bounds for
        both assets and factors.
        """
        self.parameter["exposure"] = cvx.Parameter(
            shape=(self.k, self.assets),
            name="exposure",
            value=np.zeros((self.k, self.assets)),
        )

        self.parameter["idiosyncratic_risk"] = cvx.Parameter(
            shape=self.assets, name="idiosyncratic risk", value=np.zeros(self.assets)
        )

        self.parameter["chol"] = cvx.Parameter(
            shape=(self.k, self.k),
            name="cholesky of covariance",
            value=np.zeros((self.k, self.k)),
        )

        self.bounds_assets = Bounds(m=self.assets, name="assets")
        self.bounds_factors = Bounds(m=self.k, name="factors")

    def estimate(self, weights: cvx.Variable, **kwargs) -> cvx.Expression:
        """Compute the total portfolio risk using the factor model.

        Combines systematic risk (from factor exposures) and idiosyncratic risk
        to calculate the total portfolio risk.

        Args:
            weights: CVXPY variable representing portfolio weights

            **kwargs: Additional keyword arguments, may include:

                - y: Factor exposures (if not provided, calculated as exposure @ weights)

        Returns:
            CVXPY expression: The total portfolio risk

        """
        var_residual = cvx.norm2(cvx.multiply(self.parameter["idiosyncratic_risk"], weights))

        y = kwargs.get("y", self.parameter["exposure"] @ weights)

        return cvx.norm2(cvx.vstack([cvx.norm2(self.parameter["chol"] @ y), var_residual]))

    def update(self, **kwargs) -> None:
        """Update the factor model parameters.

        Args:
            **kwargs: Keyword arguments containing:

                - exposure: Factor exposure matrix

                - idiosyncratic_risk: Vector of idiosyncratic risks

                - cov: Factor covariance matrix

                - Other parameters passed to bounds_assets.update() and bounds_factors.update()

        """
        self.parameter["exposure"].value = np.zeros((self.k, self.assets))
        self.parameter["chol"].value = np.zeros((self.k, self.k))
        self.parameter["idiosyncratic_risk"].value = np.zeros(self.assets)

        # get the exposure
        exposure = kwargs["exposure"]

        # extract dimensions
        k, assets = exposure.shape
        if k > self.k:
            raise ValueError("Number of factors exceeds maximal number of factors")
        if assets > self.assets:
            raise ValueError("Number of assets exceeds maximal number of assets")

        self.parameter["exposure"].value[:k, :assets] = kwargs["exposure"]
        self.parameter["idiosyncratic_risk"].value[:assets] = kwargs["idiosyncratic_risk"]
        self.parameter["chol"].value[:k, :k] = cholesky(kwargs["cov"])
        self.bounds_assets.update(**kwargs)
        self.bounds_factors.update(**kwargs)

    def constraints(self, weights: cvx.Variable, **kwargs) -> list[cvx.Constraint]:
        """Return constraints for the factor model.

        Args:
            weights: CVXPY variable representing portfolio weights

            **kwargs: Additional keyword arguments, may include:

                - y: Factor exposures (if not provided, calculated as exposure @ weights)

        Returns:
            List of CVXPY constraints including asset bounds, factor bounds,
            and the constraint that y equals exposure @ weights

        """
        y = kwargs.get("y", self.parameter["exposure"] @ weights)

        return (
            self.bounds_assets.constraints(weights)
            + self.bounds_factors.constraints(y)
            + [y == self.parameter["exposure"] @ weights]
        )
