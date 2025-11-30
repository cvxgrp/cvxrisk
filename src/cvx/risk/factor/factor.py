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
"""Factor risk model.

This module provides the FactorModel class, which implements a factor-based
risk model for portfolio optimization. Factor models decompose portfolio risk
into systematic (factor) risk and idiosyncratic (residual) risk.

Example:
    Create a factor model and estimate portfolio risk:

    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.factor import FactorModel
    >>> # Create factor model with 10 assets and 3 factors
    >>> model = FactorModel(assets=10, k=3)
    >>> # Set up factor exposure and covariance
    >>> np.random.seed(42)
    >>> exposure = np.random.randn(3, 10)  # 3 factors x 10 assets
    >>> factor_cov = np.eye(3)  # Factor covariance matrix
    >>> idio_risk = np.abs(np.random.randn(10))  # Idiosyncratic risk
    >>> model.update(
    ...     exposure=exposure,
    ...     cov=factor_cov,
    ...     idiosyncratic_risk=idio_risk,
    ...     lower_assets=np.zeros(10),
    ...     upper_assets=np.ones(10),
    ...     lower_factors=-0.1 * np.ones(3),
    ...     upper_factors=0.1 * np.ones(3)
    ... )
    >>> # Model is ready for optimization
    >>> weights = cp.Variable(10)
    >>> risk = model.estimate(weights)
    >>> isinstance(risk, cp.Expression)
    True

"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from cvx.risk.bounds import Bounds
from cvx.risk.linalg import cholesky
from cvx.risk.model import Model


@dataclass
class FactorModel(Model):
    """Factor risk model for portfolio optimization.

    Factor models decompose portfolio risk into systematic risk (from factor
    exposures) and idiosyncratic risk (residual risk). The total portfolio
    variance is:

        Var(w) = w' @ exposure' @ cov @ exposure @ w + sum((idio_risk * w)^2)

    This implementation uses the Cholesky decomposition of the factor covariance
    matrix for efficient risk computation.

    Attributes:
        assets: Maximum number of assets in the portfolio.
        k: Maximum number of factors in the model.

    Example:
        Create and use a factor model:

        >>> import cvxpy as cp
        >>> import numpy as np
        >>> from cvx.risk.factor import FactorModel
        >>> # Create model
        >>> model = FactorModel(assets=5, k=2)
        >>> # Factor exposure: 2 factors x 5 assets
        >>> exposure = np.array([[1.0, 0.8, 0.6, 0.4, 0.2],
        ...                      [0.2, 0.4, 0.6, 0.8, 1.0]])
        >>> # Factor covariance
        >>> factor_cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        >>> # Idiosyncratic risk per asset
        >>> idio_risk = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        >>> model.update(
        ...     exposure=exposure,
        ...     cov=factor_cov,
        ...     idiosyncratic_risk=idio_risk,
        ...     lower_assets=np.zeros(5),
        ...     upper_assets=np.ones(5),
        ...     lower_factors=-0.5 * np.ones(2),
        ...     upper_factors=0.5 * np.ones(2)
        ... )
        >>> weights = cp.Variable(5)
        >>> risk = model.estimate(weights)
        >>> isinstance(risk, cp.Expression)
        True

    """

    assets: int = 0
    """Maximum number of assets in the portfolio."""

    k: int = 0
    """Maximum number of factors in the model."""

    def __post_init__(self):
        """Initialize the parameters after the class is instantiated.

        Creates parameters for factor exposure, idiosyncratic risk, and the Cholesky
        decomposition of the factor covariance matrix. Also initializes bounds for
        both assets and factors.

        Example:
            >>> from cvx.risk.factor import FactorModel
            >>> model = FactorModel(assets=10, k=3)
            >>> # Parameters are automatically created
            >>> model.parameter["exposure"].shape
            (3, 10)
            >>> model.parameter["idiosyncratic_risk"].shape
            (10,)
            >>> model.parameter["chol"].shape
            (3, 3)

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
        to calculate the total portfolio risk. The formula is:

            risk = sqrt(||chol @ y||^2 + ||idio_risk * w||^2)

        where y = exposure @ weights (factor exposures).

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments, may include:
                - y: Factor exposures variable. If not provided, calculated
                  as exposure @ weights.

        Returns:
            CVXPY expression representing the total portfolio risk.

        Example:
            >>> import cvxpy as cp
            >>> import numpy as np
            >>> from cvx.risk.factor import FactorModel
            >>> model = FactorModel(assets=3, k=2)
            >>> model.update(
            ...     exposure=np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]),
            ...     cov=np.eye(2),
            ...     idiosyncratic_risk=np.array([0.1, 0.1, 0.1]),
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3),
            ...     lower_factors=-np.ones(2),
            ...     upper_factors=np.ones(2)
            ... )
            >>> weights = cp.Variable(3)
            >>> y = cp.Variable(2)  # Factor exposures
            >>> risk = model.estimate(weights, y=y)
            >>> isinstance(risk, cp.Expression)
            True

        """
        var_residual = cvx.norm2(cvx.multiply(self.parameter["idiosyncratic_risk"], weights))

        y = kwargs.get("y", self.parameter["exposure"] @ weights)

        return cvx.norm2(cvx.vstack([cvx.norm2(self.parameter["chol"] @ y), var_residual]))

    def update(self, **kwargs) -> None:
        """Update the factor model parameters.

        Updates the factor exposure matrix, idiosyncratic risk vector, and
        factor covariance Cholesky decomposition. The input dimensions can
        be smaller than the maximum dimensions.

        Args:
            **kwargs: Keyword arguments containing:
                - exposure: Factor exposure matrix (k x assets).
                - idiosyncratic_risk: Vector of idiosyncratic risks.
                - cov: Factor covariance matrix.
                - lower_assets: Array of lower bounds for asset weights.
                - upper_assets: Array of upper bounds for asset weights.
                - lower_factors: Array of lower bounds for factor exposures.
                - upper_factors: Array of upper bounds for factor exposures.

        Raises:
            ValueError: If number of factors or assets exceeds maximum.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.factor import FactorModel
            >>> model = FactorModel(assets=5, k=3)
            >>> # Update with 2 factors and 4 assets
            >>> model.update(
            ...     exposure=np.random.randn(2, 4),
            ...     cov=np.eye(2),
            ...     idiosyncratic_risk=np.abs(np.random.randn(4)),
            ...     lower_assets=np.zeros(4),
            ...     upper_assets=np.ones(4),
            ...     lower_factors=-np.ones(2),
            ...     upper_factors=np.ones(2)
            ... )

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

        Returns constraints including asset bounds, factor exposure bounds,
        and the constraint that relates factor exposures to asset weights.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments, may include:
                - y: Factor exposures variable. If not provided, calculated
                  as exposure @ weights.

        Returns:
            List of CVXPY constraints including asset bounds, factor bounds,
            and the constraint that y equals exposure @ weights.

        Example:
            >>> import cvxpy as cp
            >>> import numpy as np
            >>> from cvx.risk.factor import FactorModel
            >>> model = FactorModel(assets=3, k=2)
            >>> model.update(
            ...     exposure=np.array([[1.0, 0.5, 0.0], [0.0, 0.5, 1.0]]),
            ...     cov=np.eye(2),
            ...     idiosyncratic_risk=np.array([0.1, 0.1, 0.1]),
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3),
            ...     lower_factors=-np.ones(2),
            ...     upper_factors=np.ones(2)
            ... )
            >>> weights = cp.Variable(3)
            >>> y = cp.Variable(2)
            >>> constraints = model.constraints(weights, y=y)
            >>> len(constraints) == 5  # 2 asset bounds + 2 factor bounds + 1 exposure
            True

        """
        y = kwargs.get("y", self.parameter["exposure"] @ weights)

        return (
            self.bounds_assets.constraints(weights)
            + self.bounds_factors.constraints(y)
            + [y == self.parameter["exposure"] @ weights]
        )
