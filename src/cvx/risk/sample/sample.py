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
"""Risk models based on the sample covariance matrix.

This module provides the SampleCovariance class, which implements a risk model
based on the Cholesky decomposition of the sample covariance matrix. This is
one of the most common approaches to portfolio risk estimation.

Example:
    Create and use a sample covariance risk model:

    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> # Create risk model for up to 3 assets
    >>> model = SampleCovariance(num=3)
    >>> # Update with a covariance matrix
    >>> cov = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
    >>> model.update(
    ...     cov=cov,
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> # Estimate risk for a given portfolio
    >>> weights = np.array([0.4, 0.3, 0.3])
    >>> risk = model.estimate(weights)
    >>> isinstance(risk, float)
    True

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from cvx.core.parameter import Parameter
from cvx.risk.bounds import Bounds
from cvx.risk.linalg import cholesky
from cvx.risk.model import Model


@dataclass
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample covariance matrix.

    This model computes portfolio risk as the L2 norm of the product of the
    Cholesky factor and the weights vector. Mathematically, if R is the upper
    triangular Cholesky factor of the covariance matrix (R^T @ R = cov), then:

        risk = ||R @ w||_2 = sqrt(w^T @ cov @ w)

    This represents the portfolio standard deviation (volatility).

    Attributes:
        num: Maximum number of assets the model can handle. The model can be
            updated with fewer assets, but not more.

    Example:
        Basic usage:

        >>> import numpy as np
        >>> from cvx.risk.sample import SampleCovariance
        >>> model = SampleCovariance(num=2)
        >>> model.update(
        ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> # Equal weight portfolio
        >>> weights = np.array([0.5, 0.5])
        >>> risk = model.estimate(weights)
        >>> # Risk should be sqrt(0.5^2 * 1 + 0.5^2 * 2 + 2 * 0.5 * 0.5 * 0.5)
        >>> bool(np.isclose(risk, 1.0))
        True

        Using in optimization:

        >>> from cvx.risk.portfolio import minrisk_problem
        >>> from cvx.core.variable import Variable
        >>> weights = Variable(2)
        >>> problem = minrisk_problem(model, weights)
        >>> problem.solve()
        >>> # Lower variance asset gets higher weight
        >>> bool(weights.value[0] > weights.value[1])
        True

        Mathematical verification - the risk estimate equals sqrt(w^T @ cov @ w):

        >>> model = SampleCovariance(num=3)
        >>> cov = np.array([[0.04, 0.01, 0.02],
        ...                 [0.01, 0.09, 0.01],
        ...                 [0.02, 0.01, 0.16]])
        >>> model.update(
        ...     cov=cov,
        ...     lower_assets=np.zeros(3),
        ...     upper_assets=np.ones(3)
        ... )
        >>> w = np.array([0.4, 0.35, 0.25])
        >>> # Model estimate
        >>> model_risk = model.estimate(w)
        >>> # Manual calculation: sqrt(w^T @ cov @ w)
        >>> manual_risk = np.sqrt(w @ cov @ w)
        >>> bool(np.isclose(model_risk, manual_risk, rtol=1e-6))
        True

    """

    num: int = 0
    """Maximum number of assets the model can handle."""

    def __post_init__(self) -> None:
        """Initialize the parameters after the class is instantiated.

        Creates the Cholesky decomposition parameter and initializes the bounds.
        The Cholesky parameter is a square matrix of size (num, num), and bounds
        are created for asset weights.

        Example:
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=5)
            >>> # Parameters are automatically created
            >>> model.parameter["chol"].shape
            (5, 5)

        """
        self.parameter["chol"] = Parameter(
            shape=(self.num, self.num),
            name="cholesky of covariance",
        )
        self.bounds = Bounds(m=self.num, name="assets")

    def estimate(self, weights: np.ndarray, **kwargs: Any) -> float:
        """Estimate the portfolio risk using the Cholesky decomposition.

        Computes the L2 norm of the product of the Cholesky factor and the
        weights vector. This is equivalent to the square root of the portfolio
        variance (i.e., portfolio volatility).

        Args:
            weights: Numpy array representing portfolio weights.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Float representing the portfolio risk (standard deviation).

        Example:
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=2)
            >>> # Identity covariance (uncorrelated assets with unit variance)
            >>> model.update(
            ...     cov=np.eye(2),
            ...     lower_assets=np.zeros(2),
            ...     upper_assets=np.ones(2)
            ... )
            >>> risk = model.estimate(np.array([0.5, 0.5]))
            >>> isinstance(risk, float)
            True

        """
        return float(np.linalg.norm(self.parameter["chol"].value @ np.asarray(weights)))

    def update(self, **kwargs: Any) -> None:
        """Update the Cholesky decomposition parameter and bounds.

        Computes the Cholesky decomposition of the provided covariance matrix
        and updates the model parameters. The covariance matrix can be smaller
        than num x num.

        Args:
            **kwargs: Keyword arguments containing:
                - cov: Covariance matrix (numpy.ndarray). Must be positive definite.
                - lower_assets: Array of lower bounds for asset weights.
                - upper_assets: Array of upper bounds for asset weights.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=5)
            >>> # Update with a 3x3 covariance (smaller than max)
            >>> cov = np.array([[1.0, 0.3, 0.1],
            ...                 [0.3, 1.0, 0.2],
            ...                 [0.1, 0.2, 1.0]])
            >>> model.update(
            ...     cov=cov,
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3)
            ... )
            >>> # Cholesky factor is updated
            >>> model.parameter["chol"].value[:3, :3].shape
            (3, 3)

        """
        cov = kwargs["cov"]
        n = cov.shape[0]

        chol = np.zeros((self.num, self.num))
        chol[:n, :n] = cholesky(cov)
        self.parameter["chol"].value = chol
        self.bounds.update(**kwargs)
