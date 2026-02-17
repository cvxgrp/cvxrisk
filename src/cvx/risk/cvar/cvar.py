"""Conditional Value at Risk (CVaR) risk model implementation.

This module provides the CVar class, which implements the Conditional Value at Risk
(also known as Expected Shortfall) risk measure for portfolio optimization.

CVaR measures the expected loss in the tail of the portfolio's return distribution,
making it a popular choice for risk-averse portfolio optimization.

Example:
    Create a CVaR model and compute the tail risk:

    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.cvar import CVar
    >>> # Create CVaR model with 95% confidence level
    >>> model = CVar(alpha=0.95, n=100, m=5)
    >>> # Generate sample returns
    >>> np.random.seed(42)
    >>> returns = np.random.randn(100, 5)
    >>> # Update model with returns data
    >>> model.update(
    ...     returns=returns,
    ...     lower_assets=np.zeros(5),
    ...     upper_assets=np.ones(5)
    ... )
    >>> # The model is ready for optimization
    >>> weights = cp.Variable(5)
    >>> risk = model.estimate(weights)
    >>> isinstance(risk, cp.Expression)
    True

"""

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
from typing import Any, cast

import cvxpy as cvx
import numpy as np

from cvx.risk.bounds import Bounds
from cvx.risk.model import Model


@dataclass
class CVar(Model):
    """Conditional Value at Risk (CVaR) risk model.

    CVaR, also known as Expected Shortfall, measures the expected loss in the
    worst (1-alpha) fraction of scenarios. For example, with alpha=0.95, CVaR
    is the average of the worst 5% of returns.

    This implementation uses historical returns to estimate CVaR, which is
    computed as the negative average of the k smallest portfolio returns,
    where k = n * (1 - alpha).

    Attributes:
        alpha: Confidence level, typically 0.95 or 0.99. Higher alpha means
            focusing on more extreme tail events.
        n: Number of historical return observations (scenarios).
        m: Maximum number of assets in the portfolio.

    Example:
        Basic CVaR model setup and optimization:

        >>> import cvxpy as cp
        >>> import numpy as np
        >>> from cvx.risk.cvar import CVar
        >>> from cvx.risk.portfolio import minrisk_problem
        >>> # Create model for 95% CVaR with 50 scenarios and 3 assets
        >>> model = CVar(alpha=0.95, n=50, m=3)
        >>> # Number of tail samples: k = 50 * (1 - 0.95) = 2.5 -> 2
        >>> model.k
        2
        >>> # Generate sample returns
        >>> np.random.seed(42)
        >>> returns = np.random.randn(50, 3)
        >>> model.update(
        ...     returns=returns,
        ...     lower_assets=np.zeros(3),
        ...     upper_assets=np.ones(3)
        ... )
        >>> # Create and solve optimization
        >>> weights = cp.Variable(3)
        >>> problem = minrisk_problem(model, weights)
        >>> _ = problem.solve(solver="CLARABEL")

        Mathematical verification of CVaR calculation:

        >>> model = CVar(alpha=0.95, n=20, m=2)
        >>> # Simple returns: asset 1 always returns 0.05, asset 2 returns vary
        >>> returns = np.zeros((20, 2))
        >>> returns[:, 0] = 0.05  # Asset 1 constant return
        >>> returns[:, 1] = np.linspace(-0.20, 0.18, 20)  # Asset 2 varying
        >>> model.update(
        ...     returns=returns,
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> # k = 20 * (1 - 0.95) = 1, so we take the single worst return
        >>> model.k
        1
        >>> # For 100% in asset 2, worst return is -0.20
        >>> w = np.array([0.0, 1.0])
        >>> cvar = model.estimate(w).value
        >>> expected_cvar = 0.20  # negative of worst return
        >>> bool(np.isclose(cvar, expected_cvar, rtol=1e-6))
        True

        Different alpha values affect the tail focus:

        >>> # Higher alpha = focus on more extreme events
        >>> model_95 = CVar(alpha=0.95, n=100, m=2)
        >>> model_95.k  # Only 5 worst scenarios
        5
        >>> model_75 = CVar(alpha=0.75, n=100, m=2)
        >>> model_75.k  # 25 worst scenarios
        25

    """

    alpha: float = 0.95
    """Confidence level for CVaR (e.g., 0.95 for 95% CVaR)."""

    n: int = 0
    """Number of historical return observations (scenarios)."""

    m: int = 0
    """Maximum number of assets in the portfolio."""

    def __post_init__(self) -> None:
        """Initialize the parameters after the class is instantiated.

        Calculates the number of samples in the tail (k) based on alpha,
        creates the returns parameter matrix, and initializes the bounds.

        Example:
            >>> from cvx.risk.cvar import CVar
            >>> model = CVar(alpha=0.95, n=100, m=5)
            >>> # k is the number of samples in the tail
            >>> model.k
            5
            >>> # Returns parameter is created
            >>> model.parameter["R"].shape
            (100, 5)

        """
        self.k = int(self.n * (1 - self.alpha))
        self.parameter["R"] = cvx.Parameter(shape=(self.n, self.m), name="returns", value=np.zeros((self.n, self.m)))
        self.bounds = Bounds(m=self.m, name="assets")

    def estimate(self, weights: cvx.Variable, **kwargs: Any) -> cvx.Expression:
        """Estimate the Conditional Value at Risk (CVaR) for the given weights.

        Computes the negative average of the k smallest returns in the portfolio,
        where k is determined by the alpha parameter. This represents the expected
        loss in the worst (1-alpha) fraction of scenarios.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            CVXPY expression representing the CVaR (expected tail loss).

        Example:
            >>> import cvxpy as cp
            >>> import numpy as np
            >>> from cvx.risk.cvar import CVar
            >>> model = CVar(alpha=0.95, n=100, m=3)
            >>> np.random.seed(42)
            >>> returns = np.random.randn(100, 3)
            >>> model.update(
            ...     returns=returns,
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3)
            ... )
            >>> weights = cp.Variable(3)
            >>> cvar = model.estimate(weights)
            >>> isinstance(cvar, cp.Expression)
            True

        """
        # R is a matrix of returns, n is the number of rows in R
        # k is the number of returns in the left tail
        # average value of the k elements in the left tail
        return cast(
            cvx.Expression,
            -cvx.sum_smallest(self.parameter["R"] @ weights, k=self.k) / self.k,
        )

    def update(self, **kwargs: Any) -> None:
        """Update the returns data and bounds parameters.

        Updates the returns matrix and asset bounds. The returns matrix can
        have fewer columns than m (maximum assets), in which case only the
        first columns are updated.

        Args:
            **kwargs: Keyword arguments containing:
                - returns: Matrix of returns with shape (n, num_assets).
                - lower_assets: Array of lower bounds for asset weights.
                - upper_assets: Array of upper bounds for asset weights.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.cvar import CVar
            >>> model = CVar(alpha=0.95, n=50, m=5)
            >>> # Update with 3 assets (less than maximum of 5)
            >>> np.random.seed(42)
            >>> returns = np.random.randn(50, 3)
            >>> model.update(
            ...     returns=returns,
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3)
            ... )
            >>> model.parameter["R"].value[:, :3].shape
            (50, 3)

        """
        ret = kwargs["returns"]
        num_assets = ret.shape[1]

        returns_arr = np.zeros((self.n, self.m))
        returns_arr[:, :num_assets] = ret
        self.parameter["R"].value = returns_arr
        self.bounds.update(**kwargs)

    def constraints(self, weights: cvx.Variable, **kwargs: Any) -> list[cvx.Constraint]:
        """Return constraints for the CVaR model.

        Returns the asset bounds constraints from the internal bounds object.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments passed to bounds.constraints().

        Returns:
            List of CVXPY constraints from the bounds object.

        Example:
            >>> import cvxpy as cp
            >>> import numpy as np
            >>> from cvx.risk.cvar import CVar
            >>> model = CVar(alpha=0.95, n=50, m=3)
            >>> np.random.seed(42)
            >>> returns = np.random.randn(50, 3)
            >>> model.update(
            ...     returns=returns,
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3)
            ... )
            >>> weights = cp.Variable(3)
            >>> constraints = model.constraints(weights)
            >>> len(constraints)
            2

        """
        return self.bounds.constraints(weights)
