"""Conditional Value at Risk (CVaR) risk model implementation.

This module provides the CVar class, which implements the Conditional Value at Risk
(also known as Expected Shortfall) risk measure for portfolio optimization.

CVaR measures the expected loss in the tail of the portfolio's return distribution,
making it a popular choice for risk-averse portfolio optimization.

Example:
    Create a CVaR model and compute the tail risk:

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
    >>> # The model is ready for use
    >>> w = np.ones(5) / 5
    >>> risk = model.estimate(w)
    >>> isinstance(risk, float)
    True

"""

#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import clarabel
import numpy as np
from scipy import sparse

from cvx.core import Bounds, ConeProgramBuilder, Model, Parameter, Variable


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
        Basic CVaR model setup:

        >>> import numpy as np
        >>> from cvx.risk.cvar import CVar
        >>> from cvx.risk.portfolio import minrisk_problem
        >>> from cvx.core.variable import Variable
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
        >>> weights = Variable(3)
        >>> problem = minrisk_problem(model, weights)
        >>> problem.solve()

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
        >>> cvar = model.estimate(w)
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
        self.parameter["R"] = Parameter(shape=(self.n, self.m), name="returns")
        self.bounds = Bounds(m=self.m, name="assets")

    def estimate(self, weights: np.ndarray, **kwargs: Any) -> float:  # noqa: ARG002
        """Estimate the Conditional Value at Risk (CVaR) for the given weights.

        Computes the negative average of the k smallest returns in the portfolio,
        where k is determined by the alpha parameter. This represents the expected
        loss in the worst (1-alpha) fraction of scenarios.

        Args:
            weights: Numpy array representing portfolio weights.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            Float representing the CVaR (expected tail loss).

        Example:
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
            >>> w = np.array([1/3, 1/3, 1/3])
            >>> cvar = model.estimate(w)
            >>> isinstance(cvar, float)
            True

        """
        portfolio_returns = self.parameter["R"].value @ np.asarray(weights)
        sorted_returns = np.sort(portfolio_returns)
        # Take the k smallest (worst) returns and average them
        return float(-np.mean(sorted_returns[: self.k]))

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

        Raises:
            ValueError: If ``returns`` is missing, has the wrong number of
                scenarios, or more columns than the model capacity ``m``.

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
        if "returns" not in kwargs:
            msg = "update() requires a 'returns' argument"
            raise ValueError(msg)
        ret = np.asarray(kwargs["returns"])
        if ret.ndim != 2:
            msg = f"returns must be a 2d matrix of shape (n, num_assets), got shape {ret.shape}"
            raise ValueError(msg)
        if ret.shape[0] != self.n:
            msg = f"returns has {ret.shape[0]} scenarios but the model expects n={self.n}"
            raise ValueError(msg)
        if ret.shape[1] > self.m:
            msg = f"Too many assets: returns has {ret.shape[1]} columns but the model capacity is m={self.m}"
            raise ValueError(msg)
        num_assets = ret.shape[1]

        returns_arr = np.zeros((self.n, self.m))
        returns_arr[:, :num_assets] = ret
        self.parameter["R"].value = returns_arr
        self.bounds.update(**kwargs)

    def solve_minrisk(
        self,
        weights: Variable,
        base: np.ndarray,
        extra_constraints: list[tuple[np.ndarray, float | None, float | None]],
        y_var: Variable | None = None,  # noqa: ARG002
    ) -> tuple[float | None, float | None, str]:
        """Build and solve the Clarabel LP for this model.

        Raises:
            ValueError: If the weights dimension exceeds the model capacity ``m``.

        """
        n = weights.n
        if n > self.m:
            msg = f"weights has dimension {n} but the model capacity is m={self.m}"
            raise ValueError(msg)
        T = self.n  # noqa: N806
        k = self.k
        R = self.parameter["R"].value  # noqa: N806
        lb_w, ub_w = self.bounds.get_bounds()

        R_n = R[:, :n]  # noqa: N806

        # Variables: x = [w, gamma, u] (Rockafellar-Uryasev formulation)
        # with gamma the VaR level and u the scenario excess losses.
        w_cols = slice(0, n)
        gamma_col = n
        u_cols = slice(n + 1, n + 1 + T)
        builder = ConeProgramBuilder(n_vars=n + 1 + T)

        # u >= -R @ (w - base) - gamma  (scenario losses beyond VaR)
        # Built directly in sparse form: dense (T x n_vars) blocks would be
        # O(T^2) memory for what is mostly an identity over the u variables.
        a_cvar = sparse.hstack(
            [sparse.csr_matrix(-R_n), sparse.csr_matrix(-np.ones((T, 1))), -sparse.identity(T, format="csr")],
            format="csr",
        )
        builder.add(a_cvar, R_n @ base, clarabel.NonnegativeConeT(T))

        # u >= 0
        a_u = sparse.hstack(
            [sparse.csr_matrix((T, n + 1)), -sparse.identity(T, format="csr")],
            format="csr",
        )
        builder.add(a_u, np.zeros(T), clarabel.NonnegativeConeT(T))

        builder.add_sum_constraint(w_cols)
        builder.add_variable_bounds(w_cols, lb_w[:n], ub_w[:n])
        builder.add_linear_constraints(extra_constraints, w_cols)

        q = np.zeros(builder.n_vars)
        q[gamma_col] = 1.0
        q[u_cols] = 1.0 / k
        sol, status = builder.solve(q)

        if "Solved" in status:
            weights.value = np.array(sol.x[w_cols])
            cvar_val = float(q @ sol.x)
            return cvar_val, cvar_val, status
        return None, None, status
