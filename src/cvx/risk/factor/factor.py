#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
"""Factor risk model.

This module provides the FactorModel class, which implements a factor-based
risk model for portfolio optimization. Factor models decompose portfolio risk
into systematic (factor) risk and idiosyncratic (residual) risk.

Example:
    Create a factor model and estimate portfolio risk:

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
    >>> w = np.zeros(10)
    >>> w[:5] = 0.2
    >>> risk = model.estimate(w)
    >>> isinstance(risk, float)
    True

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import clarabel
import numpy as np
from cvx.linalg import cholesky, norm
from scipy import sparse

from cvx.core import Bounds, ConeProgramBuilder, Model, Parameter, Variable


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
        >>> w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        >>> risk = model.estimate(w)
        >>> isinstance(risk, float)
        True

        Mathematical verification of risk decomposition:

        >>> model = FactorModel(assets=3, k=2)
        >>> # Factor exposure: how much each asset is exposed to each factor
        >>> exposure = np.array([[1.0, 0.5, 0.0],   # Market factor
        ...                      [0.0, 0.5, 1.0]])  # Sector factor
        >>> # Factor covariance (diagonal = uncorrelated factors)
        >>> factor_cov = np.array([[0.04, 0.0],     # Market vol = 20%
        ...                        [0.0, 0.0225]])  # Sector vol = 15%
        >>> # Idiosyncratic risk per asset
        >>> idio = np.array([0.10, 0.12, 0.08])
        >>> model.update(
        ...     exposure=exposure,
        ...     cov=factor_cov,
        ...     idiosyncratic_risk=idio,
        ...     lower_assets=np.zeros(3),
        ...     upper_assets=np.ones(3),
        ...     lower_factors=-np.ones(2),
        ...     upper_factors=np.ones(2)
        ... )
        >>> # Equal weight portfolio
        >>> w = np.array([1/3, 1/3, 1/3])
        >>> model_risk = model.estimate(w)
        >>> # Manual: total_var = y^T @ cov @ y + sum((idio * w)^2)
        >>> y = exposure @ w  # Factor exposures
        >>> systematic_var = y @ factor_cov @ y
        >>> idio_var = np.sum((idio * w)**2)
        >>> manual_risk = np.sqrt(systematic_var + idio_var)
        >>> bool(np.isclose(model_risk, manual_risk, rtol=1e-5))
        True

        Error handling for dimension violations:

        >>> model = FactorModel(assets=3, k=2)
        >>> try:
        ...     model.update(
        ...         exposure=np.random.randn(5, 3),  # 5 factors > k=2
        ...         cov=np.eye(5),
        ...         idiosyncratic_risk=np.ones(3),
        ...         lower_assets=np.zeros(3),
        ...         upper_assets=np.ones(3),
        ...         lower_factors=-np.ones(5),
        ...         upper_factors=np.ones(5)
        ...     )
        ... except ValueError as e:
        ...     print("Caught:", str(e))
        Caught: Too many factors

    """

    assets: int = 0
    """Maximum number of assets in the portfolio."""

    k: int = 0
    """Maximum number of factors in the model."""

    def __post_init__(self) -> None:
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
            10
            >>> model.parameter["chol"].shape
            (3, 3)

        """
        self.parameter["exposure"] = Parameter(
            shape=(self.k, self.assets),
            name="exposure",
        )

        self.parameter["idiosyncratic_risk"] = Parameter(shape=self.assets, name="idiosyncratic risk")

        self.parameter["chol"] = Parameter(
            shape=(self.k, self.k),
            name="cholesky of covariance",
        )

        self.bounds_assets = Bounds(m=self.assets, name="assets")
        self.bounds_factors = Bounds(m=self.k, name="factors")

    def estimate(self, weights: np.ndarray, **kwargs: Any) -> float:
        """Compute the total portfolio risk using the factor model.

        Combines systematic risk (from factor exposures) and idiosyncratic risk
        to calculate the total portfolio risk. The formula is:

            risk = sqrt(||chol @ y||^2 + ||idio_risk * w||^2)

        where y = exposure @ weights (factor exposures).

        Args:
            weights: Numpy array representing portfolio weights.
            **kwargs: Additional keyword arguments, may include:
                - y: Factor exposures as a numpy array. If not provided, calculated
                  as exposure @ weights.

        Returns:
            Float representing the total portfolio risk.

        Example:
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
            >>> w = np.array([0.4, 0.3, 0.3])
            >>> risk = model.estimate(w)
            >>> isinstance(risk, float)
            True

        """
        w = np.asarray(weights)
        y = np.asarray(kwargs.get("y", self.parameter["exposure"].value @ w))

        var_systematic = norm(self.parameter["chol"].value @ y)
        var_residual = norm(self.parameter["idiosyncratic_risk"].value * w)

        return float(np.sqrt(var_systematic**2 + var_residual**2))

    def update(self, **kwargs: Any) -> None:
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
            ValueError: If a required argument is missing, the number of
                factors or assets exceeds the maximum, or the shapes of
                ``cov`` and ``idiosyncratic_risk`` are inconsistent with
                ``exposure``.

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
        missing = [key for key in ("exposure", "cov", "idiosyncratic_risk") if key not in kwargs]
        if missing:
            msg = f"update() missing required arguments: {', '.join(missing)}"
            raise ValueError(msg)

        exposure = np.asarray(kwargs["exposure"])
        cov = np.asarray(kwargs["cov"])
        idiosyncratic_risk = np.asarray(kwargs["idiosyncratic_risk"])

        # extract dimensions
        k, assets = exposure.shape
        if k > self.k:
            msg = "Too many factors"
            raise ValueError(msg)
        if assets > self.assets:
            msg = "Too many assets"
            raise ValueError(msg)
        if cov.shape != (k, k):
            msg = f"cov must have shape ({k}, {k}) to match exposure, got {cov.shape}"
            raise ValueError(msg)
        if idiosyncratic_risk.shape != (assets,):
            msg = f"idiosyncratic_risk must have shape ({assets},) to match exposure, got {idiosyncratic_risk.shape}"
            raise ValueError(msg)

        self.parameter["exposure"].value = np.zeros((self.k, self.assets))
        self.parameter["chol"].value = np.zeros((self.k, self.k))
        self.parameter["idiosyncratic_risk"].value = np.zeros(self.assets)

        self.parameter["exposure"].value[:k, :assets] = exposure
        self.parameter["idiosyncratic_risk"].value[:assets] = idiosyncratic_risk
        self.parameter["chol"].value[:k, :k] = cholesky(cov)
        self.bounds_assets.update(**kwargs)
        self.bounds_factors.update(**kwargs)

    def solve_minrisk(
        self,
        weights: Variable,
        base: np.ndarray,
        extra_constraints: list[tuple[np.ndarray, float | None, float | None]],
        y_var: Variable | None = None,
    ) -> tuple[float | None, float | None, str]:
        """Build and solve the Clarabel SOC problem for this model.

        Raises:
            ValueError: If the weights dimension does not match the model
                capacity ``assets``.

        """
        n = weights.n
        if n != self.assets:
            msg = f"weights has dimension {n} but the model capacity is assets={self.assets}"
            raise ValueError(msg)
        k = self.k

        chol = self.parameter["chol"].value
        exposure = self.parameter["exposure"].value
        idio = self.parameter["idiosyncratic_risk"].value

        lb_w, ub_w = self.bounds_assets.get_bounds()
        lb_y, ub_y = self.bounds_factors.get_bounds()

        # Variables: x = [t, w, y] with t bounding the total volatility
        # and y the factor exposures.
        w_cols = slice(1, 1 + n)
        y_cols = slice(1 + n, 1 + n + k)
        builder = ConeProgramBuilder(n_vars=1 + n + k)

        # SOC: || [chol @ exposure @ (w - base); idio * (w - base)] ||_2 <= t
        # encoded via y = exposure @ w, so the systematic term is chol @ (y - exposure @ base).
        # Built directly in sparse form: the block has only O(n + k^2) nonzeros.
        soc_size = 1 + k + n
        a_soc = sparse.bmat(
            [
                [sparse.csr_matrix(np.array([[-1.0]])), None, None],
                [None, None, sparse.csr_matrix(-chol)],
                [None, sparse.diags(-idio), None],
            ],
            format="csr",
        )
        b_soc = np.zeros(soc_size)
        b_soc[1 : 1 + k] = -chol @ (exposure @ base)
        b_soc[1 + k :] = -idio * base
        builder.add(a_soc, b_soc, clarabel.SecondOrderConeT(soc_size))

        builder.add_sum_constraint(w_cols)

        # Equality: y = exposure @ w
        a_exp = builder.block(k)
        a_exp[:, w_cols] = -exposure
        a_exp[:, y_cols] = np.eye(k)
        builder.add(a_exp, np.zeros(k), clarabel.ZeroConeT(k))

        builder.add_variable_bounds(w_cols, lb_w, ub_w)
        builder.add_variable_bounds(y_cols, lb_y, ub_y)
        builder.add_linear_constraints(extra_constraints, w_cols)

        q = np.zeros(builder.n_vars)
        q[0] = 1.0
        sol, status = builder.solve(q)

        if "Solved" in status:
            weights.value = np.array(sol.x[w_cols])
            if y_var is not None:
                y_var.value = np.array(sol.x[y_cols])
            return float(sol.obj_val), float(sol.x[0]), status
        return None, None, status
