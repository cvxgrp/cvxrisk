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
"""Bounds for portfolio optimization.

This module provides the Bounds class for defining and enforcing lower and upper
bounds on portfolio weights or other variables in optimization problems.

Example:
    Create bounds for a portfolio and use them as constraints:

    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.bounds import Bounds
    >>> # Create bounds for 3 assets
    >>> bounds = Bounds(m=3, name="assets")
    >>> # Update bounds with actual values
    >>> bounds.update(
    ...     lower_assets=np.array([0.0, 0.1, 0.0]),
    ...     upper_assets=np.array([0.5, 0.4, 0.3])
    ... )
    >>> # Create constraints
    >>> weights = cp.Variable(3)
    >>> constraints = bounds.constraints(weights)
    >>> len(constraints)
    2

"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .model import Model


@dataclass
class Bounds(Model):
    """Representation of bounds for a model, defining constraints and parameters.

    This dataclass provides functionality to establish and manage bounds for a model.
    It includes methods to handle bound parameters, update them dynamically, and
    generate constraints that can be used in optimization models.

    The Bounds class creates CVXPY Parameter objects for lower and upper bounds,
    which can be updated without reconstructing the optimization problem.

    Attributes:
        m: Maximum number of bounds (e.g., number of assets or factors).
        name: Name for the bounds used in parameter naming (e.g., "assets" or "factors").

    Example:
        Create and use bounds for portfolio weights:

        >>> import cvxpy as cp
        >>> import numpy as np
        >>> from cvx.risk.bounds import Bounds
        >>> # Create bounds with capacity for 5 assets
        >>> bounds = Bounds(m=5, name="assets")
        >>> # Initialize with actual bounds (can be smaller than m)
        >>> bounds.update(
        ...     lower_assets=np.array([0.0, 0.0, 0.1]),
        ...     upper_assets=np.array([0.5, 0.5, 0.4])
        ... )
        >>> # Check parameter values
        >>> bounds.parameter["lower_assets"].value[:3]
        array([0. , 0. , 0.1])
        >>> bounds.parameter["upper_assets"].value[:3]
        array([0.5, 0.5, 0.4])

        Bounds can be used with different variable types (factors, sectors, etc.):

        >>> factor_bounds = Bounds(m=3, name="factors")
        >>> factor_bounds.update(
        ...     lower_factors=np.array([-0.1, -0.2, -0.15]),
        ...     upper_factors=np.array([0.1, 0.2, 0.15])
        ... )
        >>> # Factor exposure variable
        >>> y = cp.Variable(3)
        >>> factor_constraints = factor_bounds.constraints(y)
        >>> len(factor_constraints)
        2

        Verify bounds are enforced correctly in optimization:

        >>> weights = cp.Variable(5)
        >>> bounds.update(
        ...     lower_assets=np.array([0.3, 0.0, 0.0, 0.0, 0.0]),
        ...     upper_assets=np.array([0.5, 0.2, 0.2, 0.2, 0.2])
        ... )
        >>> prob = cp.Problem(
        ...     cp.Minimize(weights[0]),  # Minimize first weight
        ...     bounds.constraints(weights) + [cp.sum(weights) == 1.0]
        ... )
        >>> _ = prob.solve(solver="CLARABEL")
        >>> # First weight should be at lower bound (0.3)
        >>> bool(np.isclose(weights.value[0], 0.3, atol=1e-4))
        True

    """

    m: int = 0
    """Maximum number of bounds (e.g., number of assets)."""

    name: str = ""
    """Name for the bounds, used in parameter naming (e.g., 'assets' or 'factors')."""

    def estimate(self, weights: cp.Variable, **kwargs) -> cp.Expression:
        """No estimation for bounds.

        Bounds do not provide a risk estimate; they only provide constraints.
        This method raises NotImplementedError.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Always raised as bounds do not provide risk estimates.

        Example:
            >>> import cvxpy as cp
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> weights = cp.Variable(3)
            >>> try:
            ...     bounds.estimate(weights)
            ... except NotImplementedError:
            ...     print("estimate not implemented for Bounds")
            estimate not implemented for Bounds

        """
        raise NotImplementedError("No estimation for bounds")

    def _f(self, str_prefix: str) -> str:
        """Create a parameter name by appending the name attribute.

        This internal method creates consistent parameter names by combining
        a prefix with the bounds name (e.g., "lower_assets" or "upper_factors").

        Args:
            str_prefix: Base string for the parameter name (e.g., "lower" or "upper").

        Returns:
            Combined parameter name in the format "{str_prefix}_{self.name}".

        Example:
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> bounds._f("lower")
            'lower_assets'
            >>> bounds._f("upper")
            'upper_assets'

        """
        return f"{str_prefix}_{self.name}"

    def __post_init__(self):
        """Initialize the parameters after the class is instantiated.

        Creates lower and upper bound CVXPY Parameter objects with appropriate
        shapes and default values. Lower bounds default to zeros, upper bounds
        default to ones.

        Example:
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> # Parameters are automatically created
            >>> bounds.parameter["lower_assets"].shape
            (3,)
            >>> bounds.parameter["upper_assets"].shape
            (3,)

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
        """Update the lower and upper bound parameters.

        This method updates the bound parameters with new values. The input
        arrays can be shorter than m, in which case remaining values are set
        to zero.

        Args:
            **kwargs: Keyword arguments containing lower and upper bounds
                with keys formatted as "{lower/upper}_{self.name}".

        Example:
            >>> import numpy as np
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=5, name="assets")
            >>> # Update with bounds for only 3 assets
            >>> bounds.update(
            ...     lower_assets=np.array([0.0, 0.1, 0.2]),
            ...     upper_assets=np.array([0.5, 0.4, 0.3])
            ... )
            >>> bounds.parameter["lower_assets"].value[:3]
            array([0. , 0.1, 0.2])

        """
        lower = kwargs[self._f("lower")]
        self.parameter[self._f("lower")].value = np.zeros(self.m)
        self.parameter[self._f("lower")].value[: len(lower)] = lower

        upper = kwargs[self._f("upper")]
        self.parameter[self._f("upper")].value = np.zeros(self.m)
        self.parameter[self._f("upper")].value[: len(upper)] = upper

    def constraints(self, weights: cp.Variable, **kwargs) -> list[cp.Constraint]:
        """Return constraints that enforce the bounds on weights.

        Creates CVXPY constraints that enforce the lower and upper bounds
        on the weights variable.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            List of two CVXPY constraints: lower bound and upper bound.

        Example:
            >>> import cvxpy as cp
            >>> import numpy as np
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=2, name="assets")
            >>> bounds.update(
            ...     lower_assets=np.array([0.1, 0.2]),
            ...     upper_assets=np.array([0.6, 0.7])
            ... )
            >>> weights = cp.Variable(2)
            >>> constraints = bounds.constraints(weights)
            >>> len(constraints)
            2

        """
        return [
            weights >= self.parameter[self._f("lower")],
            weights <= self.parameter[self._f("upper")],
        ]
