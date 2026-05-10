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
    Create bounds for a portfolio and query them:

    >>> import numpy as np
    >>> from cvx.risk.bounds import Bounds
    >>> # Create bounds for 3 assets
    >>> bounds = Bounds(m=3, name="assets")
    >>> # Update bounds with actual values
    >>> bounds.update(
    ...     lower_assets=np.array([0.0, 0.1, 0.0]),
    ...     upper_assets=np.array([0.5, 0.4, 0.3])
    ... )
    >>> lb, ub = bounds.get_bounds()
    >>> lb
    array([0. , 0.1, 0. ])
    >>> ub
    array([0.5, 0.4, 0.3])

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core.parameter import Parameter
from .model import Model


@dataclass
class Bounds(Model):
    """Representation of bounds for a model.

    This dataclass provides functionality to establish and manage bounds for
    a model. It includes methods to handle bound parameters and update them
    dynamically. The bounds are stored as :class:`~cvx.risk.parameter.Parameter`
    objects and are used internally by the portfolio optimizer.

    Attributes:
        m: Maximum number of bounds (e.g., number of assets or factors).
        name: Name for the bounds used in parameter naming (e.g., "assets" or "factors").

    Example:
        Create and use bounds for portfolio weights:

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
        >>> lb, ub = factor_bounds.get_bounds()
        >>> lb
        array([-0.1 , -0.2 , -0.15])
        >>> ub
        array([0.1 , 0.2 , 0.15])

    """

    m: int = 0
    """Maximum number of bounds (e.g., number of assets)."""

    name: str = ""
    """Name for the bounds, used in parameter naming (e.g., 'assets' or 'factors')."""

    def estimate(self, weights: np.ndarray, **kwargs: Any) -> float:
        """No estimation for bounds.

        Bounds do not provide a risk estimate; they only provide bound constraints.
        This method raises NotImplementedError.

        Args:
            weights: Numpy array representing portfolio weights.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: Always raised as bounds do not provide risk estimates.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> try:
            ...     bounds.estimate(np.zeros(3))
            ... except NotImplementedError:
            ...     print("estimate not implemented for Bounds")
            estimate not implemented for Bounds

        """
        raise NotImplementedError("No estimation for bounds")

    def _f(self, str_prefix: str) -> str:
        """Create a parameter name by appending the name attribute.

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

    def __post_init__(self) -> None:
        """Initialize the parameters after the class is instantiated.

        Creates lower and upper bound Parameter objects with appropriate
        shapes and default values. Lower bounds default to zeros, upper bounds
        default to ones.

        Example:
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> # Parameters are automatically created
            >>> bounds.parameter["lower_assets"].shape
            3
            >>> bounds.parameter["upper_assets"].shape
            3

        """
        self.parameter[self._f("lower")] = Parameter(
            shape=self.m,
            name="lower bound",
        )
        self.parameter[self._f("upper")] = Parameter(
            shape=self.m,
            name="upper bound",
        )
        self.parameter[self._f("upper")].value = np.ones(self.m)

    def update(self, **kwargs: Any) -> None:
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
        lower_arr = np.zeros(self.m)
        lower_arr[: len(lower)] = lower
        self.parameter[self._f("lower")].value = lower_arr

        upper = kwargs[self._f("upper")]
        upper_arr = np.zeros(self.m)
        upper_arr[: len(upper)] = upper
        self.parameter[self._f("upper")].value = upper_arr

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current lower and upper bound arrays.

        Returns:
            A tuple ``(lb, ub)`` where each element is a numpy array of
            length ``m`` containing the current lower and upper bounds.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> bounds.update(
            ...     lower_assets=np.array([0.1, 0.2, 0.0]),
            ...     upper_assets=np.array([0.6, 0.7, 0.5])
            ... )
            >>> lb, ub = bounds.get_bounds()
            >>> lb
            array([0.1, 0.2, 0. ])
            >>> ub
            array([0.6, 0.7, 0.5])

        """
        return (
            self.parameter[self._f("lower")].value.copy(),
            self.parameter[self._f("upper")].value.copy(),
        )
