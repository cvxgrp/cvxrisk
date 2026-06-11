#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
"""Box constraints for optimization variables.

This module provides the :class:`Bounds` class, which tracks lower and upper
bound constraints for a named group of variables.  It works for any bounded
quantity — portfolio weights, factor exposures, sector allocations, etc.

Example:
    >>> import numpy as np
    >>> from cvx.core.bounds import Bounds
    >>> bounds = Bounds(m=3, name="assets")
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

from cvx.core.model import Model
from cvx.core.parameter import Parameter


@dataclass
class Bounds(Model):
    """Box constraints for a named group of optimization variables.

    Stores lower and upper bounds as :class:`~cvx.core.parameter.Parameter`
    objects so they can be updated between solves without rebuilding the
    problem structure.  The ``name`` attribute identifies the variable group
    (e.g. ``"assets"``, ``"factors"``); bound keys are derived as
    ``lower_{name}`` / ``upper_{name}``.

    Attributes:
        m: Capacity — maximum number of variables in the group.
        name: Label for the variable group, used to form parameter key names.

    Example:
        >>> import numpy as np
        >>> from cvx.core.bounds import Bounds
        >>> bounds = Bounds(m=5, name="assets")
        >>> bounds.update(
        ...     lower_assets=np.array([0.0, 0.0, 0.1]),
        ...     upper_assets=np.array([0.5, 0.5, 0.4])
        ... )
        >>> bounds.parameter["lower_assets"].value[:3]
        array([0. , 0. , 0.1])
        >>> bounds.parameter["upper_assets"].value[:3]
        array([0.5, 0.5, 0.4])

        Any variable group name works:

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
    """Capacity — maximum number of variables."""

    name: str = ""
    """Label for the variable group."""

    def estimate(self, weights: np.ndarray, **kwargs: Any) -> float:
        """Not implemented — ``Bounds`` only provides constraint data.

        Args:
            weights: Ignored.
            **kwargs: Ignored.

        Raises:
            NotImplementedError: Always.

        Example:
            >>> import numpy as np
            >>> from cvx.core.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> try:
            ...     bounds.estimate(np.zeros(3))
            ... except NotImplementedError:
            ...     print("estimate not implemented for Bounds")
            estimate not implemented for Bounds

        """
        raise NotImplementedError("Bounds does not implement estimate")

    def _f(self, str_prefix: str) -> str:
        """Return the parameter key ``{str_prefix}_{name}``.

        Example:
            >>> from cvx.core.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
            >>> bounds._f("lower")
            'lower_assets'
            >>> bounds._f("upper")
            'upper_assets'

        """
        return f"{str_prefix}_{self.name}"

    def __post_init__(self) -> None:
        """Create lower (zeros) and upper (ones) bound parameters.

        Example:
            >>> from cvx.core.bounds import Bounds
            >>> bounds = Bounds(m=3, name="assets")
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
        """Update bound parameters from keyword arguments.

        Input arrays shorter than ``m`` are zero-padded on the right.

        Args:
            **kwargs: Must contain ``lower_{name}`` and ``upper_{name}`` keys
                with numpy arrays of length ≤ ``m``.

        Raises:
            ValueError: If a required key is missing or an array is longer
                than ``m``.

        Example:
            >>> import numpy as np
            >>> from cvx.core.bounds import Bounds
            >>> bounds = Bounds(m=5, name="assets")
            >>> bounds.update(
            ...     lower_assets=np.array([0.0, 0.1, 0.2]),
            ...     upper_assets=np.array([0.5, 0.4, 0.3])
            ... )
            >>> bounds.parameter["lower_assets"].value[:3]
            array([0. , 0.1, 0.2])

        """
        for key in (self._f("lower"), self._f("upper")):
            if key not in kwargs:
                msg = f"update() requires a '{key}' argument"
                raise ValueError(msg)
            values = kwargs[key]
            if len(values) > self.m:
                msg = f"'{key}' has length {len(values)} but the maximum is {self.m}"
                raise ValueError(msg)
            arr = np.zeros(self.m)
            arr[: len(values)] = values
            self.parameter[key].value = arr

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` bound arrays of length ``m``.

        Example:
            >>> import numpy as np
            >>> from cvx.core.bounds import Bounds
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
