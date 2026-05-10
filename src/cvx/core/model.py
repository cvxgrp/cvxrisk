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
"""Abstract parametric model with named numpy-array parameters.

This module provides the :class:`Model` abstract base class for any
parametric optimization model whose data can be stored as named
:class:`~cvx.core.parameter.Parameter` objects and updated independently
of the problem structure.

Example:
    Concrete subclasses implement ``estimate`` and ``update``:

    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.eye(3),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> isinstance(model.estimate(np.ones(3) / 3), float)
    True

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cvx.core.parameter import Parameter
from cvx.core.variable import Variable


@dataclass
class Model(ABC):
    """Abstract base class for parametric optimization models.

    A ``Model`` holds a dictionary of named :class:`~cvx.core.parameter.Parameter`
    objects (numpy arrays) that can be updated between solver calls without
    reconstructing the optimization problem structure.  Subclasses implement
    :meth:`estimate` to evaluate the model output and :meth:`update` to refresh
    the parameter values.

    Attributes:
        parameter: Dictionary of named :class:`~cvx.core.parameter.Parameter`
            objects.  Parameters can be updated independently of the problem
            structure, making it cheap to solve a sequence of related problems.

    Example:
        >>> import numpy as np
        >>> from cvx.risk.sample import SampleCovariance
        >>> model = SampleCovariance(num=2)
        >>> model.update(
        ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> 'chol' in model.parameter
        True

        Parameters are :class:`~cvx.core.parameter.Parameter` instances:

        >>> from cvx.core.parameter import Parameter
        >>> isinstance(model.parameter['chol'], Parameter)
        True

    """

    parameter: dict[str, Parameter] = field(default_factory=dict)
    """Dictionary of named parameters."""

    @abstractmethod
    def estimate(self, weights: np.ndarray, **kwargs: Any) -> float:
        """Evaluate the model for the given input vector.

        Args:
            weights: Input vector (e.g. portfolio weights or factor exposures).
            **kwargs: Additional keyword arguments for subclass-specific logic.

        Returns:
            Scalar float result (e.g. risk, cost, or objective contribution).

        Example:
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=2)
            >>> model.update(
            ...     cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
            ...     lower_assets=np.zeros(2),
            ...     upper_assets=np.ones(2)
            ... )
            >>> isinstance(model.estimate(np.array([0.5, 0.5])), float)
            True

        """

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update the parameter values from keyword arguments.

        Updating parameters allows the same problem structure to be re-solved
        with new data without any symbolic re-compilation.

        Args:
            **kwargs: New parameter values.  The expected keys depend on the
                concrete subclass.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=3)
            >>> model.update(
            ...     cov=np.eye(3),
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3)
            ... )

        """

    def solve_minrisk(
        self,
        weights: Variable,
        base: np.ndarray,
        extra_constraints: list[tuple[np.ndarray, float | None, float | None]],
        y_var: Variable | None = None,
    ) -> tuple[float | None, float | None, str]:
        """Solve the minimum-risk problem for this model.

        Subclasses that support direct Clarabel solving override this method.
        """
        msg = f"{type(self).__name__} does not implement solve_minrisk"
        raise NotImplementedError(msg)
