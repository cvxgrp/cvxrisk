#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cvx.core.parameter import Parameter

if TYPE_CHECKING:
    from cvx.core.conic import ConeProgramBuilder
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

    def _solve_and_unpack(
        self,
        builder: ConeProgramBuilder,
        q: np.ndarray,
        weights: Variable,
        w_cols: slice,
        result: Callable[[Any], tuple[float, float]],
        *,
        y_var: Variable | None = None,
        y_cols: slice | None = None,
    ) -> tuple[float | None, float | None, str]:
        """Solve an assembled cone program and unpack the primal solution.

        This is the shared tail of every concrete :meth:`solve_minrisk`: it runs
        the Clarabel solver on the linear objective ``q`` and, on a solved
        status, copies the optimal asset weights (and, when both ``y_var`` and
        ``y_cols`` are supplied, the factor exposures) back into the caller's
        variables. The reported ``(objective, risk)`` pair is derived from the
        solution by the model-specific ``result`` callback. If the solver does
        not converge, returns ``(None, None, status)`` and leaves the variables
        untouched.

        Args:
            builder: The cone program with all constraints already added.
            q: Linear objective coefficients.
            weights: Variable that receives the optimal asset weights.
            w_cols: Columns of the solution vector holding the asset weights.
            result: Maps the solved solution to the ``(objective, risk)`` pair.
            y_var: Optional variable that receives the optimal factor exposures.
            y_cols: Columns of the solution vector holding the factor exposures.

        Returns:
            ``(objective, risk, status)`` on a solved status, otherwise
            ``(None, None, status)``.

        """
        sol, status = builder.solve(q)
        if "Solved" not in status:
            return None, None, status
        weights.value = np.array(sol.x[w_cols])
        if y_var is not None and y_cols is not None:
            y_var.value = np.array(sol.x[y_cols])
        objective, risk = result(sol)
        return objective, risk, status
