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
from typing import Any

import numpy as np

from cvx.core.conic import ConeProgramBuilder
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

    def _finalize_solve(
        self,
        builder: ConeProgramBuilder,
        q: np.ndarray,
        weights: Variable,
        w_cols: slice,
        result: Callable[[Any], tuple[float, float]],
    ) -> tuple[float | None, float | None, str]:
        """Solve the assembled conic program and populate the solution.

        Shared epilogue for every :meth:`solve_minrisk` implementation: run the
        Clarabel solver, and on success populate ``weights`` from the ``w_cols``
        slice and derive the ``(objective, risk)`` pair via ``result``. On any
        non-``Solved`` status the weights are left untouched and
        ``(None, None, status)`` is returned, honouring the failure contract of
        :meth:`~cvx.risk.portfolio.min_risk.MinRiskProblem.solve`.

        Args:
            builder: The :class:`~cvx.core.conic.ConeProgramBuilder` holding the
                assembled constraint blocks.
            q: Linear objective vector passed to the solver.
            weights: Variable populated with ``sol.x[w_cols]`` when solved.
            w_cols: Column slice selecting the weight variables in the solution.
            result: Callback mapping the solved Clarabel solution to the
                ``(objective, risk)`` return pair. It may also populate
                model-specific extra variables (e.g. the factor exposures).

        Returns:
            Tuple ``(objective, risk, status)``; the first two are ``None`` when
            the solver did not reach a ``Solved`` status.

        """
        sol, status = builder.solve(q)
        if "Solved" not in status:
            return None, None, status
        weights.value = np.array(sol.x[w_cols])
        objective, risk = result(sol)
        return objective, risk, status
