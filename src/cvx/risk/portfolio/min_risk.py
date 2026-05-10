"""Minimum risk portfolio optimization.

This module provides functions for creating and solving minimum risk portfolio
optimization problems using various risk models. Problems are solved directly
with the Clarabel conic solver, without using cvxpy.

Example:
    Create and solve a minimum risk portfolio problem:

    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> from cvx.risk.portfolio import minrisk_problem
    >>> from cvx.core.variable import Variable
    >>> # Create risk model
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> # Create optimization problem
    >>> weights = Variable(3)
    >>> problem = minrisk_problem(model, weights)
    >>> # Solve the problem
    >>> problem.solve()
    >>> # Optimal weights sum to 1
    >>> bool(np.isclose(np.sum(weights.value), 1.0))
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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cvx.core import Model, Variable

# Type alias for user-supplied linear constraints: (a, lb, ub)
# meaning lb <= a @ w <= ub.  Use None for one-sided bounds.
LinearConstraint = tuple[np.ndarray, float | None, float | None]


@dataclass
class MinRiskProblem:
    """A minimum-risk portfolio optimization problem solved with Clarabel.

    This class stores the problem structure and allows the problem to be
    solved (and re-solved after parameter updates) via the :meth:`solve` method.
    After solving, the optimal weights are available via the ``weights`` variable's
    ``value`` attribute, and the optimal risk value is available via ``value``.

    Attributes:
        riskmodel: The risk model defining portfolio risk.
        weights: Variable that will hold the optimal weights after solving.
        base: Base portfolio (numpy array or 0.0). The problem minimizes the
            risk of ``weights - base``.
        value: Optimal objective value after solving (None before solving).
        status: Solver status string after solving (None before solving).

    Example:
        >>> import numpy as np
        >>> from cvx.risk.sample import SampleCovariance
        >>> from cvx.risk.portfolio import minrisk_problem
        >>> from cvx.core.variable import Variable
        >>> model = SampleCovariance(num=2)
        >>> model.update(
        ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> weights = Variable(2)
        >>> problem = minrisk_problem(model, weights)
        >>> problem.solve()
        >>> problem.status
        'Solved'
        >>> bool(np.isclose(np.sum(weights.value), 1.0))
        True

    """

    riskmodel: Model
    weights: Variable
    base: Any = 0.0
    _extra_constraints: list[LinearConstraint] = field(default_factory=list)
    _kwargs: dict[str, Any] = field(default_factory=dict)

    value: float | None = field(default=None, init=False)
    status: str | None = field(default=None, init=False)
    _y_var: Variable | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Extract and store the optional y Variable from kwargs."""
        y = self._kwargs.get("y")
        if isinstance(y, Variable):
            self._y_var = y

    def _get_base_array(self) -> np.ndarray:
        """Return the base portfolio as a numpy array of length weights.n."""
        n = self.weights.n
        if isinstance(self.base, (int, float)) and self.base == 0:
            return np.zeros(n)
        base = np.asarray(self.base)
        result = np.zeros(n)
        m = min(len(base), n)
        result[:m] = base[:m]
        return result

    def solve(self) -> None:
        """Build the Clarabel problem from current parameter values and solve it.

        Updates the ``value`` and ``status`` attributes, and populates
        ``weights.value`` (and ``y.value`` for FactorModel) with the solution.

        After calling ``solve()``, you can update the model parameters and call
        ``solve()`` again without reconstructing the problem structure.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> from cvx.risk.portfolio import minrisk_problem
            >>> from cvx.core.variable import Variable
            >>> model = SampleCovariance(num=2)
            >>> weights = Variable(2)
            >>> problem = minrisk_problem(model, weights)
            >>> model.update(
            ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
            ...     lower_assets=np.zeros(2),
            ...     upper_assets=np.ones(2)
            ... )
            >>> problem.solve()
            >>> bool('Solved' in problem.status)
            True

        """
        base = self._get_base_array()
        obj, _risk, status = self.riskmodel.solve_minrisk(self.weights, base, self._extra_constraints, self._y_var)
        self.value = obj
        self.status = status


def minrisk_problem(
    riskmodel: Model,
    weights: Variable,
    base: Any = 0.0,
    constraints: list[LinearConstraint] | None = None,
    **kwargs: Any,
) -> MinRiskProblem:
    """Create a minimum-risk portfolio optimization problem.

    This function creates a :class:`MinRiskProblem` that minimizes portfolio
    risk subject to standard constraints (weights sum to 1, weight bounds from
    the model) plus any user-supplied linear constraints. The problem is solved
    directly with Clarabel.

    Args:
        riskmodel: A risk model implementing the :class:`~cvx.core.model.Model`
            interface. Supported types: :class:`~cvx.risk.sample.SampleCovariance`,
            :class:`~cvx.risk.factor.FactorModel`,
            :class:`~cvx.risk.cvar.CVar`.
        weights: :class:`~cvx.risk.variable.Variable` that will hold the optimal
            weights after calling :meth:`MinRiskProblem.solve`.
        base: Base portfolio for tracking-error minimization. Can be a numpy array
            of length ``weights.n`` or a scalar (default 0.0 means no base).
        constraints: Optional list of linear constraints on portfolio weights.
            Each constraint is a tuple ``(a, lb, ub)`` specifying
            ``lb <= a @ w <= ub``. Use ``None`` for one-sided bounds.
            For an equality constraint use ``lb == ub``.
        **kwargs: Additional keyword arguments. For :class:`~cvx.risk.factor.FactorModel`,
            pass ``y=Variable(k)`` to expose the factor-exposure solution.

    Returns:
        A :class:`MinRiskProblem` object. Call :meth:`MinRiskProblem.solve` to
        solve it and populate ``weights.value``.

    Example:
        Basic minimum risk portfolio:

        >>> import numpy as np
        >>> from cvx.risk.sample import SampleCovariance
        >>> from cvx.risk.portfolio import minrisk_problem
        >>> from cvx.core.variable import Variable
        >>> model = SampleCovariance(num=2)
        >>> model.update(
        ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> weights = Variable(2)
        >>> problem = minrisk_problem(model, weights)
        >>> problem.solve()
        >>> # Lower variance asset gets higher weight
        >>> bool(weights.value[0] > weights.value[1])
        True

        With base portfolio (tracking error minimization):

        >>> benchmark = np.array([0.5, 0.5])
        >>> problem = minrisk_problem(model, weights, base=benchmark)
        >>> problem.solve()

        With custom constraints (at least 30% in first asset):

        >>> custom_constraints = [(np.array([1, 0]), 0.3, None)]
        >>> problem = minrisk_problem(model, weights, constraints=custom_constraints)
        >>> problem.solve()
        >>> bool(weights.value[0] >= 0.3 - 1e-6)
        True

    """
    return MinRiskProblem(
        riskmodel=riskmodel,
        weights=weights,
        base=base,
        _extra_constraints=constraints or [],
        _kwargs=kwargs,
    )
