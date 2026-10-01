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

#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from cvx.core import Variable

# Type alias for user-supplied linear constraints: (a, lb, ub)
# meaning lb <= a @ w <= ub.  Use None for one-sided bounds.
LinearConstraint = tuple[np.ndarray, float | None, float | None]


class _SolvableModel(Protocol):
    """Protocol for risk models that support direct Clarabel solving."""

    def solve_minrisk(
        self,
        weights: Variable,
        base: np.ndarray,
        extra_constraints: list[LinearConstraint],
        y_var: Variable | None = None,
    ) -> tuple[float | None, float | None, str]:
        """Solve the minimum-risk problem and return (objective, risk, status)."""
        ...


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
        base: Base portfolio (numpy array or scalar). The problem minimizes the
            risk of ``weights - base``. A scalar is broadcast to every asset;
            a shorter array is zero-padded, a longer one is rejected.
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

    riskmodel: _SolvableModel
    weights: Variable
    base: Any = 0.0
    _extra_constraints: list[LinearConstraint] = field(default_factory=list)
    _kwargs: dict[str, Any] = field(default_factory=dict)

    value: float | None = field(default=None, init=False)
    status: str | None = field(default=None, init=False)
    _y_var: Variable | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate the constraints and kwargs, and store the optional y Variable.

        Raises:
            TypeError: If kwargs holds any key other than ``y``, or ``y`` is not
                a :class:`~cvx.core.variable.Variable`.
            ValueError: If a constraint's coefficient vector does not have
                length ``weights.n``.

        """
        unexpected = sorted(set(self._kwargs) - {"y"})
        if unexpected:
            msg = f"minrisk_problem() got unexpected keyword argument(s): {', '.join(map(repr, unexpected))}"
            raise TypeError(msg)
        if "y" in self._kwargs:
            y = self._kwargs["y"]
            if not isinstance(y, Variable):
                msg = f"y must be a Variable, got {type(y).__name__}"
                raise TypeError(msg)
            self._y_var = y

        n = self.weights.n
        for index, (coeffs, _, _) in enumerate(self._extra_constraints):
            length = np.asarray(coeffs).size
            if length != n:
                msg = f"constraint {index} has {length} coefficients but weights has dimension {n}"
                raise ValueError(msg)

    def _get_base_array(self) -> np.ndarray:
        """Return the base portfolio as a numpy array of length weights.n.

        A scalar base is broadcast to every asset. An array base shorter than
        ``weights.n`` is zero-padded on purpose: the risk models accept fewer
        active assets than their capacity, and the padded slots stay zero.

        Raises:
            ValueError: If an array base is longer than ``weights.n``.

        """
        n = self.weights.n
        base = np.asarray(self.base, dtype=float)
        if base.ndim == 0:
            return np.full(n, float(base))
        if len(base) > n:
            msg = f"base has length {len(base)} but weights has dimension {n}"
            raise ValueError(msg)
        result = np.zeros(n)
        result[: len(base)] = base
        return result

    def solve(self) -> None:
        """Build the Clarabel problem from current parameter values and solve it.

        Updates the ``value`` and ``status`` attributes, and populates
        ``weights.value`` (and ``y.value`` for FactorModel) with the solution.

        After calling ``solve()``, you can update the model parameters and call
        ``solve()`` again without reconstructing the problem structure.

        Failure contract: ``solve()`` does not raise when the problem cannot be
        solved (e.g. it is infeasible or unbounded). Instead, ``status`` is set
        to the solver status, ``value`` stays ``None``, and ``weights.value``
        is left untouched. Always check ``status`` (or ``value is not None``)
        before using the weights.

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
        obj, _, status = self.riskmodel.solve_minrisk(self.weights, base, self._extra_constraints, self._y_var)
        self.value = obj
        self.status = status


def minrisk_problem(
    riskmodel: _SolvableModel,
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
        weights: :class:`~cvx.core.variable.Variable` that will hold the optimal
            weights after calling :meth:`MinRiskProblem.solve`.
        base: Base portfolio for tracking-error minimization. Can be a numpy array
            of length ``weights.n`` or a scalar (default 0.0 means no base).
            A scalar is broadcast to every asset; an array shorter than
            ``weights.n`` is zero-padded, and a longer one raises ``ValueError``
            when the problem is solved.
        constraints: Optional list of linear constraints on portfolio weights.
            Each constraint is a tuple ``(a, lb, ub)`` specifying
            ``lb <= a @ w <= ub``. Use ``None`` for one-sided bounds.
            For an equality constraint use ``lb == ub``.
        **kwargs: Additional keyword arguments. For :class:`~cvx.risk.factor.FactorModel`,
            pass ``y=Variable(k)`` to expose the factor-exposure solution.
            Any other keyword is rejected.

    Returns:
        A :class:`MinRiskProblem` object. Call :meth:`MinRiskProblem.solve` to
        solve it and populate ``weights.value``.

    Raises:
        TypeError: If an unexpected keyword is passed, or ``y`` is not a
            :class:`~cvx.core.variable.Variable`.
        ValueError: If a constraint's coefficient vector does not have length
            ``weights.n``.

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
