"""Minimum risk portfolio optimization.

This module provides functions for creating and solving minimum risk portfolio
optimization problems using various risk models. Problems are solved directly
with the Clarabel conic solver, without using cvxpy.

Example:
    Create and solve a minimum risk portfolio problem:

    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> from cvx.risk.portfolio import minrisk_problem
    >>> from cvx.risk.variable import Variable
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

import clarabel
import numpy as np
from scipy import sparse

from cvx.risk.cvar.cvar import CVar
from cvx.risk.factor.factor import FactorModel
from cvx.risk.model import Model
from cvx.risk.sample.sample import SampleCovariance
from cvx.risk.variable import Variable

# Type alias for user-supplied linear constraints: (a, lb, ub)
# meaning lb <= a @ w <= ub.  Use None for one-sided bounds.
LinearConstraint = tuple[np.ndarray, float | None, float | None]


def _clarabel_settings() -> clarabel.DefaultSettings:
    """Return default Clarabel solver settings with verbose output disabled."""
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    return settings


def _solve_sample(
    riskmodel: SampleCovariance,
    weights: Variable,
    base: np.ndarray,
    extra_constraints: list[LinearConstraint],
) -> tuple[float | None, float | None, str]:
    """Build and solve the Clarabel problem for a SampleCovariance model.

    The problem is:
        minimize  t
        subject to  [t; chol @ (w - base)] in SOC(n+1)
                    sum(w) == 1
                    lb <= w <= ub
                    user-supplied linear constraints

    Args:
        riskmodel: The SampleCovariance risk model.
        weights: Variable to store optimal weights.
        base: Base portfolio array (length n).
        extra_constraints: Additional linear constraints.

    Returns:
        Tuple of (objective_value, risk_value, status_string).
    """
    n = weights.n
    chol = riskmodel.parameter["chol"].value  # (n, n) upper triangular
    lb, ub = riskmodel.bounds.get_bounds()

    # Variables: x = [t, w_0, ..., w_{n-1}]  (1 + n total)
    n_vars = 1 + n

    P = sparse.csc_matrix((n_vars, n_vars))  # noqa: N806
    q = np.zeros(n_vars)
    q[0] = 1.0  # minimize t

    A_rows: list[np.ndarray] = []  # noqa: N806
    b_rows: list[np.ndarray] = []
    cones: list[Any] = []

    # 1. SOC: [t; chol @ (w - base)] in SOC(n+1)
    #    b_soc - A_soc @ x in SOC  =>  b_soc - A_soc @ x = [t; chol@(w-base)]
    A_soc = np.zeros((n + 1, n_vars))  # noqa: N806
    A_soc[0, 0] = -1.0  # -t
    A_soc[1:, 1:] = -chol  # -chol @ w
    b_soc = np.zeros(n + 1)
    b_soc[1:] = -chol @ base  # shift by base
    A_rows.append(A_soc)
    b_rows.append(b_soc)
    cones.append(clarabel.SecondOrderConeT(n + 1))

    # 2. Equality: sum(w) == 1
    A_eq = np.zeros((1, n_vars))  # noqa: N806
    A_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])
    A_rows.append(A_eq)
    b_rows.append(b_eq)
    cones.append(clarabel.ZeroConeT(1))

    # 3. Lower bound: w >= lb
    A_lb = np.zeros((n, n_vars))  # noqa: N806
    A_lb[:, 1:] = -np.eye(n)
    A_rows.append(A_lb)
    b_rows.append(-lb)
    cones.append(clarabel.NonnegativeConeT(n))

    # 4. Upper bound: w <= ub
    A_ub = np.zeros((n, n_vars))  # noqa: N806
    A_ub[:, 1:] = np.eye(n)
    A_rows.append(A_ub)
    b_rows.append(ub)
    cones.append(clarabel.NonnegativeConeT(n))

    # 5. User-supplied linear constraints
    for a, lb_val, ub_val in extra_constraints:
        a = np.asarray(a)
        if lb_val is not None and ub_val is not None and lb_val == ub_val:
            # Equality constraint: a @ w == eq
            A_extra = np.zeros((1, n_vars))  # noqa: N806
            A_extra[0, 1:] = a
            A_rows.append(A_extra)
            b_rows.append(np.array([lb_val]))
            cones.append(clarabel.ZeroConeT(1))
        else:
            if lb_val is not None:
                # a @ w >= lb_val
                A_extra = np.zeros((1, n_vars))  # noqa: N806
                A_extra[0, 1:] = -a
                A_rows.append(A_extra)
                b_rows.append(np.array([-lb_val]))
                cones.append(clarabel.NonnegativeConeT(1))
            if ub_val is not None:
                # a @ w <= ub_val
                A_extra = np.zeros((1, n_vars))  # noqa: N806
                A_extra[0, 1:] = a
                A_rows.append(A_extra)
                b_rows.append(np.array([ub_val]))
                cones.append(clarabel.NonnegativeConeT(1))

    A = sparse.csc_matrix(np.vstack(A_rows))  # noqa: N806
    b = np.concatenate(b_rows)

    sol = clarabel.DefaultSolver(P, q, A, b, cones, _clarabel_settings()).solve()
    status = str(sol.status)

    if "Solved" in status:
        weights.value = np.array(sol.x[1 : 1 + n])
        return float(sol.obj_val), float(sol.x[0]), status
    return None, None, status


def _solve_factor(
    riskmodel: FactorModel,
    weights: Variable,
    base: np.ndarray,
    extra_constraints: list[LinearConstraint],
    y_var: Variable | None,
) -> tuple[float | None, float | None, str]:
    """Build and solve the Clarabel problem for a FactorModel.

    The problem is:
        minimize  t
        subject to  [t; chol @ y; diag(idio) @ w] in SOC(1+k+n)
                    sum(w) == 1
                    y == exposure @ w
                    lb_w <= w <= ub_w
                    lb_y <= y <= ub_y
                    user-supplied linear constraints

    Args:
        riskmodel: The FactorModel risk model.
        weights: Variable to store optimal weights.
        base: Base portfolio array (length n).
        extra_constraints: Additional linear constraints.
        y_var: Optional Variable for factor exposures.

    Returns:
        Tuple of (objective_value, risk_value, status_string).
    """
    n = weights.n
    k = riskmodel.k

    chol = riskmodel.parameter["chol"].value  # (k, k) upper triangular
    exposure = riskmodel.parameter["exposure"].value  # (k, n)
    idio = riskmodel.parameter["idiosyncratic_risk"].value  # (n,)

    lb_w, ub_w = riskmodel.bounds_assets.get_bounds()
    lb_y, ub_y = riskmodel.bounds_factors.get_bounds()

    # Variables: x = [t, w_0..w_{n-1}, y_0..y_{k-1}]  (1 + n + k total)
    n_vars = 1 + n + k

    P = sparse.csc_matrix((n_vars, n_vars))  # noqa: N806
    q = np.zeros(n_vars)
    q[0] = 1.0  # minimize t

    A_rows: list[np.ndarray] = []  # noqa: N806
    b_rows: list[np.ndarray] = []
    cones: list[Any] = []

    # 1. SOC: [t; chol @ y; diag(idio) @ (w - base)] in SOC(1+k+n)
    soc_size = 1 + k + n
    A_soc = np.zeros((soc_size, n_vars))  # noqa: N806
    # Row 0: -t
    A_soc[0, 0] = -1.0
    # Rows 1..k: -chol @ y  (y columns are indices 1+n..1+n+k-1)
    A_soc[1 : 1 + k, 1 + n :] = -chol
    # Rows k+1..k+n: -diag(idio) @ w  (w columns are indices 1..1+n-1)
    A_soc[1 + k :, 1 : 1 + n] = -np.diag(idio)

    b_soc = np.zeros(soc_size)
    # Shift rows k+1..k+n by -diag(idio) @ base for the base portfolio
    b_soc[1 + k :] = -idio * base

    A_rows.append(A_soc)
    b_rows.append(b_soc)
    cones.append(clarabel.SecondOrderConeT(soc_size))

    # 2. Equality: sum(w) == 1
    A_eq_sum = np.zeros((1, n_vars))  # noqa: N806
    A_eq_sum[0, 1 : 1 + n] = 1.0
    A_rows.append(A_eq_sum)
    b_rows.append(np.array([1.0]))
    cones.append(clarabel.ZeroConeT(1))

    # 3. Equality: y == exposure @ w  (k equations)
    #    b - A @ x = 0  where b=0, A=[0, -exposure, I_k]
    A_eq_exp = np.zeros((k, n_vars))  # noqa: N806
    A_eq_exp[:, 1 : 1 + n] = -exposure
    A_eq_exp[:, 1 + n :] = np.eye(k)
    A_rows.append(A_eq_exp)
    b_rows.append(np.zeros(k))
    cones.append(clarabel.ZeroConeT(k))

    # 4. Lower bound: w >= lb_w
    A_wlb = np.zeros((n, n_vars))  # noqa: N806
    A_wlb[:, 1 : 1 + n] = -np.eye(n)
    A_rows.append(A_wlb)
    b_rows.append(-lb_w)
    cones.append(clarabel.NonnegativeConeT(n))

    # 5. Upper bound: w <= ub_w
    A_wub = np.zeros((n, n_vars))  # noqa: N806
    A_wub[:, 1 : 1 + n] = np.eye(n)
    A_rows.append(A_wub)
    b_rows.append(ub_w)
    cones.append(clarabel.NonnegativeConeT(n))

    # 6. Lower bound: y >= lb_y
    A_ylb = np.zeros((k, n_vars))  # noqa: N806
    A_ylb[:, 1 + n :] = -np.eye(k)
    A_rows.append(A_ylb)
    b_rows.append(-lb_y)
    cones.append(clarabel.NonnegativeConeT(k))

    # 7. Upper bound: y <= ub_y
    A_yub = np.zeros((k, n_vars))  # noqa: N806
    A_yub[:, 1 + n :] = np.eye(k)
    A_rows.append(A_yub)
    b_rows.append(ub_y)
    cones.append(clarabel.NonnegativeConeT(k))

    # 8. User-supplied linear constraints (on w only)
    for a, lb_val, ub_val in extra_constraints:
        a = np.asarray(a)
        if lb_val is not None and ub_val is not None and lb_val == ub_val:
            A_extra = np.zeros((1, n_vars))  # noqa: N806
            A_extra[0, 1 : 1 + n] = a
            A_rows.append(A_extra)
            b_rows.append(np.array([lb_val]))
            cones.append(clarabel.ZeroConeT(1))
        else:
            if lb_val is not None:
                A_extra = np.zeros((1, n_vars))  # noqa: N806
                A_extra[0, 1 : 1 + n] = -a
                A_rows.append(A_extra)
                b_rows.append(np.array([-lb_val]))
                cones.append(clarabel.NonnegativeConeT(1))
            if ub_val is not None:
                A_extra = np.zeros((1, n_vars))  # noqa: N806
                A_extra[0, 1 : 1 + n] = a
                A_rows.append(A_extra)
                b_rows.append(np.array([ub_val]))
                cones.append(clarabel.NonnegativeConeT(1))

    A = sparse.csc_matrix(np.vstack(A_rows))  # noqa: N806
    b = np.concatenate(b_rows)

    sol = clarabel.DefaultSolver(P, q, A, b, cones, _clarabel_settings()).solve()
    status = str(sol.status)

    if "Solved" in status:
        weights.value = np.array(sol.x[1 : 1 + n])
        if y_var is not None:
            y_var.value = np.array(sol.x[1 + n : 1 + n + k])
        return float(sol.obj_val), float(sol.x[0]), status
    return None, None, status


def _solve_cvar(
    riskmodel: CVar,
    weights: Variable,
    base: np.ndarray,
    extra_constraints: list[LinearConstraint],
) -> tuple[float | None, float | None, str]:
    """Build and solve the Clarabel LP for a CVar model.

    The CVaR minimization problem is formulated as an LP:
        minimize  z + (1/k) * sum_i u_i
        subject to  u_i >= -R_i @ (w - base) - z   for i = 1..T
                    u_i >= 0                          for i = 1..T
                    sum(w) == 1
                    lb <= w <= ub
                    user-supplied linear constraints

    where T = riskmodel.n (number of scenarios) and
    k = riskmodel.k (number of tail scenarios).

    Args:
        riskmodel: The CVar risk model.
        weights: Variable to store optimal weights.
        base: Base portfolio array (length n).
        extra_constraints: Additional linear constraints.

    Returns:
        Tuple of (objective_value, cvar_value, status_string).
    """
    n = weights.n
    T = riskmodel.n  # number of scenarios  # noqa: N806
    k = riskmodel.k  # tail scenarios
    R = riskmodel.parameter["R"].value  # (T, m) returns matrix, m >= n  # noqa: N806
    lb_w, ub_w = riskmodel.bounds.get_bounds()

    # Restrict R to first n columns (actual assets in problem)
    R_n = R[:, :n]  # noqa: N806

    # Variables: x = [w_1..w_n, z, u_1..u_T]  (n + 1 + T total)
    n_vars = n + 1 + T

    P = sparse.csc_matrix((n_vars, n_vars))  # noqa: N806
    q = np.zeros(n_vars)
    # No cost on w
    q[n] = 1.0  # cost z
    q[n + 1 :] = 1.0 / k  # cost u_i

    A_rows: list[np.ndarray] = []  # noqa: N806
    b_rows: list[np.ndarray] = []
    cones: list[Any] = []

    # 1. Nonneg (T): u_i + R_i @ (w - base) + z >= 0
    #    b - A @ [w, z, u] = R_n @ (w - base) + z + u >= 0
    #    A = [-R_n, -1_col, -I_T], b = R_n @ base
    A_cvar = np.zeros((T, n_vars))  # noqa: N806
    A_cvar[:, :n] = -R_n
    A_cvar[:, n] = -1.0
    A_cvar[:, n + 1 :] = -np.eye(T)
    b_cvar = R_n @ base  # zero when base=0

    A_rows.append(A_cvar)
    b_rows.append(b_cvar)
    cones.append(clarabel.NonnegativeConeT(T))

    # 2. Nonneg (T): u >= 0
    A_u = np.zeros((T, n_vars))  # noqa: N806
    A_u[:, n + 1 :] = -np.eye(T)
    A_rows.append(A_u)
    b_rows.append(np.zeros(T))
    cones.append(clarabel.NonnegativeConeT(T))

    # 3. Equality: sum(w) == 1
    A_eq = np.zeros((1, n_vars))  # noqa: N806
    A_eq[0, :n] = 1.0
    A_rows.append(A_eq)
    b_rows.append(np.array([1.0]))
    cones.append(clarabel.ZeroConeT(1))

    # 4. Lower bound: w >= lb_w
    A_lb = np.zeros((n, n_vars))  # noqa: N806
    A_lb[:, :n] = -np.eye(n)
    A_rows.append(A_lb)
    b_rows.append(-lb_w[:n])
    cones.append(clarabel.NonnegativeConeT(n))

    # 5. Upper bound: w <= ub_w
    A_ub = np.zeros((n, n_vars))  # noqa: N806
    A_ub[:, :n] = np.eye(n)
    A_rows.append(A_ub)
    b_rows.append(ub_w[:n])
    cones.append(clarabel.NonnegativeConeT(n))

    # 6. User-supplied linear constraints
    for a, lb_val, ub_val in extra_constraints:
        a = np.asarray(a)
        if lb_val is not None and ub_val is not None and lb_val == ub_val:
            A_extra = np.zeros((1, n_vars))  # noqa: N806
            A_extra[0, :n] = a
            A_rows.append(A_extra)
            b_rows.append(np.array([lb_val]))
            cones.append(clarabel.ZeroConeT(1))
        else:
            if lb_val is not None:
                A_extra = np.zeros((1, n_vars))  # noqa: N806
                A_extra[0, :n] = -a
                A_rows.append(A_extra)
                b_rows.append(np.array([-lb_val]))
                cones.append(clarabel.NonnegativeConeT(1))
            if ub_val is not None:
                A_extra = np.zeros((1, n_vars))  # noqa: N806
                A_extra[0, :n] = a
                A_rows.append(A_extra)
                b_rows.append(np.array([ub_val]))
                cones.append(clarabel.NonnegativeConeT(1))

    A = sparse.csc_matrix(np.vstack(A_rows))  # noqa: N806
    b = np.concatenate(b_rows)

    sol = clarabel.DefaultSolver(P, q, A, b, cones, _clarabel_settings()).solve()
    status = str(sol.status)

    if "Solved" in status:
        weights.value = np.array(sol.x[:n])
        cvar_val = float(q @ sol.x)
        return cvar_val, cvar_val, status
    return None, None, status


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
        >>> from cvx.risk.variable import Variable
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
        # Pad or truncate to length n
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
            >>> from cvx.risk.variable import Variable
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

        if isinstance(self.riskmodel, SampleCovariance):
            obj, _risk, status = _solve_sample(self.riskmodel, self.weights, base, self._extra_constraints)
        elif isinstance(self.riskmodel, FactorModel):
            obj, _risk, status = _solve_factor(self.riskmodel, self.weights, base, self._extra_constraints, self._y_var)
        elif isinstance(self.riskmodel, CVar):
            obj, _risk, status = _solve_cvar(self.riskmodel, self.weights, base, self._extra_constraints)
        else:
            msg = f"Unsupported risk model type: {type(self.riskmodel).__name__}"
            raise NotImplementedError(msg)

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
        riskmodel: A risk model implementing the :class:`~cvx.risk.model.Model`
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
        >>> from cvx.risk.variable import Variable
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

        Equality constraint (force first weight to be zero):

        >>> equality_constraints = [(np.array([1, 0, 0, 0]), 0.0, 0.0)]
        >>> model4 = SampleCovariance(num=4)
        >>> model4.update(
        ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> w4 = Variable(4)
        >>> problem = minrisk_problem(model4, w4, constraints=equality_constraints)
        >>> problem.solve()
        >>> bool(abs(w4.value[0]) < 1e-5)
        True

        With FactorModel and explicit factor exposure variable:

        >>> from cvx.risk.factor import FactorModel
        >>> factor_model = FactorModel(assets=4, k=2)
        >>> factor_model.update(
        ...     exposure=np.array([[1.0, 0.8, 0.2, 0.1],
        ...                        [0.1, 0.2, 0.9, 1.0]]),
        ...     cov=np.eye(2) * 0.04,
        ...     idiosyncratic_risk=np.array([0.1, 0.1, 0.1, 0.1]),
        ...     lower_assets=np.zeros(4),
        ...     upper_assets=np.ones(4),
        ...     lower_factors=-np.ones(2),
        ...     upper_factors=np.ones(2)
        ... )
        >>> w = Variable(4)
        >>> y = Variable(2)  # Factor exposures
        >>> problem = minrisk_problem(factor_model, w, y=y)
        >>> problem.solve()
        >>> bool(np.isclose(np.sum(w.value), 1.0, atol=1e-4))
        True

    """
    return MinRiskProblem(
        riskmodel=riskmodel,
        weights=weights,
        base=base,
        _extra_constraints=constraints or [],
        _kwargs=kwargs,
    )
