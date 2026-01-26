"""Minimum risk portfolio optimization.

This module provides functions for creating and solving minimum risk portfolio
optimization problems using various risk models.

Example:
    Create and solve a minimum risk portfolio problem:

    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> from cvx.risk.portfolio import minrisk_problem
    >>> # Create risk model
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]]),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> # Create optimization problem
    >>> weights = cp.Variable(3)
    >>> problem = minrisk_problem(model, weights)
    >>> # Solve the problem
    >>> _ = problem.solve(solver="CLARABEL")
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

import cvxpy as cp

from cvx.risk import Model


def minrisk_problem(
    riskmodel: Model,
    weights: cp.Variable,
    base: cp.Expression | float = 0.0,
    constraints: list[cp.Constraint] | None = None,
    **kwargs,
) -> cp.Problem:
    """Create a minimum-risk portfolio optimization problem.

    This function creates a CVXPY optimization problem that minimizes portfolio
    risk subject to constraints. The problem includes standard constraints
    (weights sum to 1, weights are non-negative) plus any model-specific and
    custom constraints.

    Args:
        riskmodel: A risk model implementing the `Model` interface, used to
            compute portfolio risk. Can be SampleCovariance, FactorModel,
            CVar, etc.
        weights: CVXPY variable representing the portfolio weights. Should have
            shape (n,) where n is the number of assets.
        base: Expression representing the base portfolio (default 0.0). Use this
            for tracking error minimization where you want to minimize the risk
            of deviating from a benchmark.
        constraints: Optional list of additional CVXPY constraints to apply to
            the optimization problem.
        **kwargs: Additional keyword arguments passed to the risk model's
            estimate and constraints methods.

    Returns:
        A CVXPY Problem that minimizes portfolio risk subject to constraints.
        The problem includes:
        - Objective: minimize risk(weights - base)
        - Constraint: sum(weights) == 1
        - Constraint: weights >= 0
        - Model-specific constraints from riskmodel.constraints()
        - Any additional constraints passed in the constraints argument

    Example:
        Basic minimum risk portfolio:

        >>> import cvxpy as cp
        >>> import numpy as np
        >>> from cvx.risk.sample import SampleCovariance
        >>> from cvx.risk.portfolio import minrisk_problem
        >>> model = SampleCovariance(num=2)
        >>> model.update(
        ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> weights = cp.Variable(2)
        >>> problem = minrisk_problem(model, weights)
        >>> _ = problem.solve(solver="CLARABEL")
        >>> # Lower variance asset gets higher weight
        >>> bool(weights.value[0] > weights.value[1])
        True

        With tracking error (minimize deviation from benchmark):

        >>> benchmark = np.array([0.5, 0.5])
        >>> problem = minrisk_problem(model, weights, base=benchmark)
        >>> _ = problem.solve(solver="CLARABEL")

        With custom constraints:

        >>> custom_constraints = [weights[0] >= 0.3]  # At least 30% in first asset
        >>> problem = minrisk_problem(model, weights, constraints=custom_constraints)
        >>> _ = problem.solve(solver="CLARABEL")
        >>> bool(weights.value[0] >= 0.3 - 1e-6)
        True

    """
    # if no constraints are specified
    constraints = constraints or []

    problem = cp.Problem(
        objective=cp.Minimize(riskmodel.estimate(weights - base, **kwargs)),
        constraints=[cp.sum(weights) == 1.0, weights >= 0, *riskmodel.constraints(weights, **kwargs), *constraints],
    )

    return problem
