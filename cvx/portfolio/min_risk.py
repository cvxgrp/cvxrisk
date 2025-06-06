"""Minimum risk portfolio optimization"""

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

from ..risk import Model


def minrisk_problem(
    riskmodel: Model,
    weights: cp.Variable,
    base: cp.Expression | float = 0.0,
    constraints: list[cp.Constraint] | None = None,
    **kwargs,
) -> cp.Problem:
    """
    Creates a minimum-risk portfolio optimization problem.

    Args:

        riskmodel: A risk model implementing the `Model` interface, used to compute portfolio risk.

        weights: CVXPY variable representing the portfolio weights.

        base: Expression representing the base portfolio (e.g. for tracking error minimization).

        constraints: List of CVXPY constraints applied to the optimization problem.

        **kwargs: Additional keyword arguments passed to the risk model's risk expression.

    Returns:

        A CVXPY problem that minimizes portfolio risk subject to the given constraints.

    """
    # if no constraints are specified
    constraints = constraints or []

    problem = cp.Problem(
        objective=cp.Minimize(riskmodel.estimate(weights - base, **kwargs)),
        constraints=[cp.sum(weights) == 1.0, weights >= 0] + riskmodel.constraints(weights, **kwargs) + constraints,
    )

    return problem
