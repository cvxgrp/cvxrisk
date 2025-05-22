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


def minrisk_problem(riskmodel, weights, base=0.0, constraints=None, **kwargs):
    """
    Create a minimum risk portfolio optimization problem.

    Args:
        riskmodel: Risk model that implements the Model interface
        weights: CVXPY variable representing portfolio weights
        base: minrisk for weights - base
        constraints: List of constraints applied to the portfolio
        **kwargs: Additional keyword arguments to pass to the risk model

    Returns:
        cp.Problem: A CVXPY problem that minimizes the risk subject to constraints
    """
    # if no constraints are specified
    constraints = constraints or []

    problem = cp.Problem(
        objective=cp.Minimize(riskmodel.estimate(weights - base, **kwargs)),
        constraints=[cp.sum(weights) == 1.0, weights >= 0] + riskmodel.constraints(weights, **kwargs) + constraints,
    )

    return problem
