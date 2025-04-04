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


def minrisk_problem(riskmodel, weights, **kwargs):
    problem = cp.Problem(
        cp.Minimize(riskmodel.estimate(weights, **kwargs)),
        [cp.sum(weights) == 1.0, weights >= 0] + riskmodel.constraints(weights, **kwargs),
    )

    return problem
