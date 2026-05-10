"""Portfolio optimization models.

This subpackage provides functions for creating portfolio optimization problems
using various risk models. Problems are solved directly with the Clarabel solver.

Example:
    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> from cvx.risk.portfolio import minrisk_problem
    >>> from cvx.core.variable import Variable
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.eye(3),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> weights = Variable(3)
    >>> problem = minrisk_problem(model, weights)
    >>> problem.solve()
    >>> bool(abs(sum(weights.value) - 1.0) < 1e-5)
    True

Functions:
    minrisk_problem: Create a minimum-risk portfolio optimization problem

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

from .min_risk import minrisk_problem as minrisk_problem
