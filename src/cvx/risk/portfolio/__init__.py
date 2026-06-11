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
#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.

from .min_risk import minrisk_problem as minrisk_problem
