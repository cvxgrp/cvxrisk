"""Risk models for portfolio optimization.

The cvxrisk package provides a collection of risk models for portfolio optimization
using the Clarabel conic solver directly. It supports various risk measures including
sample covariance, factor models, and Conditional Value at Risk (CVaR).

Example:
    Basic usage with sample covariance:

    >>> import numpy as np
    >>> from cvx.risk import Model
    >>> from cvx.risk.sample import SampleCovariance
    >>> from cvx.risk.portfolio import minrisk_problem
    >>> from cvx.core.variable import Variable
    >>> # Create a risk model
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.eye(3),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> # Create and solve optimization
    >>> weights = Variable(3)
    >>> problem = minrisk_problem(model, weights)
    >>> problem.solve()
    >>> np.allclose(weights.value, [1/3, 1/3, 1/3], atol=1e-5)
    True

Modules:
    cvar: Conditional Value at Risk risk model
    factor: Factor-based risk model
    portfolio: Portfolio optimization functions
    sample: Sample covariance risk model

"""

#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
import importlib.metadata

__version__ = importlib.metadata.version("cvxrisk")

from cvx.core import Model as Model
