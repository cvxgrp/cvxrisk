"""Risk models for portfolio optimization.

The cvxrisk package provides a collection of risk models for portfolio optimization
using CVXPY. It supports various risk measures including sample covariance,
factor models, and Conditional Value at Risk (CVaR).

Example:
    Basic usage with sample covariance:

    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk import Model
    >>> from cvx.risk.sample import SampleCovariance
    >>> from cvx.risk.portfolio import minrisk_problem
    >>> # Create a risk model
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.eye(3),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> # Create and solve optimization
    >>> weights = cp.Variable(3)
    >>> problem = minrisk_problem(model, weights)
    >>> _ = problem.solve(solver="CLARABEL")
    >>> np.allclose(weights.value, [1/3, 1/3, 1/3], atol=1e-5)
    True

Modules:
    bounds: Bounds constraints for portfolio weights
    cvar: Conditional Value at Risk risk model
    factor: Factor-based risk model
    linalg: Linear algebra utilities (Cholesky, PCA, validation)
    model: Abstract base class for risk models
    portfolio: Portfolio optimization functions
    random: Random data generation utilities
    sample: Sample covariance risk model

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
import importlib.metadata

__version__ = importlib.metadata.version("cvxrisk")

from .model import Model as Model  # noqa: F401
