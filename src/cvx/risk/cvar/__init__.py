"""Conditional Value at Risk (CVaR) models for portfolio optimization.

This subpackage provides the CVar class for CVaR-based risk estimation.
CVaR, also known as Expected Shortfall, measures the expected loss in the
tail of the return distribution.

Example:
    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.cvar import CVar
    >>> # Create CVaR model
    >>> model = CVar(alpha=0.95, n=100, m=5)
    >>> # Update with historical returns
    >>> np.random.seed(42)
    >>> returns = np.random.randn(100, 5)
    >>> model.update(
    ...     returns=returns,
    ...     lower_assets=np.zeros(5),
    ...     upper_assets=np.ones(5)
    ... )
    >>> weights = cp.Variable(5)
    >>> cvar = model.estimate(weights)
    >>> isinstance(cvar, cp.Expression)
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
from .cvar import CVar as CVar  # noqa: F401
