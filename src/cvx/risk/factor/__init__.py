"""Factor risk models for portfolio optimization.

This subpackage provides factor-based risk models for portfolio optimization.
Factor models decompose portfolio risk into systematic (factor) risk and
idiosyncratic (residual) risk.

Example:
    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.factor import FactorModel
    >>> # Create factor model with 5 assets and 2 factors
    >>> model = FactorModel(assets=5, k=2)
    >>> np.random.seed(42)
    >>> model.update(
    ...     exposure=np.random.randn(2, 5),
    ...     cov=np.eye(2),
    ...     idiosyncratic_risk=np.abs(np.random.randn(5)),
    ...     lower_assets=np.zeros(5),
    ...     upper_assets=np.ones(5),
    ...     lower_factors=-np.ones(2),
    ...     upper_factors=np.ones(2)
    ... )
    >>> weights = cp.Variable(5)
    >>> risk = model.estimate(weights)
    >>> isinstance(risk, cp.Expression)
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
from .factor import FactorModel as FactorModel
