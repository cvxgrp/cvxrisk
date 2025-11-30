"""Sample covariance risk models for portfolio optimization.

This subpackage provides the SampleCovariance class for risk estimation
based on the sample covariance matrix.

Example:
    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.eye(3),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> weights = cp.Variable(3)
    >>> risk = model.estimate(weights)
    >>> isinstance(risk, cp.Expression)
    True

Classes:
    SampleCovariance: Risk model based on sample covariance matrix

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
from .sample import SampleCovariance as SampleCovariance  # noqa: F401
