"""Random data generation utilities for testing and simulation.

This subpackage provides functions for generating random data useful for
testing portfolio optimization algorithms.

Example:
    >>> import numpy as np
    >>> from cvx.risk.rand import rand_cov
    >>> # Generate a random 5x5 covariance matrix
    >>> cov = rand_cov(5, seed=42)
    >>> cov.shape
    (5, 5)
    >>> # Verify positive semi-definite
    >>> bool(np.all(np.linalg.eigvals(cov) >= -1e-10))
    True

Functions:
    rand_cov: Generate a random positive semi-definite covariance matrix

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

from .rand_cov import rand_cov as rand_cov

__version__ = importlib.metadata.version("cvxrisk")
