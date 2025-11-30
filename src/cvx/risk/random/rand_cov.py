"""Random covariance matrix generation utilities.

This module provides functions for generating random positive semi-definite
covariance matrices. These are useful for testing and simulation purposes.

Example:
    Generate a random covariance matrix:

    >>> import numpy as np
    >>> from cvx.risk.random import rand_cov
    >>> # Generate a 5x5 random covariance matrix
    >>> cov = rand_cov(5, seed=42)
    >>> cov.shape
    (5, 5)
    >>> # Verify it's symmetric
    >>> bool(np.allclose(cov, cov.T))
    True
    >>> # Verify it's positive semi-definite
    >>> bool(np.all(np.linalg.eigvals(cov) >= -1e-10))
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
from __future__ import annotations

import numpy as np


def rand_cov(n: int, seed: int | None = None) -> np.ndarray:
    """Construct a random positive semi-definite covariance matrix of size n x n.

    The matrix is constructed as A^T @ A where A is a random n x n matrix with
    elements drawn from a standard normal distribution. This ensures the result
    is symmetric and positive semi-definite.

    Args:
        n: Size of the covariance matrix (n x n).
        seed: Random seed for reproducibility. If None, uses the current
            random state.

    Returns:
        A random positive semi-definite n x n covariance matrix.

    Example:
        Generate a reproducible random covariance matrix:

        >>> import numpy as np
        >>> from cvx.risk.random import rand_cov
        >>> cov1 = rand_cov(3, seed=42)
        >>> cov2 = rand_cov(3, seed=42)
        >>> np.allclose(cov1, cov2)
        True

        Use in portfolio optimization:

        >>> from cvx.risk.sample import SampleCovariance
        >>> import cvxpy as cp
        >>> model = SampleCovariance(num=4)
        >>> cov = rand_cov(4, seed=42)
        >>> model.update(
        ...     cov=cov,
        ...     lower_assets=np.zeros(4),
        ...     upper_assets=np.ones(4)
        ... )
        >>> weights = cp.Variable(4)
        >>> risk = model.estimate(weights)
        >>> isinstance(risk, cp.Expression)
        True

    Note:
        The generated matrix is guaranteed to be positive semi-definite because
        it is constructed as A^T @ A. In practice, it will typically be positive
        definite (all eigenvalues strictly positive) unless n is very large.

    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return np.transpose(a) @ a
