"""Cholesky decomposition utilities for covariance matrices.

This module provides a function to compute the upper triangular Cholesky
decomposition of a positive definite covariance matrix.

Example:
    Compute the Cholesky decomposition of a covariance matrix:

    >>> import numpy as np
    >>> from cvx.risk.linalg import cholesky
    >>> # Create a positive definite matrix
    >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> # Compute upper triangular Cholesky factor
    >>> R = cholesky(cov)
    >>> # Verify: R.T @ R = cov
    >>> np.allclose(R.T @ R, cov)
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
from numpy.linalg import cholesky as _cholesky


def cholesky(cov: np.ndarray) -> np.ndarray:
    """Compute the upper triangular part of the Cholesky decomposition.

    This function computes the Cholesky decomposition of a positive definite matrix
    and returns the upper triangular matrix R such that R^T @ R = cov.

    The Cholesky decomposition is useful in portfolio optimization because it
    provides an efficient way to compute portfolio risk as ||R @ w||_2, where
    w is the portfolio weights vector.

    Args:
        cov: A positive definite covariance matrix of shape (n, n).

    Returns:
        The upper triangular Cholesky factor R of shape (n, n).

    Example:
        Basic usage with a simple covariance matrix:

        >>> import numpy as np
        >>> from cvx.risk.linalg import cholesky
        >>> # Identity matrix
        >>> cov = np.eye(3)
        >>> R = cholesky(cov)
        >>> np.allclose(R, np.eye(3))
        True

        With a more complex covariance matrix:

        >>> cov = np.array([[1.0, 0.5, 0.0],
        ...                 [0.5, 1.0, 0.5],
        ...                 [0.0, 0.5, 1.0]])
        >>> R = cholesky(cov)
        >>> np.allclose(R.T @ R, cov)
        True

    Note:
        This function returns the upper triangular factor (R), whereas
        numpy.linalg.cholesky returns the lower triangular factor (L).
        The relationship is: L @ L^T = cov and R^T @ R = cov, where R = L^T.

    """
    return _cholesky(cov).transpose()
