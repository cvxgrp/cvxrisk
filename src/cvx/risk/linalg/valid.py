"""Matrix validation utilities for handling non-finite values.

This module provides functions for validating and cleaning matrices that may
contain non-finite values (NaN or infinity). This is particularly useful when
working with financial data where missing values are common.

Example:
    Extract the valid submatrix from a covariance matrix with missing data:

    >>> import numpy as np
    >>> from cvx.risk.linalg import valid
    >>> # Create a covariance matrix with some NaN values on diagonal
    >>> cov = np.array([[np.nan, 0.5, 0.2],
    ...                 [0.5, 2.0, 0.3],
    ...                 [0.2, 0.3, np.nan]])
    >>> # Get valid indicator and submatrix
    >>> v, submatrix = valid(cov)
    >>> v  # Second row/column is valid
    array([False,  True, False])
    >>> submatrix
    array([[2.]])

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


def valid(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract the valid subset of a matrix by removing rows/columns with non-finite values.

    This function identifies rows and columns in a square matrix that contain
    non-finite values (NaN or infinity) on the diagonal and removes them,
    returning both the indicator vector and the resulting valid submatrix.

    This is useful when working with covariance matrices where some assets
    may have missing or invalid data.

    Args:
        matrix: A square n x n matrix to be validated. Typically a covariance
            or correlation matrix.

    Returns:
        A tuple containing:
            - v: Boolean vector of shape (n,) indicating which rows/columns are
              valid (True for valid, False for invalid).
            - submatrix: The valid submatrix with invalid rows/columns removed.
              Shape is (k, k) where k is the number of True values in v.

    Raises:
        AssertionError: If the input matrix is not square (n x n).

    Example:
        Basic usage with a covariance matrix:

        >>> import numpy as np
        >>> from cvx.risk.linalg import valid
        >>> # Create a 3x3 matrix with one invalid entry
        >>> cov = np.array([[1.0, 0.5, 0.2],
        ...                 [0.5, np.nan, 0.3],
        ...                 [0.2, 0.3, 1.0]])
        >>> v, submatrix = valid(cov)
        >>> v
        array([ True, False,  True])
        >>> submatrix
        array([[1. , 0.2],
               [0.2, 1. ]])

        Handling a fully valid matrix:

        >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        >>> v, submatrix = valid(cov)
        >>> v
        array([ True,  True])
        >>> np.allclose(submatrix, cov)
        True

    Note:
        The function checks only the diagonal elements for validity. It assumes
        that if the diagonal is finite, the entire row/column is valid. This is
        a common assumption for covariance matrices.

    """
    # make sure matrix  is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    v = np.isfinite(np.diag(matrix))
    return v, matrix[:, v][v]
