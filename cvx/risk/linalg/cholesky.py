"""Cholesky decomposition utilities for covariance matrices."""

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
from scipy.linalg import cholesky as _cholesky


def cholesky(cov: np.ndarray) -> np.ndarray:
    """Compute the upper triangular part of the Cholesky decomposition.

    This function computes the Cholesky decomposition of a positive definite matrix.
    It returns the upper triangular matrix R such that R^T R = cov.

    Args:
        cov: A positive definite covariance matrix

    Returns:
        The upper triangular Cholesky factor

    Note:
        This uses scipy.linalg.cholesky which returns the upper triangular part,
        unlike numpy.linalg.cholesky which returns the lower triangular part.

    """
    return _cholesky(cov)
