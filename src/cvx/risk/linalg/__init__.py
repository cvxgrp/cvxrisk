"""Linear algebra utilities for risk models.

This subpackage provides linear algebra utilities commonly used in risk modeling,
including Cholesky decomposition, Principal Component Analysis, and matrix
validation.

Example:
    >>> import numpy as np
    >>> from cvx.risk.linalg import cholesky, pca, valid
    >>> # Cholesky decomposition
    >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> R = cholesky(cov)
    >>> np.allclose(R.T @ R, cov)
    True

Functions:
    cholesky: Compute upper triangular Cholesky decomposition
    pca: Compute principal components of return data
    valid: Extract valid submatrix from a matrix with NaN values

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
from .cholesky import cholesky as cholesky  # noqa: F401
from .pca import pca as pca  # noqa: F401
from .valid import valid as valid  # noqa: F401
