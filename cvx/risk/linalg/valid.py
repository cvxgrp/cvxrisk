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
    """
    Extract the valid subset of a matrix by removing rows/columns with non-finite values.

    This function identifies rows and columns in a matrix that contain non-finite values
    (NaN or infinity) on the diagonal and removes them, returning both the indicator
    vector and the resulting valid submatrix.

    Args:
        matrix: An n x n matrix to be validated

    Returns:
        A tuple containing:
            - Boolean vector indicating which rows/columns are valid
            - The valid submatrix with invalid rows/columns removed

    Raises:
        AssertionError: If the input matrix is not square
    """
    # make sure matrix  is quadratic
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    v = np.isfinite(np.diag(matrix))
    return v, matrix[:, v][v]
