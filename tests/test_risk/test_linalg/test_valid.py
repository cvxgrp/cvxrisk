from __future__ import annotations

import numpy as np
import pytest

from cvx.risk.linalg import valid


def test_valid():
    """
    Test that the valid function correctly identifies valid rows/columns in a matrix.

    This test verifies that:
    1. The valid function can process a matrix with NaN values
    2. The function returns the correct boolean vector indicating valid rows/columns
    3. The function returns the correct submatrix with invalid rows/columns removed
    """
    a = np.array([[np.nan, np.nan], [np.nan, 4]])
    v, mat = valid(a)

    assert np.allclose(mat, np.array([[4]]))
    assert np.allclose(v, np.array([False, True]))


def test_invalid():
    """
    Test that the valid function raises an AssertionError for non-square matrices.

    This test verifies that:
    1. The valid function checks that the input matrix is square
    2. The function raises an AssertionError when given a non-square matrix
    """
    a = np.zeros((3, 2))
    with pytest.raises(AssertionError):
        valid(a)
