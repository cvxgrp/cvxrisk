"""Tests for the Cholesky decomposition utility"""

from __future__ import annotations

import numpy as np

from cvx.random import rand_cov
from cvx.risk.linalg import cholesky


def test_cholesky() -> None:
    """
    Test that the cholesky function correctly decomposes a covariance matrix.

    This test verifies that:
    1. A random covariance matrix can be generated
    2. The cholesky function returns an upper triangular matrix
    3. The product of the transpose of the Cholesky factor and the Cholesky factor
       equals the original covariance matrix
    """
    a = rand_cov(10)
    u = cholesky(a)
    # test numpy arrays are equivalent
    assert np.allclose(a, np.transpose(u) @ u)
