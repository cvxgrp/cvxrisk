"""Tests for the random covariance matrix generation utilities."""

from __future__ import annotations

import numpy as np

from cvx.random import rand_cov


def test_rand_cov() -> None:
    """Test that the rand_cov function generates a positive definite matrix.

    This test verifies that:
    1. The rand_cov function can generate a random covariance matrix
    2. The generated matrix is positive definite (all eigenvalues are positive)
    """
    a = rand_cov(5)
    assert np.all(np.linalg.eigvals(a) > 0)
