from __future__ import annotations

import numpy as np

from cvx.random import rand_cov
from cvx.risk.linalg import cholesky


def test_cholesky():
    a = rand_cov(10)
    u = cholesky(a)
    # test numpy arrays are equivalent
    assert np.allclose(a, np.transpose(u) @ u)
