from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest

from cvx.risk.bounds import Bounds


def test_raise_not_implemented():
    weights = cp.Variable(3)
    bounds = Bounds(m=3, name="assets")

    with pytest.raises(NotImplementedError):
        bounds.estimate(weights)


def test_constraints():
    weights = cp.Variable(3)
    bounds = Bounds(m=3, name="assets")
    bounds.update(lower_assets=np.array([0.1, 0.2]), upper_assets=np.array([0.3, 0.4, 0.5]))

    assert bounds.parameter["lower_assets"].value == pytest.approx(np.array([0.1, 0.2, 0]))
    assert bounds.parameter["upper_assets"].value == pytest.approx(np.array([0.3, 0.4, 0.5]))

    assert len(bounds.constraints(weights)) == 2
