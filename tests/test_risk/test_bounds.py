"""Tests for the bounds module."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.risk.bounds import Bounds


def test_raise_not_implemented() -> None:
    """Test that the estimate method of Bounds raises NotImplementedError.

    This test verifies that calling the estimate method on a Bounds object
    raises a NotImplementedError, as this method is not implemented for Bounds.
    """
    bounds = Bounds(m=3, name="assets")

    with pytest.raises(NotImplementedError):
        bounds.estimate(np.zeros(3))


def test_constraints() -> None:
    """Test that the constraints method of Bounds returns the expected constraints.

    This test verifies that:
    1. The update method correctly sets the lower and upper bound parameters
    2. The get_bounds method returns the expected arrays
    """
    bounds = Bounds(m=3, name="assets")
    bounds.update(lower_assets=np.array([0.1, 0.2]), upper_assets=np.array([0.3, 0.4, 0.5]))

    assert bounds.parameter["lower_assets"].value == pytest.approx(np.array([0.1, 0.2, 0]))
    assert bounds.parameter["upper_assets"].value == pytest.approx(np.array([0.3, 0.4, 0.5]))

    lb, ub = bounds.get_bounds()
    assert lb == pytest.approx(np.array([0.1, 0.2, 0.0]))
    assert ub == pytest.approx(np.array([0.3, 0.4, 0.5]))
