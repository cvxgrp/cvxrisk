"""Tests for the Bounds class in cvx.core."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.core.bounds import Bounds


def test_bounds_initialization() -> None:
    """Test default initialization of Bounds parameters."""
    bounds = Bounds(m=5, name="test")

    assert bounds.m == 5
    assert bounds.name == "test"
    assert "lower_test" in bounds.parameter
    assert "upper_test" in bounds.parameter
    assert bounds.parameter["lower_test"].value.shape == (5,)
    assert bounds.parameter["upper_test"].value.shape == (5,)
    assert np.all(bounds.parameter["lower_test"].value == 0)
    assert np.all(bounds.parameter["upper_test"].value == 1)


def test_constraints() -> None:
    """Test that get_bounds returns correctly padded lower/upper arrays."""
    bounds = Bounds(m=3, name="assets")
    bounds.update(lower_assets=np.array([0.1, 0.2]), upper_assets=np.array([0.3, 0.4, 0.5]))

    assert bounds.parameter["lower_assets"].value == pytest.approx(np.array([0.1, 0.2, 0]))
    assert bounds.parameter["upper_assets"].value == pytest.approx(np.array([0.3, 0.4, 0.5]))

    lb, ub = bounds.get_bounds()
    assert lb == pytest.approx(np.array([0.1, 0.2, 0.0]))
    assert ub == pytest.approx(np.array([0.3, 0.4, 0.5]))


def test_estimate_raises() -> None:
    """Test that estimate() raises NotImplementedError."""
    bounds = Bounds(m=3, name="assets")
    with pytest.raises(NotImplementedError):
        bounds.estimate(np.zeros(3))
