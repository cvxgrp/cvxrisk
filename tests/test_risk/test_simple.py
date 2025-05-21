from __future__ import annotations

import numpy as np

from cvx.risk.bounds import Bounds


def test_bounds_initialization():
    """Test that Bounds can be initialized with default parameters."""
    bounds = Bounds(m=5, name="test")

    # Check that the parameters are initialized correctly
    assert bounds.m == 5
    assert bounds.name == "test"

    # Check that the parameter dictionary contains the expected keys
    assert "lower_test" in bounds.parameter
    assert "upper_test" in bounds.parameter

    # Check that the parameters have the expected values
    assert bounds.parameter["lower_test"].value.shape == (5,)
    assert bounds.parameter["upper_test"].value.shape == (5,)
    assert np.all(bounds.parameter["lower_test"].value == 0)
    assert np.all(bounds.parameter["upper_test"].value == 1)
