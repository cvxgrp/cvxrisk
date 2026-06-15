"""Contract tests for the core building blocks, pinned by mutation testing.

Each test here kills a mutant that survived the initial mutation-testing run:
the documented :class:`~cvx.core.parameter.Parameter` defaults, the
:class:`~cvx.core.bounds.Bounds` error messages and parameter names, and the
equality-constraint handling of
:class:`~cvx.core.conic.ConeProgramBuilder`.
"""

from __future__ import annotations

import clarabel
import numpy as np
import pytest

from cvx.core import Bounds, ConeProgramBuilder
from cvx.core.parameter import Parameter


def test_parameter_default_name_is_empty_string():
    """A Parameter created without a name defaults to "" (not a placeholder or None)."""
    assert Parameter(shape=3).name == ""


def test_parameter_value_is_not_a_constructor_argument():
    """``value`` is initialised in __post_init__, never accepted by the constructor."""
    with pytest.raises(TypeError):
        Parameter(shape=3, value=np.zeros(3))


def test_bounds_estimate_raises_with_documented_message():
    """``Bounds.estimate`` always raises NotImplementedError with its exact documented text."""
    bounds = Bounds(m=3, name="assets")
    with pytest.raises(NotImplementedError) as exc:
        bounds.estimate(np.zeros(3))
    assert str(exc.value) == "Bounds does not implement estimate"


def test_bounds_parameter_names_are_labelled():
    """The lower/upper bound parameters carry their human-readable names."""
    bounds = Bounds(m=3, name="assets")
    assert bounds.parameter["lower_assets"].name == "lower bound"
    assert bounds.parameter["upper_assets"].name == "upper bound"


def test_bounds_update_missing_key_message():
    """A missing bound key reports exactly which argument is required."""
    bounds = Bounds(m=3, name="assets")
    with pytest.raises(ValueError) as exc:
        bounds.update(upper_assets=np.ones(3))
    assert str(exc.value) == "update() requires a 'lower_assets' argument"


def test_bounds_update_too_long_message():
    """An over-long bound array reports its length and the maximum capacity, verbatim."""
    bounds = Bounds(m=3, name="assets")
    with pytest.raises(ValueError) as exc:
        bounds.update(lower_assets=np.zeros(4), upper_assets=np.ones(3))
    assert str(exc.value) == "'lower_assets' has length 4 but the maximum is 3"


def test_equality_linear_constraint_uses_a_single_zero_cone():
    """An equality constraint (lb == ub) becomes one ZeroConeT block, not two inequalities."""
    builder = ConeProgramBuilder(n_vars=2)
    builder.add_linear_constraints([(np.array([1.0, 1.0]), 0.5, 0.5)], cols=slice(0, 2))
    assert len(builder._cones) == 1
    assert isinstance(builder._cones[0], clarabel.ZeroConeT)


def test_two_sided_linear_constraint_uses_two_inequality_cones():
    """A genuine two-sided constraint (lb < ub) adds two NonnegativeConeT blocks."""
    builder = ConeProgramBuilder(n_vars=2)
    builder.add_linear_constraints([(np.array([1.0, 1.0]), 0.2, 0.8)], cols=slice(0, 2))
    assert len(builder._cones) == 2
    assert all(isinstance(cone, clarabel.NonnegativeConeT) for cone in builder._cones)
