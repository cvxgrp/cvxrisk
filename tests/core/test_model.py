"""Tests for Model base class."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.core.model import Model
from cvx.core.variable import Variable


class _ConcreteModel(Model):
    """Minimal subclass that does not override solve_minrisk."""

    def estimate(self, weights, **kwargs):
        return 0.0

    def update(self, **kwargs):
        pass


def test_solve_minrisk_not_implemented() -> None:
    """Base Model.solve_minrisk raises NotImplementedError for subclasses that don't override it."""
    model = _ConcreteModel()
    weights = Variable(2)
    with pytest.raises(NotImplementedError, match="does not implement solve_minrisk"):
        model.solve_minrisk(weights, np.zeros(2), [])
