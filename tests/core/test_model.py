"""Tests for Model base class."""

from __future__ import annotations

from cvx.core.model import Model


class _ConcreteModel(Model):
    """Minimal subclass that does not override solve_minrisk."""

    def estimate(self, weights, **kwargs):
        return 0.0

    def update(self, **kwargs):
        pass
