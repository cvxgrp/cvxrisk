"""Additional tests to reach 100% coverage for factor.py.

These tests target error branches in FactorModel.update that were previously
uncovered in the coverage report.
"""
from __future__ import annotations

import numpy as np
import pytest

from cvx.risk.factor import FactorModel


def test_update_raises_when_too_many_factors() -> None:
    """update should raise ValueError if provided more factors than the model's max k."""
    # Model can handle at most k=2 factors and 3 assets
    model = FactorModel(assets=3, k=2)

    # Provide exposure with 3 factors (> k) and 2 assets (<= assets)
    exposure = np.zeros((3, 2))

    with pytest.raises(ValueError, match="Number of factors exceeds"):
        model.update(
            exposure=exposure,
            cov=np.eye(3),  # consistent with number of factors in exposure
            idiosyncratic_risk=np.zeros(2),
            lower_assets=np.zeros(2),
            upper_assets=np.ones(2),
            lower_factors=-np.ones(3),
            upper_factors=np.ones(3),
        )


def test_update_raises_when_too_many_assets() -> None:
    """update should raise ValueError if provided more assets than the model's max assets."""
    # Model can handle at most 2 factors and 3 assets
    model = FactorModel(assets=3, k=2)

    # Provide exposure with 2 factors (<= k) and 4 assets (> assets)
    exposure = np.zeros((2, 4))

    with pytest.raises(ValueError, match="Number of assets exceeds"):
        model.update(
            exposure=exposure,
            cov=np.eye(2),  # consistent with number of factors in exposure
            idiosyncratic_risk=np.zeros(4),
            lower_assets=np.zeros(4),
            upper_assets=np.ones(4),
            lower_factors=-np.ones(2),
            upper_factors=np.ones(2),
        )
