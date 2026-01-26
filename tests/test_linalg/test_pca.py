"""Tests for the Principal Component Analysis (PCA) utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cvx.risk.linalg import pca


@pytest.fixture
def returns(resource_dir) -> pd.DataFrame:
    """Pytest fixture that provides stock return data for testing.

    This fixture loads stock price data from a CSV file, calculates returns
    using percentage change, and fills any NaN values with zeros.

    Args:
        resource_dir: Pytest fixture providing the path to the test resources directory

    Returns:
        pandas.DataFrame: DataFrame containing stock returns

    """
    prices = pd.read_csv(resource_dir / "stock_prices.csv", index_col=0, header=0, parse_dates=True)
    return prices.pct_change().fillna(0.0)


def test_pca(returns: pd.DataFrame) -> None:
    """Test that the pca function correctly calculates the principal components.

    This test verifies that:
    1. The pca function can process returns data
    2. The explained variance ratios match the expected values

    Args:
        returns: Pytest fixture providing stock return data

    """
    xxx = pca(returns)

    assert np.allclose(
        xxx.explained_variance,
        np.array(
            [
                0.33383825,
                0.19039608,
                0.11567561,
                0.07965253,
                0.06379108,
                0.04580062,
                0.03461307,
                0.02205145,
                0.01876345,
                0.01757195,
            ]
        ),
    )
