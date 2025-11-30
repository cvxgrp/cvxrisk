#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""PCA analysis (pure NumPy implementation).

This module provides Principal Component Analysis (PCA) for dimensionality
reduction of return data. PCA is commonly used to construct factor models
for portfolio optimization.

Example:
    Perform PCA on stock returns:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from cvx.risk.linalg import pca
    >>> # Create sample returns data
    >>> np.random.seed(42)
    >>> returns = pd.DataFrame(
    ...     np.random.randn(100, 5),
    ...     columns=['A', 'B', 'C', 'D', 'E']
    ... )
    >>> # Compute PCA with 3 components
    >>> result = pca(returns, n_components=3)
    >>> # Access explained variance
    >>> len(result.explained_variance)
    3
    >>> # Access factors (principal components)
    >>> result.factors.shape
    (100, 3)
    >>> # Access factor exposures (loadings)
    >>> result.exposure.shape
    (3, 5)

"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd

PCA = namedtuple(
    "PCA",
    ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
)
"""Named tuple containing the results of PCA analysis.

Attributes:
    explained_variance: Explained variance ratio for each component.
        An array of shape (n_components,) where each element represents
        the proportion of total variance explained by that component.
    factors: Factor returns (principal components) as a DataFrame.
        Shape is (n_samples, n_components). Each column is a factor.
    exposure: Factor exposures (loadings) for each asset as a DataFrame.
        Shape is (n_components, n_assets). Each row contains the loadings
        of one component on all assets.
    cov: Covariance matrix of the factors as a DataFrame.
        Shape is (n_components, n_components).
    systematic: Systematic returns explained by the factors as a DataFrame.
        Shape is (n_samples, n_assets). This is the part of returns
        explained by the factor model.
    idiosyncratic: Idiosyncratic returns not explained by factors as a DataFrame.
        Shape is (n_samples, n_assets). This is the residual part of returns.

Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> from cvx.risk.linalg import pca
    >>> np.random.seed(42)
    >>> returns = pd.DataFrame(np.random.randn(50, 4))
    >>> result = pca(returns, n_components=2)
    >>> # Check explained variance sums to less than 1
    >>> result.explained_variance.sum() < 1
    True
    >>> # Systematic + idiosyncratic approximately equals original
    >>> np.allclose(
    ...     result.systematic.values + result.idiosyncratic.values,
    ...     returns.values,
    ...     atol=1e-10
    ... )
    True

"""


def pca(returns: pd.DataFrame, n_components: int = 10) -> PCA:
    """Compute the first n principal components for a return matrix using SVD.

    This function performs Principal Component Analysis on asset returns to
    extract the main sources of variance. The results can be used to construct
    a factor model for portfolio optimization.

    Args:
        returns: DataFrame of asset returns with shape (n_samples, n_assets).
            Rows represent time periods, columns represent assets.
        n_components: Number of principal components to extract. Defaults to 10.

    Returns:
        PCA named tuple containing:
            - explained_variance: Ratio of variance explained by each component
            - factors: Factor returns (scores)
            - exposure: Factor exposures (loadings)
            - cov: Factor covariance matrix
            - systematic: Returns explained by factors
            - idiosyncratic: Residual returns

    Example:
        Basic PCA on synthetic returns:

        >>> import numpy as np
        >>> import pandas as pd
        >>> from cvx.risk.linalg import pca
        >>> np.random.seed(42)
        >>> # Create returns with 100 periods and 10 assets
        >>> returns = pd.DataFrame(np.random.randn(100, 10))
        >>> result = pca(returns, n_components=3)
        >>> # First component explains most variance
        >>> bool(result.explained_variance[0] > result.explained_variance[1])
        True
        >>> # Factors are orthogonal
        >>> factor_corr = np.corrcoef(result.factors.T)
        >>> bool(np.allclose(factor_corr, np.eye(3), atol=0.1))
        True

        Using PCA results for a factor model:

        >>> from cvx.risk.factor import FactorModel
        >>> import cvxpy as cp
        >>> model = FactorModel(assets=10, k=3)
        >>> model.update(
        ...     exposure=result.exposure.values,
        ...     cov=result.cov.values,
        ...     idiosyncratic_risk=result.idiosyncratic.std().values,
        ...     lower_assets=np.zeros(10),
        ...     upper_assets=np.ones(10),
        ...     lower_factors=-np.ones(3),
        ...     upper_factors=np.ones(3)
        ... )

    """
    # Demean the returns
    x = returns.to_numpy()
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    # Singular Value Decomposition
    # x = u s V^T, where columns of V are principal axes
    u, s_full, vt = np.linalg.svd(x_centered, full_matrices=False)

    # Take only the first n components
    u = u[:, :n_components]
    s = s_full[:n_components]
    vt = vt[:n_components, :]

    # Factor exposures (loadings): each component's weight per asset
    exposure = pd.DataFrame(vt, columns=returns.columns)

    # Factor returns (scores): projection of data onto components
    factors = pd.DataFrame(u * s, index=returns.index, columns=[f"PC{i + 1}" for i in range(n_components)])

    # Explained variance ratio (normalize by total variance across ALL components)
    explained_variance = (s**2) / np.sum(s_full**2)

    # Covariance of factor returns
    cov = factors.cov()

    # Systematic + Idiosyncratic returns
    systematic = pd.DataFrame(
        data=(u * s) @ vt + x_mean,
        index=returns.index,
        columns=returns.columns,
    )
    idiosyncratic = pd.DataFrame(
        data=x_centered - (u * s) @ vt,
        index=returns.index,
        columns=returns.columns,
    )

    return PCA(
        explained_variance=explained_variance,
        factors=factors,
        exposure=exposure,
        cov=cov,
        systematic=systematic,
        idiosyncratic=idiosyncratic,
    )
