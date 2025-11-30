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
"""PCA analysis (pure NumPy implementation)."""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd

PCA = namedtuple(
    "PCA",
    ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
)
"""
A named tuple containing the results of PCA analysis.

Attributes:
    explained_variance (numpy.ndarray): Explained variance ratio for each component
    factors (pandas.DataFrame): Factor returns (principal components)
    exposure (pandas.DataFrame): Factor exposures (loadings) for each asset
    cov (pandas.DataFrame): Covariance matrix of the factors
    systematic (pandas.DataFrame): Systematic returns explained by the factors
    idiosyncratic (pandas.DataFrame): Idiosyncratic returns not explained by the factors
"""


def pca(returns: pd.DataFrame, n_components: int = 10) -> PCA:
    """Compute the first n principal components for a return matrix using SVD.

    Args:
        returns: DataFrame of asset returns (rows: time, columns: assets)
        n_components: Number of principal components to extract. Defaults to 10.

    Returns:
        PCA named tuple with the results.
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
