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
"""PCA analysis"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

PCA = namedtuple(
    "PCA",
    ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
)
"""
A named tuple containing the results of PCA analysis.

Attributes:
    explained_variance (numpy.ndarray): The explained variance ratio for each component
    factors (numpy.ndarray): The factor returns (principal components)
    exposure (pandas.DataFrame): The factor exposures (loadings) for each asset
    cov (pandas.DataFrame): The covariance matrix of the factors
    systematic (pandas.DataFrame): The systematic returns explained by the factors
    idiosyncratic (pandas.DataFrame): The idiosyncratic returns not explained by the factors
"""


def pca(returns: pd.DataFrame, n_components: int = 10) -> PCA:
    """
    Compute the first n principal components for a return matrix.

    Performs Principal Component Analysis (PCA) on the returns data to extract
    the most important factors that explain the variance in the returns.

    Args:
        returns: DataFrame of asset returns
        n_components: Number of principal components to extract. Defaults to 10.

    Returns:
        A named tuple containing the PCA results with the following fields:
            - explained_variance: The explained variance ratio for each component
            - factors: The factor returns (principal components)
            - exposure: The factor exposures (loadings) for each asset
            - cov: The covariance matrix of the factors
            - systematic: The systematic returns explained by the factors
            - idiosyncratic: The idiosyncratic returns not explained by the factors
    """

    # USING SKLEARN. Let's look at the first n components
    sklearn_pca = sklearnPCA(n_components=n_components)
    sklearn_pca.fit_transform(returns)

    exposure = sklearn_pca.components_
    factors = returns @ np.transpose(exposure)

    return PCA(
        explained_variance=sklearn_pca.explained_variance_ratio_,
        factors=factors,
        exposure=pd.DataFrame(data=exposure, columns=returns.columns),
        cov=factors.cov(),
        systematic=pd.DataFrame(data=factors.values @ exposure, index=returns.index, columns=returns.columns),
        idiosyncratic=pd.DataFrame(
            data=returns.values - factors.values @ exposure,
            index=returns.index,
            columns=returns.columns,
        ),
    )
