# -*- coding: utf-8 -*-
"""PCA analysis
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
from sklearn.decomposition import PCA as sklearnPCA

PCA = namedtuple("PCA", ["explained_variance", "component", "returns"])


def pca(returns, n_components=10):
    """
    Compute the first n principal components for a return matrix

    Args:
        returns: DataFrame of prices
        n_components: Number of components
    """

    # USING SKLEARN. Let's look at the first n components
    sklearn_pca = sklearnPCA(n_components=n_components)
    sklearn_pca.fit_transform(returns)

    return PCA(
        explained_variance=sklearn_pca.explained_variance_ratio_,
        component=sklearn_pca.components_,
        returns=returns @ np.transpose(sklearn_pca.components_),
    )
