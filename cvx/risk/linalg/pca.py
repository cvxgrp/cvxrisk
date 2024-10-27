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
"""PCA analysis
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA

PCA = namedtuple(
    "PCA",
    ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
)


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

    exposure = sklearn_pca.components_
    factors = returns @ np.transpose(exposure)

    return PCA(
        explained_variance=sklearn_pca.explained_variance_ratio_,
        factors=factors,
        exposure=pd.DataFrame(data=exposure, columns=returns.columns),
        cov=factors.cov(),
        systematic=pd.DataFrame(
            data=factors.values @ exposure, index=returns.index, columns=returns.columns
        ),
        idiosyncratic=pd.DataFrame(
            data=returns.values - factors.values @ exposure,
            index=returns.index,
            columns=returns.columns,
        ),
    )
