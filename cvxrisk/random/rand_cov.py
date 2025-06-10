"""Random covariance matrix generation utilities."""

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
from __future__ import annotations

import numpy as np


def rand_cov(n: int, seed: int | None = None) -> np.ndarray:
    """Construct a random positive semi-definite covariance matrix of size n x n.

    The matrix is constructed as A^T A where A is a random n x n matrix with
    elements drawn from a standard normal distribution.

    Args:
        n: Size of the covariance matrix
        seed: Random seed for reproducibility (optional)

    Returns:
        A random positive semi-definite n x n covariance matrix

    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return np.transpose(a) @ a
