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
"""Risk models based on the sample covariance matrix
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np

from ..bounds import Bounds
from ..linalg import cholesky
from ..model import Model


@dataclass
class SampleCovariance(Model):
    """Risk model based on the Cholesky decomposition of the sample cov matrix"""

    num: int = 0

    def __post_init__(self):
        self.parameter["chol"] = cvx.Parameter(
            shape=(self.num, self.num),
            name="cholesky of covariance",
            value=np.zeros((self.num, self.num)),
        )
        self.bounds = Bounds(m=self.num, name="assets")

    def estimate(self, weights, **kwargs):
        """Estimate the risk by computing the Cholesky decomposition of self.cov"""
        return cvx.norm2(self.parameter["chol"] @ weights)

    def update(self, **kwargs):
        cov = kwargs["cov"]
        n = cov.shape[0]

        self.parameter["chol"].value[:n, :n] = cholesky(cov)
        self.bounds.update(**kwargs)

    def constraints(self, weights):
        return self.bounds.constraints(weights)
