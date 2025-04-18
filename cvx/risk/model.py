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
"""Abstract risk model"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cvxpy as cp


@dataclass
class Model(ABC):
    """Abstract risk model"""

    parameter: dict[str, cp.Parameter] = field(default_factory=dict)
    """parameter for the riskmodel"""

    @abstractmethod
    def estimate(self, weights, **kwargs):
        """
        Estimate the variance given the portfolio weights
        """

    @abstractmethod
    def update(self, **kwargs):
        """
        Update the data in the risk model
        """

    @abstractmethod
    def constraints(self, weights, **kwargs):
        """
        Return the constraints for the risk model
        """
