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
"""Abstract risk model.

This module provides the abstract base class for all risk models in cvxrisk.
Risk models are used to estimate portfolio risk and build portfolio optimization
problems solved directly with the Clarabel solver.

Example:
    All risk models inherit from the Model class and must implement
    the estimate and update methods:

    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> # Create a sample covariance risk model
    >>> model = SampleCovariance(num=3)
    >>> # Update the model with a covariance matrix
    >>> cov = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
    >>> model.update(cov=cov, lower_assets=np.zeros(3), upper_assets=np.ones(3))
    >>> # Evaluate risk for given weights
    >>> weights = np.array([1/3, 1/3, 1/3])
    >>> risk = model.estimate(weights)
    >>> isinstance(risk, float)
    True

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .parameter import Parameter


@dataclass
class Model(ABC):
    """Abstract base class for risk models.

    This class defines the interface that all risk models must implement.
    Risk models are used in portfolio optimization to estimate portfolio risk.
    The underlying optimization problems are solved directly with Clarabel.

    Attributes:
        parameter: Dictionary mapping parameter names to :class:`~cvx.risk.parameter.Parameter`
            objects. These parameters can be updated between solver calls without
            reconstructing the optimization problem.

    Example:
        Subclasses must implement the abstract methods:

        >>> import numpy as np
        >>> from cvx.risk.sample import SampleCovariance
        >>> model = SampleCovariance(num=2)
        >>> model.update(
        ...     cov=np.array([[1.0, 0.5], [0.5, 2.0]]),
        ...     lower_assets=np.zeros(2),
        ...     upper_assets=np.ones(2)
        ... )
        >>> # Access model parameters
        >>> 'chol' in model.parameter
        True

        The parameter dictionary holds Parameter objects that can be
        updated without reconstructing the optimization problem:

        >>> param = model.parameter['chol']
        >>> from cvx.risk.parameter import Parameter
        >>> isinstance(param, Parameter)
        True
        >>> param.shape
        (2, 2)

    """

    parameter: dict[str, Parameter] = field(default_factory=dict)
    """Dictionary of parameters for the risk model."""

    @abstractmethod
    def estimate(self, weights: np.ndarray, **kwargs: Any) -> float:
        """Estimate the risk given the portfolio weights.

        This method evaluates the risk measure for the given portfolio weights
        and returns a scalar float value.

        Args:
            weights: Numpy array representing portfolio weights.
            **kwargs: Additional keyword arguments specific to the risk model.

        Returns:
            Float representing the estimated risk (e.g., standard deviation).

        Example:
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=2)
            >>> model.update(
            ...     cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
            ...     lower_assets=np.zeros(2),
            ...     upper_assets=np.ones(2)
            ... )
            >>> risk = model.estimate(np.array([0.5, 0.5]))
            >>> isinstance(risk, float)
            True

        """

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update the data in the risk model.

        This method updates the parameters in the model with new data.
        Because the parameters are stored as numpy arrays, updating them
        allows solving new problem instances without reconstructing the problem.

        Args:
            **kwargs: Keyword arguments containing data to update the model.
                The specific arguments depend on the risk model implementation.

        Example:
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=3)
            >>> # Update with new covariance data
            >>> cov = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
            >>> model.update(
            ...     cov=cov,
            ...     lower_assets=np.zeros(3),
            ...     upper_assets=np.ones(3)
            ... )

        """
