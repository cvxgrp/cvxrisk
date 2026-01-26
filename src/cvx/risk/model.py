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
Risk models are used to estimate portfolio risk and provide constraints for
portfolio optimization problems.

Example:
    All risk models inherit from the Model class and must implement
    the estimate, update, and constraints methods:

    >>> import cvxpy as cp
    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> # Create a sample covariance risk model
    >>> model = SampleCovariance(num=3)
    >>> # Update the model with a covariance matrix
    >>> cov = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.5], [0.0, 0.5, 1.0]])
    >>> model.update(cov=cov, lower_assets=np.zeros(3), upper_assets=np.ones(3))
    >>> # Create a weights variable and estimate risk
    >>> weights = cp.Variable(3)
    >>> risk_expr = model.estimate(weights)
    >>> isinstance(risk_expr, cp.Expression)
    True

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import cvxpy as cp


@dataclass
class Model(ABC):
    """Abstract base class for risk models.

    This class defines the interface that all risk models must implement.
    Risk models are used in portfolio optimization to estimate portfolio risk
    and provide constraints for the optimization problem.

    Attributes:
        parameter: Dictionary mapping parameter names to CVXPY Parameter objects.
            These parameters can be updated without reconstructing the optimization problem.

    Example:
        Subclasses must implement the abstract methods:

        >>> import cvxpy as cp
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

        The parameter dictionary holds CVXPY Parameter objects that can be
        updated without reconstructing the optimization problem:

        >>> param = model.parameter['chol']
        >>> isinstance(param, cp.Parameter)
        True
        >>> param.shape
        (2, 2)

        Multiple risk models can be composed by combining their constraints:

        >>> from cvx.risk.bounds import Bounds
        >>> extra_bounds = Bounds(m=2, name="extra")
        >>> extra_bounds.update(
        ...     lower_extra=np.array([0.2, 0.2]),
        ...     upper_extra=np.array([0.8, 0.8])
        ... )
        >>> weights = cp.Variable(2)
        >>> all_constraints = model.constraints(weights) + extra_bounds.constraints(weights)
        >>> len(all_constraints)
        4

    """

    parameter: dict[str, cp.Parameter] = field(default_factory=dict)
    """Dictionary of CVXPY parameters for the risk model."""

    @abstractmethod
    def estimate(self, weights: cp.Variable, **kwargs) -> cp.Expression:
        """Estimate the variance given the portfolio weights.

        This method returns a CVXPY expression representing the risk measure
        for the given portfolio weights. The expression can be used as an
        objective function in a convex optimization problem.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments specific to the risk model.

        Returns:
            CVXPY expression representing the estimated risk (e.g., standard deviation).

        Example:
            >>> import cvxpy as cp
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=2)
            >>> model.update(
            ...     cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
            ...     lower_assets=np.zeros(2),
            ...     upper_assets=np.ones(2)
            ... )
            >>> weights = cp.Variable(2)
            >>> risk = model.estimate(weights)
            >>> isinstance(risk, cp.Expression)
            True

        """

    @abstractmethod
    def update(self, **kwargs) -> None:
        """Update the data in the risk model.

        This method updates the CVXPY parameters in the model with new data.
        Because CVXPY supports parametric optimization, updating parameters
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

    @abstractmethod
    def constraints(self, weights: cp.Variable, **kwargs) -> list[cp.Constraint]:
        """Return the constraints for the risk model.

        This method returns a list of CVXPY constraints that should be included
        in the portfolio optimization problem. Common constraints include bounds
        on asset weights.

        Args:
            weights: CVXPY variable representing portfolio weights.
            **kwargs: Additional keyword arguments specific to the risk model.

        Returns:
            List of CVXPY constraints for the risk model.

        Example:
            >>> import cvxpy as cp
            >>> import numpy as np
            >>> from cvx.risk.sample import SampleCovariance
            >>> model = SampleCovariance(num=2)
            >>> model.update(
            ...     cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
            ...     lower_assets=np.zeros(2),
            ...     upper_assets=np.ones(2)
            ... )
            >>> weights = cp.Variable(2)
            >>> constraints = model.constraints(weights)
            >>> len(constraints) == 2  # Lower and upper bounds
            True

        """
