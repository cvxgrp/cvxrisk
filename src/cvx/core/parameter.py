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
"""Parameter class for risk models.

This module provides a simple parameter class that stores a named numpy array
value that can be updated without reconstructing the optimization problem.

Example:
    Create a parameter and update its value:

    >>> import numpy as np
    >>> from cvx.core.parameter import Parameter
    >>> p = Parameter(shape=3, name="weights")
    >>> p.value
    array([0., 0., 0.])
    >>> p.value = np.array([0.5, 0.3, 0.2])
    >>> p.value
    array([0.5, 0.3, 0.2])

"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Parameter:
    """A named parameter holding a mutable numpy array value.

    Parameters are used in risk models to store matrices and vectors
    (such as Cholesky factors, factor exposures, and bounds) that can
    be updated between solver calls without rebuilding the problem structure.

    Attributes:
        shape: The shape of the parameter. Use an integer for 1-D parameters
            and a tuple for 2-D parameters.
        name: A human-readable name for the parameter.
        value: The numpy array holding the current parameter value.

    Example:
        1-D parameter (e.g., lower bounds):

        >>> import numpy as np
        >>> from cvx.core.parameter import Parameter
        >>> p = Parameter(shape=4, name="lower_assets")
        >>> p.value.shape
        (4,)
        >>> p.value = np.array([0.0, 0.1, 0.0, 0.2])
        >>> p.value[1]
        np.float64(0.1)

        2-D parameter (e.g., Cholesky factor):

        >>> p2 = Parameter(shape=(3, 3), name="chol")
        >>> p2.value.shape
        (3, 3)
        >>> import numpy as np
        >>> p2.value = np.eye(3)
        >>> p2.value[0, 0]
        np.float64(1.0)

    """

    shape: int | tuple[int, ...]
    """Shape of the parameter (int for 1-D, tuple for 2-D)."""

    name: str = ""
    """Human-readable name for the parameter."""

    value: np.ndarray = field(init=False)
    """Current value of the parameter as a numpy array."""

    def __post_init__(self) -> None:
        """Initialise the value array to zeros."""
        self.value = np.zeros(self.shape)
