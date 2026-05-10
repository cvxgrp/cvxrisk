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
"""Decision variable class for portfolio optimization.

This module provides the Variable class, which acts as a placeholder for
decision variables in portfolio optimization problems. After calling
:func:`~cvx.risk.portfolio.min_risk.minrisk_problem` and solving, the
``value`` attribute is populated with the optimal solution.

Example:
    Create a variable and use it in an optimization problem:

    >>> import numpy as np
    >>> from cvx.risk.variable import Variable
    >>> w = Variable(3)
    >>> w.n
    3
    >>> w.value is None
    True

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Variable:
    """A decision variable for portfolio optimization.

    Acts as a placeholder whose ``value`` attribute is populated with
    the optimal solution once the problem has been solved.

    Attributes:
        n: Dimension of the variable (number of assets or factors).
        value: Optimal solution populated by the solver, or ``None`` before
            the problem has been solved.

    Example:
        >>> from cvx.risk.variable import Variable
        >>> w = Variable(4)
        >>> w.n
        4
        >>> w.value is None
        True

    """

    n: int
    """Dimension of the variable."""

    value: Optional[np.ndarray] = field(default=None, init=False)
    """Optimal value set after solving, or ``None`` before solving."""
