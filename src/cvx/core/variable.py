#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
"""Decision variable class for portfolio optimization.

This module provides the Variable class, which acts as a placeholder for
decision variables in portfolio optimization problems. After calling
:func:`~cvx.risk.portfolio.min_risk.minrisk_problem` and solving, the
``value`` attribute is populated with the optimal solution.

Example:
    Create a variable and use it in an optimization problem:

    >>> import numpy as np
    >>> from cvx.core.variable import Variable
    >>> w = Variable(3)
    >>> w.n
    3
    >>> w.value is None
    True

"""

from __future__ import annotations

from dataclasses import dataclass, field

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
        >>> from cvx.core.variable import Variable
        >>> w = Variable(4)
        >>> w.n
        4
        >>> w.value is None
        True

    """

    n: int
    """Dimension of the variable."""

    value: np.ndarray | None = field(default=None, init=False)  # pragma: no mutate
    """Optimal value set after solving, or ``None`` before solving."""
