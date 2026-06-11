"""Conditional Value at Risk (CVaR) models for portfolio optimization.

This subpackage provides the CVar class for CVaR-based risk estimation.
CVaR, also known as Expected Shortfall, measures the expected loss in the
tail of the return distribution.

Example:
    >>> import numpy as np
    >>> from cvx.risk.cvar import CVar
    >>> # Create CVaR model
    >>> model = CVar(alpha=0.95, n=100, m=5)
    >>> # Update with historical returns
    >>> np.random.seed(42)
    >>> returns = np.random.randn(100, 5)
    >>> model.update(
    ...     returns=returns,
    ...     lower_assets=np.zeros(5),
    ...     upper_assets=np.ones(5)
    ... )
    >>> w = np.ones(5) / 5
    >>> cvar = model.estimate(w)
    >>> isinstance(cvar, float)
    True

"""

#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
from .cvar import CVar as CVar
