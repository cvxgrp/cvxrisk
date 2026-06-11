"""Factor risk models for portfolio optimization.

This subpackage provides factor-based risk models for portfolio optimization.
Factor models decompose portfolio risk into systematic (factor) risk and
idiosyncratic (residual) risk.

Example:
    >>> import numpy as np
    >>> from cvx.risk.factor import FactorModel
    >>> # Create factor model with 5 assets and 2 factors
    >>> model = FactorModel(assets=5, k=2)
    >>> np.random.seed(42)
    >>> model.update(
    ...     exposure=np.random.randn(2, 5),
    ...     cov=np.eye(2),
    ...     idiosyncratic_risk=np.abs(np.random.randn(5)),
    ...     lower_assets=np.zeros(5),
    ...     upper_assets=np.ones(5),
    ...     lower_factors=-np.ones(2),
    ...     upper_factors=np.ones(2)
    ... )
    >>> w = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> risk = model.estimate(w)
    >>> isinstance(risk, float)
    True

"""

#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
from .factor import FactorModel as FactorModel
