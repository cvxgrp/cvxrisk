"""Sample covariance risk models for portfolio optimization.

This subpackage provides the SampleCovariance class for risk estimation
based on the sample covariance matrix.

Example:
    >>> import numpy as np
    >>> from cvx.risk.sample import SampleCovariance
    >>> model = SampleCovariance(num=3)
    >>> model.update(
    ...     cov=np.eye(3),
    ...     lower_assets=np.zeros(3),
    ...     upper_assets=np.ones(3)
    ... )
    >>> risk = model.estimate(np.array([1/3, 1/3, 1/3]))
    >>> isinstance(risk, float)
    True

Classes:
    SampleCovariance: Risk model based on sample covariance matrix

"""

#    Copyright (c) 2025 Jebel Quant Research
#
#    Licensed under the MIT License. See the LICENSE file in the project root
#    for the full license text.
from .sample import SampleCovariance as SampleCovariance
