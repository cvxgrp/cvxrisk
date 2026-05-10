"""Core building blocks for cvxrisk.

Exposes :class:`Variable` and :class:`Parameter`, the two fundamental types
shared across all risk models and portfolio optimization problems.

"""

from .parameter import Parameter as Parameter
from .variable import Variable as Variable
