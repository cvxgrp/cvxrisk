"""Core building blocks shared across cvx packages.

Exposes the fundamental types used to build parametric optimization models:
:class:`Model`, :class:`Bounds`, :class:`Parameter`, and :class:`Variable`,
plus the :class:`ConeProgramBuilder` helper for assembling Clarabel problems.

"""

from .bounds import Bounds as Bounds
from .conic import ConeProgramBuilder as ConeProgramBuilder
from .model import Model as Model
from .parameter import Parameter as Parameter
from .variable import Variable as Variable
