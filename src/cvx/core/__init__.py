"""Core building blocks shared across cvx packages.

Exposes the fundamental types used to build parametric optimization models:
:class:`Model`, :class:`Bounds`, :class:`Parameter`, and :class:`Variable`.

"""

from .bounds import Bounds as Bounds
from .model import Model as Model
from .parameter import Parameter as Parameter
from .variable import Variable as Variable
