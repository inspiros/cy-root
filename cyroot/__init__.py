"""
A simple root-finding package written in Cython.
"""

from ._version import __version__
from .bracketing import *
from .newton import *
from .quasi_newton import *
from .scalar_root import *
from .vector_root import *
