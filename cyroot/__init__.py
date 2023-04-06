"""
A simple root-finding package written in Cython.
"""

from ._defaults import set_default_stop_condition_args
from ._version import __version__
from .scalar_bracketing import *
from .scalar_newton import *
from .scalar_quasi_newton import *
from .scalar_root import *
from .vector_newton import *
from .vector_quasi_newton import *
from .vector_root import *
