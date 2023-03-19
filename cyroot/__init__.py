"""
A simple root-finding package written in Cython.
"""

from ._version import __version__

# Uncomment this for automatic compiling during development
# from ._cython_extension import _has_ext
from ._scalar_root import *
from .bracketing import *
from .newton import *
from .quasi_newton import *
