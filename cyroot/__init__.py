"""
A simple root-finding package written in Cython.
"""

from .status import TerminationReason
from ._version import __version__
from .scalar_bracketing import *
from .scalar_newton import *
from .scalar_quasi_newton import *
from .scalar_root import *
from .vector_bracketing import *
from .vector_newton import *
from .vector_quasi_newton import *
from .vector_root import *

from .optimization import *
from .results import RootValue, SnapshotSource, RootSnapshot, as_root_snapshot

from .native.differentiation import (
    AdaptiveHessian, NumericalHessianEstimate, AdaptiveSecondDerivative, ComplexStepDerivative, ComplexStepJacobian, AdaptiveDerivative, AdaptiveJacobian,
    DerivativeEstimate, NumericalJacobianEstimate, DifferentiationError,
)
from .native.damped_newton import DampedNewton, VectorDampedNewton
from .native.jvp import JacobianVectorCallback, JacobianVectorPlan, JacobianVectorEstimate
from .native.vjp import JacobianTransposeCallback
from .native.newton_krylov import NewtonKrylov, NewtonKrylovResult, newton_krylov
