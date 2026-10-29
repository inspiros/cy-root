from .status import TerminationReason as TerminationReason
from .native.damped_newton import DampedNewton as DampedNewton, VectorDampedNewton as VectorDampedNewton
from ._version import __version__ as __version__
from .scalar_bracketing import *
from .scalar_derivative_approximation import *
from .scalar_newton import *
from .scalar_quasi_newton import *
from .scalar_root import *
from .vector_bracketing import *
from .vector_derivative_approximation import *
from .vector_newton import *
from .vector_quasi_newton import *
from .vector_root import *

from .optimization import *
from .results import RootValue as RootValue, SnapshotSource as SnapshotSource, RootSnapshot as RootSnapshot, as_root_snapshot as as_root_snapshot

from typing import Any
fptr: Any

from . import (native as native, return_types as return_types,
               optimization as optimization, results as results, status as status, utils as utils)
from . import scalar_bracketing as scalar_bracketing
from . import scalar_derivative_approximation as scalar_derivative_approximation
from . import scalar_newton as scalar_newton
from . import scalar_quasi_newton as scalar_quasi_newton
from . import scalar_root as scalar_root
from . import vector_bracketing as vector_bracketing
from . import vector_derivative_approximation as vector_derivative_approximation
from . import vector_newton as vector_newton
from . import vector_quasi_newton as vector_quasi_newton
from . import vector_root as vector_root

from .native.differentiation import (
    AdaptiveHessian as AdaptiveHessian, NumericalHessianEstimate as NumericalHessianEstimate,
    AdaptiveSecondDerivative as AdaptiveSecondDerivative,
    ComplexStepDerivative as ComplexStepDerivative, ComplexStepJacobian as ComplexStepJacobian,
    AdaptiveDerivative as AdaptiveDerivative, AdaptiveJacobian as AdaptiveJacobian,
    DerivativeEstimate as DerivativeEstimate, NumericalJacobianEstimate as NumericalJacobianEstimate,
    DifferentiationError as DifferentiationError,
)

from .native.jvp import (JacobianVectorCallback as JacobianVectorCallback,
                         JacobianVectorPlan as JacobianVectorPlan,
                         JacobianVectorEstimate as JacobianVectorEstimate)

from .native.vjp import JacobianTransposeCallback as JacobianTransposeCallback
from .native.newton_krylov import (NewtonKrylov as NewtonKrylov, NewtonKrylovResult as NewtonKrylovResult, newton_krylov as newton_krylov)
