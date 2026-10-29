"""Opt-in native-core preview. The callback and storage APIs are still drafts."""
from pathlib import Path
from .scalar import (
    Bisection, Newton, NativeCallbackError, ScalarCallback, ScalarStep, NewtonStep,
    bisect, newton,
)
from .damped_newton import DampedNewton, VectorDampedNewton, damped_newton, vector_damped_newton
from .jvp import JacobianVectorCallback, JacobianVectorPlan, JacobianVectorEstimate
from .vjp import JacobianTransposeCallback
from .newton_krylov import NewtonKrylov, NewtonKrylovResult, newton_krylov
from ._buffers import DoubleBuffer
from .scalar_derivatives import ScalarDerivativePlan, ScalarSecondDerivativePlan
from .higher_derivatives import ScalarFiniteDifferencePlan, ScalarDerivativeEstimate
from .higher_mixed import TensorDifferencePlan, TensorDerivativeEstimate
from .householder_solver import Householder, HouseholderStep, householder
from .secant import Secant, SecantStep, secant
from .homeier import Homeier, homeier
from .bracketing import (
    BracketSolver, BracketStep, regula_falsi, illinois, pegasus,
    anderson_bjorck, dekker, brent, chandrupatla, ridders, toms748, wu, itp,
)
from .hybisect import Hybisect, HybisectStep, IntervalCallback, hybisect
from .complex_callbacks import ComplexScalarCallback
from .differentiation import (
    AdaptiveHessian, NumericalHessianEstimate, AdaptiveSecondDerivative, ComplexStepDerivative, ComplexStepJacobian, AdaptiveDerivative, AdaptiveJacobian,
    ComplexVectorCallback, DerivativeEstimate, NumericalJacobianEstimate, DifferentiationError,
)
from .interpolation import (
    Sidi, Steffensen, InverseQuadraticInterpolation, HyperbolicInterpolation,
    Muller, InterpolationStep, sidi, steffensen, inverse_quadratic_interp,
    hyperbolic_interp, muller,
)
from .higher_order import (
    Halley, Chebyshev, ModifiedHalley, SuperHalley, TangentHyperbolas,
    HigherOrderStep, halley, chebyshev, modified_halley, super_halley,
    tangent_hyperbolas,
)
from .broyden import BroydenGood, BroydenBad, BroydenResult, BroydenStep, Barnes, Klement, barnes, broyden_good, broyden_bad, klement
from .rectangular import RectangularCallback
from .least_squares import LeastSquaresPlan, LeastSquaresResult
from .derivatives import JacobianPlan, RectangularJacobianPlan, JacobianEstimate
from .vector import WolfeBittner, WolfeBittnerResult, WolfeBittnerStep, wolfe_bittner, VectorNewton, VectorCallback, VectorNewtonStep, Robinson, TraubSteffensen, vector_newton, robinson, traub_steffensen
from .vrahatis import Vrahatis, VrahatisStep, vrahatis
from .hessians import HessianPlan, RectangularHessianPlan, HessianEstimate
from .levenberg_marquardt import LevenbergMarquardt, LevenbergMarquardtResult, levenberg_marquardt
from .bfgs import BFGS, BFGSResult, bfgs, LBFGS, lbfgs
from .levenberg_marquardt import PowellDogleg, PowellDoglegResult, powell_dogleg
from .objective import LeastSquaresObjective
from .objective_derivatives import ObjectiveDerivativePlan
from .tensor_callbacks import HessianCallback, RectangularHessianCallback
from .vector_chebyshev import (
    VectorChebyshev, VectorChebyshevStep, VectorHigherOrderStep, vector_chebyshev,
    VectorHalley, VectorSuperHalley, vector_halley, vector_super_halley,
    VectorModifiedHalley, vector_modified_halley,
)


def get_include():
    """Directory containing native headers for extension builds."""
    return str(Path(__file__).resolve().parent)


def get_cython_include():
    """Declaration search root, including editable installations."""
    return str(Path(__file__).resolve().parents[2])


__all__ = ['JacobianTransposeCallback', 'NewtonKrylov', 'NewtonKrylovResult', 'newton_krylov', 'JacobianVectorCallback', 'JacobianVectorPlan', 'JacobianVectorEstimate', 'DampedNewton', 'VectorDampedNewton', 'damped_newton', 'vector_damped_newton', 'AdaptiveHessian', 'NumericalHessianEstimate', 'AdaptiveSecondDerivative', 'ComplexStepDerivative', 'ComplexStepJacobian', 'AdaptiveDerivative', 'AdaptiveJacobian',
           'ComplexVectorCallback', 'DerivativeEstimate', 'NumericalJacobianEstimate', 'DifferentiationError', 'PowellDogleg', 'PowellDoglegResult', 'powell_dogleg', 'LBFGS', 'lbfgs', 'BFGS', 'BFGSResult', 'bfgs', 'LevenbergMarquardt', 'LevenbergMarquardtResult', 'levenberg_marquardt', 'LeastSquaresObjective','ObjectiveDerivativePlan','Vrahatis','VrahatisStep','vrahatis','Homeier','homeier','BracketSolver','BracketStep','regula_falsi',
           'illinois','pegasus','anderson_bjorck','dekker','brent','chandrupatla',
           'ridders','toms748','wu','itp','Hybisect','HybisectStep','IntervalCallback',
           'hybisect','ComplexScalarCallback','Sidi','Steffensen',
           'InverseQuadraticInterpolation','HyperbolicInterpolation','Muller',
           'InterpolationStep','sidi','steffensen','inverse_quadratic_interp',
           'hyperbolic_interp','muller','TensorDifferencePlan', 'TensorDerivativeEstimate', 'RectangularHessianCallback', 'RectangularHessianPlan', 'WolfeBittner', 'WolfeBittnerResult', 'WolfeBittnerStep', 'wolfe_bittner', 'Robinson', 'robinson', 'Barnes', 'barnes', 'TraubSteffensen', 'traub_steffensen', 'Klement', 'klement', 'RectangularJacobianPlan', 'JacobianEstimate', 'RectangularCallback', 'LeastSquaresPlan', 'LeastSquaresResult', 'BroydenGood', 'BroydenBad', 'BroydenResult', 'BroydenStep',
           'broyden_good', 'broyden_bad', 'Bisection', 'Newton', 'NativeCallbackError', 'ScalarCallback',
           'ScalarStep', 'NewtonStep', 'bisect', 'newton', 'DoubleBuffer',
           'get_include', 'get_cython_include', 'VectorNewton', 'VectorCallback',
           'VectorNewtonStep', 'vector_newton', 'JacobianPlan',
           'ScalarDerivativePlan', 'ScalarSecondDerivativePlan',
           'ScalarFiniteDifferencePlan', 'ScalarDerivativeEstimate',
           'HessianPlan', 'HessianEstimate',
           'HessianCallback', 'VectorChebyshev', 'VectorChebyshevStep',
           'vector_chebyshev',
           'VectorHigherOrderStep', 'VectorHalley', 'VectorSuperHalley',
           'vector_halley', 'vector_super_halley',
           'VectorModifiedHalley', 'vector_modified_halley',
           'Householder', 'HouseholderStep', 'householder',
           'Secant', 'SecantStep', 'secant',
           'Halley', 'Chebyshev', 'ModifiedHalley', 'SuperHalley',
           'TangentHyperbolas', 'HigherOrderStep', 'halley', 'chebyshev',
           'modified_halley', 'super_halley', 'tangent_hyperbolas']
