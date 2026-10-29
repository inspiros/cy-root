"""Static-only examples: supported reexports and typed numerical controls/results."""
from typing import assert_type
from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
import cyroot
from cyroot.utils._function_registering import register
from cyroot.scalar_newton import ahmadi, AhmadiDeltaFuncFactory
from cyroot.vector_bracketing import compute_admissible_n_polygon, get_M, sorted_by_vertices
from cyroot import (
status as public_status,
    TerminationReason as public_TerminationReason,
    utils as public_utils,
    scalar_bracketing as public_scalar_bracketing,
    bisect as public_bisect,
    hybisect as public_hybisect,
    regula_falsi as public_regula_falsi,
    illinois as public_illinois,
    pegasus as public_pegasus,
    anderson_bjorck as public_anderson_bjorck,
    dekker as public_dekker,
    brent as public_brent,
    chandrupatla as public_chandrupatla,
    ridders as public_ridders,
    toms748 as public_toms748,
    wu as public_wu,
    itp as public_itp,
    fptr as public_fptr,
    scalar_derivative_approximation as public_scalar_derivative_approximation,
    DerivativeApproximation as public_DerivativeApproximation,
    finite_difference as public_finite_difference,
    FiniteDifference as public_FiniteDifference,
    return_types as public_return_types,
    vector_derivative_approximation as public_vector_derivative_approximation,
    native as public_native,
    scalar_newton as public_scalar_newton,
    newton as public_newton,
    homeier as public_homeier,
    halley as public_halley,
    super_halley as public_super_halley,
    chebyshev as public_chebyshev,
    tangent_hyperbolas as public_tangent_hyperbolas,
    householder as public_householder,
    scalar_quasi_newton as public_scalar_quasi_newton,
    secant as public_secant,
    sidi as public_sidi,
    steffensen as public_steffensen,
    inverse_quadratic_interp as public_inverse_quadratic_interp,
    hyperbolic_interp as public_hyperbolic_interp,
    muller as public_muller,
    optimization as public_optimization,
    scalar_root as public_scalar_root,
    SCALAR_ROOT_FINDING_METHODS as public_SCALAR_ROOT_FINDING_METHODS,
    find_scalar_root as public_find_scalar_root,
    vector_bracketing as public_vector_bracketing,
    vrahatis as public_vrahatis,
    VectorDerivativeApproximation as public_VectorDerivativeApproximation,
    GeneralizedFiniteDifference as public_GeneralizedFiniteDifference,
    generalized_finite_difference as public_generalized_finite_difference,
    vector_newton as public_vector_newton,
    generalized_newton as public_generalized_newton,
    generalized_halley as public_generalized_halley,
    generalized_super_halley as public_generalized_super_halley,
    generalized_chebyshev as public_generalized_chebyshev,
    generalized_tangent_hyperbolas as public_generalized_tangent_hyperbolas,
    vector_quasi_newton as public_vector_quasi_newton,
    wolfe_bittner as public_wolfe_bittner,
    robinson as public_robinson,
    barnes as public_barnes,
    traub_steffensen as public_traub_steffensen,
    broyden as public_broyden,
    klement as public_klement,
    vector_root as public_vector_root,
    VECTOR_ROOT_FINDING_METHODS as public_VECTOR_ROOT_FINDING_METHODS,
    find_vector_root as public_find_vector_root,
    PowellDogleg as public_PowellDogleg,
    PowellDoglegResult as public_PowellDoglegResult,
    powell_dogleg as public_powell_dogleg,
    LBFGS as public_LBFGS,
    lbfgs as public_lbfgs,
    BFGS as public_BFGS,
    BFGSResult as public_BFGSResult,
    bfgs as public_bfgs,
    LevenbergMarquardt as public_LevenbergMarquardt,
    LevenbergMarquardtResult as public_LevenbergMarquardtResult,
    levenberg_marquardt as public_levenberg_marquardt,
    LeastSquaresObjective as public_LeastSquaresObjective,
    ObjectiveDerivativePlan as public_ObjectiveDerivativePlan,
    OptimizationRootResult as public_OptimizationRootResult,
    root_via_optimizer as public_root_via_optimizer,
    OptimizationResult as public_OptimizationResult,
    OptimizationSolver as public_OptimizationSolver,
    find_optimization_root as public_find_optimization_root,
    OPTIMIZATION_ROOT_FINDING_METHODS as public_OPTIMIZATION_ROOT_FINDING_METHODS,
    scalar_levenberg_marquardt as public_scalar_levenberg_marquardt,
    vector_levenberg_marquardt as public_vector_levenberg_marquardt,
    scalar_bfgs as public_scalar_bfgs,
    vector_bfgs as public_vector_bfgs,
    scalar_lbfgs as public_scalar_lbfgs,
    vector_lbfgs as public_vector_lbfgs,
    scalar_powell_dogleg as public_scalar_powell_dogleg,
    vector_powell_dogleg as public_vector_powell_dogleg
)
from cyroot.native import (
PowellDogleg as native_PowellDogleg,
    PowellDoglegResult as native_PowellDoglegResult,
    powell_dogleg as native_powell_dogleg,
    LBFGS as native_LBFGS,
    lbfgs as native_lbfgs,
    BFGS as native_BFGS,
    BFGSResult as native_BFGSResult,
    bfgs as native_bfgs,
    LevenbergMarquardt as native_LevenbergMarquardt,
    LevenbergMarquardtResult as native_LevenbergMarquardtResult,
    levenberg_marquardt as native_levenberg_marquardt,
    LeastSquaresObjective as native_LeastSquaresObjective,
    ObjectiveDerivativePlan as native_ObjectiveDerivativePlan,
    Vrahatis as native_Vrahatis,
    VrahatisStep as native_VrahatisStep,
    vrahatis as native_vrahatis,
    Homeier as native_Homeier,
    homeier as native_homeier,
    BracketSolver as native_BracketSolver,
    BracketStep as native_BracketStep,
    regula_falsi as native_regula_falsi,
    illinois as native_illinois,
    pegasus as native_pegasus,
    anderson_bjorck as native_anderson_bjorck,
    dekker as native_dekker,
    brent as native_brent,
    chandrupatla as native_chandrupatla,
    ridders as native_ridders,
    toms748 as native_toms748,
    wu as native_wu,
    itp as native_itp,
    Hybisect as native_Hybisect,
    HybisectStep as native_HybisectStep,
    IntervalCallback as native_IntervalCallback,
    hybisect as native_hybisect,
    ComplexScalarCallback as native_ComplexScalarCallback,
    Sidi as native_Sidi,
    Steffensen as native_Steffensen,
    InverseQuadraticInterpolation as native_InverseQuadraticInterpolation,
    HyperbolicInterpolation as native_HyperbolicInterpolation,
    Muller as native_Muller,
    InterpolationStep as native_InterpolationStep,
    sidi as native_sidi,
    steffensen as native_steffensen,
    inverse_quadratic_interp as native_inverse_quadratic_interp,
    hyperbolic_interp as native_hyperbolic_interp,
    muller as native_muller,
    TensorDifferencePlan as native_TensorDifferencePlan,
    TensorDerivativeEstimate as native_TensorDerivativeEstimate,
    RectangularHessianCallback as native_RectangularHessianCallback,
    RectangularHessianPlan as native_RectangularHessianPlan,
    WolfeBittner as native_WolfeBittner,
    WolfeBittnerResult as native_WolfeBittnerResult,
    WolfeBittnerStep as native_WolfeBittnerStep,
    wolfe_bittner as native_wolfe_bittner,
    Robinson as native_Robinson,
    robinson as native_robinson,
    Barnes as native_Barnes,
    barnes as native_barnes,
    TraubSteffensen as native_TraubSteffensen,
    traub_steffensen as native_traub_steffensen,
    Klement as native_Klement,
    klement as native_klement,
    RectangularJacobianPlan as native_RectangularJacobianPlan,
    JacobianEstimate as native_JacobianEstimate,
    RectangularCallback as native_RectangularCallback,
    LeastSquaresPlan as native_LeastSquaresPlan,
    LeastSquaresResult as native_LeastSquaresResult,
    BroydenGood as native_BroydenGood,
    BroydenBad as native_BroydenBad,
    BroydenResult as native_BroydenResult,
    BroydenStep as native_BroydenStep,
    broyden_good as native_broyden_good,
    broyden_bad as native_broyden_bad,
    Bisection as native_Bisection,
    Newton as native_Newton,
    NativeCallbackError as native_NativeCallbackError,
    ScalarCallback as native_ScalarCallback,
    ScalarStep as native_ScalarStep,
    NewtonStep as native_NewtonStep,
    bisect as native_bisect,
    newton as native_newton,
    DoubleBuffer as native_DoubleBuffer,
    get_include as native_get_include,
    get_cython_include as native_get_cython_include,
    VectorNewton as native_VectorNewton,
    VectorCallback as native_VectorCallback,
    VectorNewtonStep as native_VectorNewtonStep,
    vector_newton as native_vector_newton,
    JacobianPlan as native_JacobianPlan,
    ScalarDerivativePlan as native_ScalarDerivativePlan,
    ScalarSecondDerivativePlan as native_ScalarSecondDerivativePlan,
    ScalarFiniteDifferencePlan as native_ScalarFiniteDifferencePlan,
    ScalarDerivativeEstimate as native_ScalarDerivativeEstimate,
    HessianPlan as native_HessianPlan,
    HessianEstimate as native_HessianEstimate,
    HessianCallback as native_HessianCallback,
    VectorChebyshev as native_VectorChebyshev,
    VectorChebyshevStep as native_VectorChebyshevStep,
    vector_chebyshev as native_vector_chebyshev,
    VectorHigherOrderStep as native_VectorHigherOrderStep,
    VectorHalley as native_VectorHalley,
    VectorSuperHalley as native_VectorSuperHalley,
    vector_halley as native_vector_halley,
    vector_super_halley as native_vector_super_halley,
    VectorModifiedHalley as native_VectorModifiedHalley,
    vector_modified_halley as native_vector_modified_halley,
    Householder as native_Householder,
    HouseholderStep as native_HouseholderStep,
    householder as native_householder,
    Secant as native_Secant,
    SecantStep as native_SecantStep,
    secant as native_secant,
    Halley as native_Halley,
    Chebyshev as native_Chebyshev,
    ModifiedHalley as native_ModifiedHalley,
    SuperHalley as native_SuperHalley,
    TangentHyperbolas as native_TangentHyperbolas,
    HigherOrderStep as native_HigherOrderStep,
    halley as native_halley,
    chebyshev as native_chebyshev,
    modified_halley as native_modified_halley,
    super_halley as native_super_halley,
    tangent_hyperbolas as native_tangent_hyperbolas
)
from cyroot.optimization import (
PowellDogleg as optimization_PowellDogleg,
    PowellDoglegResult as optimization_PowellDoglegResult,
    powell_dogleg as optimization_powell_dogleg,
    LBFGS as optimization_LBFGS,
    lbfgs as optimization_lbfgs,
    BFGS as optimization_BFGS,
    BFGSResult as optimization_BFGSResult,
    bfgs as optimization_bfgs,
    LevenbergMarquardt as optimization_LevenbergMarquardt,
    LevenbergMarquardtResult as optimization_LevenbergMarquardtResult,
    levenberg_marquardt as optimization_levenberg_marquardt,
    LeastSquaresObjective as optimization_LeastSquaresObjective,
    ObjectiveDerivativePlan as optimization_ObjectiveDerivativePlan,
    OptimizationRootResult as optimization_OptimizationRootResult,
    root_via_optimizer as optimization_root_via_optimizer,
    OptimizationResult as optimization_OptimizationResult,
    OptimizationSolver as optimization_OptimizationSolver,
    find_optimization_root as optimization_find_optimization_root,
    OPTIMIZATION_ROOT_FINDING_METHODS as optimization_OPTIMIZATION_ROOT_FINDING_METHODS,
    scalar_levenberg_marquardt as optimization_scalar_levenberg_marquardt,
    vector_levenberg_marquardt as optimization_vector_levenberg_marquardt,
    scalar_bfgs as optimization_scalar_bfgs,
    vector_bfgs as optimization_vector_bfgs,
    scalar_lbfgs as optimization_scalar_lbfgs,
    vector_lbfgs as optimization_vector_lbfgs,
    scalar_powell_dogleg as optimization_scalar_powell_dogleg,
    vector_powell_dogleg as optimization_vector_powell_dogleg
)

def scalar(x: float) -> float:
    return x*x-2.

def vector(x: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(v-1. for v in x)

class Initial:
    def __float__(self) -> float:
        return 1.5

class Budget:
    def __index__(self) -> int:
        return 10

analytic = cyroot.newton(scalar, lambda x: 2*x, Initial(), max_iter=Budget())
assert_type(analytic.success, bool)
assert_type(analytic.iters, int)
assert_type(analytic.error, float)
cyroot.bisect(scalar, 1., 2., algo=1).bracket
assert_type(cyroot.finite_difference(scalar, 1.5, kind='central'), float)
assert_type(cyroot.FiniteDifference(scalar)(1.5), float)
assert_type(cyroot.generalized_finite_difference(vector, (0.,)), NDArray[np.float64])
assert_type(cyroot.GeneralizedFiniteDifference(vector)((0.,)), NDArray[np.float64])
cyroot.sidi(scalar, (1., 1.5, 2.), f_x0s=(-1., .25, 2.))
cyroot.muller(lambda z: z*z+1., -1+0j, 0j, 1+0j, f_x0=2+0j)
cyroot.vrahatis(vector, (0.,), h=(1.,))
cyroot.broyden(vector, (0.,), J_x0=cyroot.native.JacobianPlan(1), algo='good')
cyroot.robinson(vector, (0.,), (2.,), orth=1)
quasi = cyroot.broyden(vector, (0.,))
assert_type(quasi.root, tuple[float, ...])
assert_type(quasi.approximation, tuple[tuple[float, ...], ...])
assert_type(quasi.success, bool)
assert_type(cyroot.wolfe_bittner(vector, ((0.,), (2.,))).affine_error, float)
state = cyroot.as_root_snapshot(analytic)
assert_type(state, cyroot.RootSnapshot)
assert_type(state.root, cyroot.RootValue)
assert_type(state.f_calls, tuple[int, ...])
assert_type(state.termination_reason, cyroot.TerminationReason | None)
assert_type(cyroot.as_root_snapshot(quasi), cyroot.RootSnapshot)
assert_type(cyroot.scalar_bfgs(scalar, None, 1.5), cyroot.OptimizationResult)
assert_type(cyroot.vector_lbfgs(vector, None, (0.,)), cyroot.OptimizationResult)
assert_type(cyroot.as_root_snapshot(cyroot.scalar_powell_dogleg(scalar, None, 1.5)), cyroot.RootSnapshot)
assert_type(cyroot.hybisect(scalar, lambda a,b: (a,b), 1., 2.).iters, list[int])
registered: Callable[[float], float] = register('tag', wrap=True)(scalar)
assert_type(registered, Callable[[float], float])

legacy_fields = tuple(analytic)

assert_type(cyroot.__version__, str)
