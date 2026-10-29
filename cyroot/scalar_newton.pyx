# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Sequence, Optional, Union

import cython
import numpy as np
cimport numpy as np
from cython cimport view
from libc cimport math
from libcpp.vector cimport vector

from ._check_args import _check_stop_cond_args
from ._types import VectorLike
from ._stopping cimport (
    StopState, init_stop, advance_x, check_stop, finish_stop, successful, residual_met,
    RUNNING, RESIDUAL_TOLERANCE, MAX_ITERATIONS, ZERO_DERIVATIVE, BREAKDOWN, NONFINITE,
)
from .fptr cimport (
    DoubleScalarFPtr, PyDoubleScalarFPtr,
    DoubleVectorReductionFPtr, PyDoubleVectorReductionFPtr,
)
from .scalar_derivative_approximation import DerivativeApproximation, FiniteDifference
from .return_types cimport NewtonMethodsReturnType
from .utils._function_registering import register

cdef extern from 'defaults.hpp' namespace 'cyroot' nogil:
    double ETOL, ERTOL, PTOL, PRTOL, FINITE_DIFF_STEP
    size_t MAX_ITER

__all__ = [
    'newton',
    'homeier',
    'halley',
    'super_halley',
    'chebyshev',
    'tangent_hyperbolas',
    'householder',
]

# Numerical plan creation can invoke user coercions. Reserve explicit callback
# adapters throughout that preparation, before the native constructor reserves
# them for its own conversions, initialization, and numerical run.
from .native.scalar cimport ScalarCallback
from .native.scalar_derivatives import ScalarDerivativePlan, ScalarSecondDerivativePlan


def _providers(f, df, d2f, h, second=False, formula=None, validate_formula=False):
    cdef ScalarCallback callback
    function = f if isinstance(f, ScalarCallback) else ScalarCallback(f)
    derivative = df
    second_derivative = d2f
    callbacks = [function]
    if df is not None and not isinstance(df, ScalarDerivativePlan):
        derivative = df if isinstance(df, ScalarCallback) else ScalarCallback(df)
        callbacks.append(derivative)
    if second and d2f is not None and not isinstance(d2f, ScalarSecondDerivativePlan):
        second_derivative = d2f if isinstance(d2f, ScalarCallback) else ScalarCallback(d2f)
        callbacks.append(second_derivative)
    for callback in callbacks:
        if callback._active:
            raise RuntimeError('This callback adapter is already active.')
    for callback in callbacks:
        callback._active = True
    try:
        if validate_formula and formula not in (1, 2):
            raise ValueError(f'Unknown formula {formula}.')
        if h is not None and derivative is not None and not (second and second_derivative is None):
            # A silently unused step usually means a misplaced provider.
            raise ValueError('h applies only to numerical derivatives; '
                             'configure an explicit derivative plan directly.')
        if derivative is None:
            derivative = ScalarDerivativePlan(step=h)
        if second and second_derivative is None:
            second_derivative = ScalarSecondDerivativePlan(step=h)
    finally:
        for callback in callbacks:
            callback._active = False
    return function, derivative, second_derivative


@register('cyroot.scalar.newton')
@cython.binding(True)
def newton(f, df, x0, f_x0=None, df_x0=None, *, h=None,
           etol=ETOL, ertol=ERTOL, ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER, **options):
    """Newton on the native engine; h=None selects automatic derivative steps.

    Residuals and analytic derivatives accept Python or native ScalarCallback
    adapters. Use cyroot.native.Newton to retain copied steps/history.
    """
    from .native.scalar import newton as solve
    f, df, _ = _providers(f, df, None, h)
    return solve(f, df, x0, f_x0=f_x0, df_x0=df_x0,
                 etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter, **options)


@register('cyroot.scalar.newton')
@cython.binding(True)
def homeier(f, df, x0, f_x0=None, df_x0=None, *, h=None,
            etol=ETOL, ertol=ERTOL, ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER, **options):
    """Homeier's two-point harmonic-mean Newton on the native engine."""
    from .native.homeier import homeier as solve
    f, df, _ = _providers(f, df, None, h)
    return solve(f, df, x0, f_x0=f_x0, df_x0=df_x0,
                 etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter, **options)


@register('cyroot.scalar.newton')
@cython.binding(True)
def halley(f, df, d2f, x0, f_x0=None, df_x0=None, d2f_x0=None,
           *, alpha=None, h=None, etol=ETOL, ertol=ERTOL, ptol=PTOL,
           prtol=PRTOL, max_iter=MAX_ITER, **options):
    """Native Halley, or its modified family when alpha is supplied."""
    from .native.higher_order import halley as solve, modified_halley
    f, df, d2f = _providers(f, df, d2f, h, second=True)
    native_options = dict(f_x0=f_x0, df_x0=df_x0, d2f_x0=d2f_x0,
                   etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter, **options)
    if alpha is None:
        return solve(f, df, d2f, x0, **native_options)
    return modified_halley(f, df, d2f, x0, alpha=alpha, **native_options)


@register('cyroot.scalar.newton')
@cython.binding(True)
def super_halley(f, df, d2f, x0, f_x0=None, df_x0=None, d2f_x0=None,
                 *, h=None, etol=ETOL, ertol=ERTOL, ptol=PTOL,
                 prtol=PRTOL, max_iter=MAX_ITER, **options):
    """Native Super-Halley, the modified Halley family with alpha=1."""
    return halley(f, df, d2f, x0, f_x0=f_x0, df_x0=df_x0, d2f_x0=d2f_x0,
                  alpha=1., h=h, etol=etol, ertol=ertol, ptol=ptol,
                  prtol=prtol, max_iter=max_iter, **options)


@register('cyroot.scalar.newton')
@cython.binding(True)
def chebyshev(f, df, d2f, x0, f_x0=None, df_x0=None, d2f_x0=None,
              *, h=None, etol=ETOL, ertol=ERTOL, ptol=PTOL,
              prtol=PRTOL, max_iter=MAX_ITER, **options):
    """Native Chebyshev, the modified Halley family with alpha=0."""
    return halley(f, df, d2f, x0, f_x0=f_x0, df_x0=df_x0, d2f_x0=d2f_x0,
                  alpha=0., h=h, etol=etol, ertol=ertol, ptol=ptol,
                  prtol=prtol, max_iter=max_iter, **options)


@register('cyroot.scalar.newton')
@cython.binding(True)
def tangent_hyperbolas(f, df, d2f, x0, f_x0=None, df_x0=None, d2f_x0=None,
                       *, formula=2, h=None, etol=ETOL, ertol=ERTOL,
                       ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER, **options):
    """Native tangent hyperbolas; formulas 1/2 share the equivalent Halley step.

    The formula selector is retained and validated for existing callers. Both
    scalar formulas are algebraically identical; the native ratio evaluation
    avoids unnecessary overflow in their derivative products.
    """
    from .native.higher_order import tangent_hyperbolas as solve
    f, df, d2f = _providers(f, df, d2f, h, second=True,
                           formula=formula, validate_formula=True)
    return solve(f, df, d2f, x0, f_x0=f_x0, df_x0=df_x0, d2f_x0=d2f_x0,
                 etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter, **options)


@register('cyroot.scalar.newton')
@cython.binding(True)
def householder(f, dfs, x0, f_x0=None, dfs_x0=None, *, d=None, h=None,
                etol=ETOL, ertol=ERTOL, ptol=PTOL, prtol=PRTOL,
                max_iter=MAX_ITER, **options):
    """Run native runtime-order Householder using the established entry point.

    ``d`` maps to native ``order`` and ``h`` to ``step``. When ``dfs`` is
    supplied, ``d`` must agree with its length. ``h=None`` chooses automatic
    numerical steps. Use ``cyroot.native.Householder`` for stepping/history.
    All arguments remain untyped so native callback guards cover conversion.
    """
    from .native.householder_solver import householder as native_householder
    if h is not None and isinstance(dfs, (list, tuple)) and all(df is not None for df in dfs):
        # Native Householder ignores an unused step; this entry point matches
        # the other Newton-type wrappers and rejects it.
        raise ValueError('h applies only to numerical derivatives; '
                         'configure an explicit derivative plan directly.')
    return native_householder(
        f, dfs, x0, f_x0=f_x0, dfs_x0=dfs_x0, order=d, step=h,
        etol=etol, ertol=ertol, ptol=ptol, prtol=prtol, max_iter=max_iter, **options)


cdef bint _initial_scalar_stopped(double x0, double f_x0,
                                  double etol, double ertol,
                                  double ptol, double prtol) noexcept nogil:
    cdef StopState stop
    init_stop(&stop, math.fabs(f_x0), math.fabs(x0), etol, ertol, ptol, prtol)
    return check_stop(&stop, math.fabs(f_x0), math.INFINITY, False)


################################################################################
# Ahmadi
################################################################################
# noinspection DuplicatedCode
cdef NewtonMethodsReturnType ahmadi_kernel(
        DoubleScalarFPtr[:] fs,
        DoubleVectorReductionFPtr delta_f,
        double x0,
        double[:] fs_x0,
        size_t d,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        size_t max_iter=MAX_ITER):
    cdef size_t step = 0
    cdef double precision, error
    cdef bint converged, optimal
    cdef vector[double] dfs_x0 = vector[double](fs_x0.shape[0] - 1)
    cdef size_t i
    for i in range(1, d + 1):
        dfs_x0[i - 1] = fs_x0[i]
    cdef StopState stop
    precision = math.INFINITY
    error = math.fabs(fs_x0[0])
    init_stop(&stop, error, math.fabs(x0), etol, ertol, ptol, prtol)
    if check_stop(&stop, error, precision, False):
        return NewtonMethodsReturnType(
            x0, fs_x0[0], tuple(dfs_x0[i] for i in range(len(dfs_x0))), step,
            tuple(fs[i].n_f_calls for i in range(len(fs))),
            precision, error, successful(stop.reason),
            stop.reason == RESIDUAL_TOLERANCE, termination_reason=<int>stop.reason)

    cdef bint[:] use_derivative_approximation = view.array(shape=(fs_x0.shape[0] - 1,),
                                                           itemsize=sizeof(int),
                                                           format='i')
    for i in range(1, d + 1):
        use_derivative_approximation[i - 1] = isinstance(fs[i], DerivativeApproximation)
    cdef double d_x, nom_x0, denom_x0
    cdef DoubleScalarFPtr f_ptr  # __pyx_vtab error workaround

    with nogil:
        while not check_stop(&stop, error, precision, step > 0):
            for i in range(fs_x0.shape[0]):
                if not math.isfinite(fs_x0[i]):
                    stop.reason = NONFINITE
                    break
            if stop.reason == NONFINITE:
                break
            if step >= max_iter > 0:
                stop.reason = MAX_ITERATIONS
                break
            d_x = delta_f.eval(fs_x0)
            if not math.isfinite(d_x) or not math.isfinite(x0 + d_x):
                stop.reason = NONFINITE
                break
            step += 1
            precision = math.fabs((x0 + d_x) - x0)
            x0 = x0 + d_x
            advance_x(&stop, math.fabs(x0))

            with gil:
                f_ptr = fs[0]
            fs_x0[0] = f_ptr.eval(x0)
            error = math.fabs(fs_x0[0])
            if check_stop(&stop, error, precision, True) or step >= max_iter > 0:
                if stop.reason == RUNNING:
                    stop.reason = MAX_ITERATIONS
                for i in range(1, d + 1):
                    fs_x0[i] = math.NAN
                break
            for i in range(1, d + 1):
                with gil:
                    f_ptr = fs[i]
                if use_derivative_approximation[i - 1]:
                    fs_x0[i] = f_ptr.eval2(x0, fs_x0[0])
                else:
                    fs_x0[i] = f_ptr.eval(x0)

        finish_stop(&stop, error, precision, step > 0)
        converged = successful(stop.reason)
        optimal = stop.reason == RESIDUAL_TOLERANCE
        for i in range(1, d + 1):
            dfs_x0[i - 1] = fs_x0[i]
    return NewtonMethodsReturnType(
        x0, fs_x0[0], tuple(dfs_x0[i] for i in range(len(dfs_x0))), step,
        tuple(fs[i].n_f_calls for i in range(len(fs))),
        precision, error, converged, optimal, termination_reason=<int>stop.reason)

#########################
# Ahmadi Delta
#########################
cdef class AD3DFPtr(DoubleVectorReductionFPtr):
    cdef inline double eval(self, double[:] fs) noexcept nogil:
        cdef double res
        if fs[2] <= 0:
            res = math.NAN
        else:
            if fs[3] == 0:  # classical Newton delta
                res = - fs[1] / fs[2]
            else:
                res = - 2 * fs[2] / fs[3] - math.cbrt(
                    (fs[1] - 2 / 3 * (fs[2] * fs[2]) / fs[3]) / ((fs[3] * fs[3]) / 12 / fs[2])
                )
                # res = - 2 * fs[2] / fs[3] - math.cbrt(
                #     12 * fs[1] * fs[2] / (fs[3] * fs[3] * fs[3]) - (2 * fs[2] / fs[3]) ** 3
                # )
        return res

# noinspection DuplicatedCode
class AhmadiDeltaFuncFactory:
    r"""
    Factory class containing compiled Ahmadi delta functions for reuse.
    """
    # dictionary of known functions
    ad_c_funcs: dict[int, DoubleVectorReductionFPtr] = {
        3: AD3DFPtr(),
    }
    ad_py_funcs: dict[int, DoubleVectorReductionFPtr] = {
        order: PyDoubleVectorReductionFPtr(func.__call__)
        for order, func in ad_c_funcs.items()
    }
    def __init__(self):
        raise RuntimeError('Do not initialize this class.')

    @classmethod
    def get(cls, d: int, c_code: bool = True):
        if d not in cls.ad_c_funcs:
            raise NotImplementedError(f'Not implemented for d={d}.')
        return cls.ad_c_funcs[d] if c_code else cls.ad_py_funcs[d]

# noinspection DuplicatedCode
@cython.binding(True)
def ahmadi(f: Callable[[float], float],
           dfs: Optional[Union[np.ndarray, Sequence[Optional[Callable[[float], float]]]]],
           x0: float,
           f_x0: Optional[float] = None,
           dfs_x0: Optional[VectorLike] = None,
           d: Optional[int] = 3,
           h: Optional[float] = FINITE_DIFF_STEP,
           etol: float = ETOL,
           ertol: float = ERTOL,
           ptol: float = PTOL,
           prtol: float = PRTOL,
           max_iter = MAX_ITER) -> NewtonMethodsReturnType:
    r"""
    Ahmadi's higher-order Newton method for scalar root-finding.

    Notes:
        - This is currently only implemented for ``d=3``.
        - Experimental: its root-finding formula is not independently verified.
        - Available only by explicit import, outside normal method discovery.

    References:
        https://arxiv.org/abs/2311.06374

    Args:
        f (function): Function for which the root is sought.
        dfs (tuple of function, optional): Tuple of derivative
         functions of ``f`` in increasing order.
        x0 (float): Initial guess.
        f_x0 (float, optional): Value evaluated at initial guess.
        dfs_x0 (tuple of float, optional): Tuple of derivatives
         in increasing order at initial guess.
        d (int, optional): Max order of derivatives, ignored
         when ``dfs`` is not None. Defaults to 3.
        h (float, optional): Finite difference step size,
         ignored when none of the ``dfs`` is not None.
         Defaults to {h}.
        etol (float, optional): Error tolerance, indicating the
         desired precision of the root. Defaults to {etol}.
        ertol (float, optional): Relative error tolerance.
         Defaults to {ertol}.
        ptol (float, optional): Precision tolerance, indicating
         the minimum change of root approximations or width of
         brackets (in bracketing methods) after each iteration.
         Defaults to {ptol}.
        prtol (float, optional): Relative precision tolerance.
         Defaults to {prtol}.
        max_iter (int, optional): Maximum number of iterations.
         If set to 0, the procedure will run indefinitely until
         stopping condition is met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    import warnings
    warnings.warn('ahmadi\'s method is experimental, currently '
                  'only implemented for third order derivative (d=3).')

    # check params
    etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if dfs is None:
        if d is None:
            raise ValueError('d must be set if dfs is None.')
        elif d < 3:
            raise ValueError(f'Requires at least third order derivative. Got {d}.')
        dfs_wrappers = [FiniteDifference(f_wrapper, h=h, order=i + 1) for i in range(d)]
    else:
        if len(dfs) < 3:
            raise ValueError(f'Requires at least third order derivative. Got {len(dfs)}.')
        dfs_wrappers = [PyDoubleScalarFPtr.from_f(df) if df is not None else
                        FiniteDifference(f_wrapper, h=h, order=i + 1)
                        for i, df in enumerate(dfs)]
        d = len(dfs_wrappers)

    if f_x0 is None:
        f_x0 = f_wrapper(x0)
    initial_stopped = _initial_scalar_stopped(x0, f_x0, etol, ertol, ptol, prtol)
    if dfs_x0 is None:
        dfs_x0 = ([math.NAN] * len(dfs_wrappers) if initial_stopped else
                   [df_wrapper(x0) for df_wrapper in dfs_wrappers])
    fs_x0 = np.empty(d + 1, dtype=np.float64)
    fs_x0[0] = f_x0
    fs_x0[1:] = np.asarray(dfs_x0, dtype=np.float64)

    res = ahmadi_kernel(
        np.asarray([f_wrapper] + dfs_wrappers),
        <DoubleVectorReductionFPtr> AhmadiDeltaFuncFactory.get(d),
        x0, fs_x0, d, etol, ertol, ptol, prtol, max_iter)
    return res
