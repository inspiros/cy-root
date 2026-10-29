# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from cyroot.defaults cimport ETOL, ERTOL, PTOL, PRTOL, MAX_ITER

import cython
from libc cimport math

from ._check_args import _check_stop_cond_args
from ._stopping cimport (
    StopState, init_stop, advance_x, check_stop, finish_stop, successful, residual_met,
    RUNNING, RESIDUAL_TOLERANCE, MAX_ITERATIONS, ZERO_DERIVATIVE, BREAKDOWN, NONFINITE,
)
from .return_types cimport NewtonMethodsReturnType
from .utils._function_registering import register


__all__ = [
    'newton',
    'damped_newton',
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
def damped_newton(f, df, x0, f_x0=None, df_x0=None, *, h=None,
                  etol=ETOL, ertol=ERTOL, ptol=PTOL, prtol=PRTOL, max_iter=MAX_ITER,
                  c1=1e-4, contraction=.5, max_line_search=32, **options):
    """Newton with native Armijo backtracking; None selects numerical df."""
    from .native.damped_newton import damped_newton as solve
    f, df, _ = _providers(f, df, None, h)
    return solve(f, df, x0, f_x0=f_x0, df_x0=df_x0, etol=etol, ertol=ertol,
                 ptol=ptol, prtol=prtol, max_iter=max_iter, c1=c1,
                 contraction=contraction, max_line_search=max_line_search, **options)


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

