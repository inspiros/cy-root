# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Sequence, Optional, Union

import cython
import numpy as np
cimport numpy as np
import sympy
import sympy.utilities.autowrap
from cython cimport view
from dynamic_default_args import dynamic_default_args, named_default
from libc cimport math
from libcpp.vector cimport vector

from ._check_args cimport _check_stop_cond_scalar_initial_guess
from ._check_args import _check_stop_cond_args
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER, FINITE_DIFF_STEP
from ._types import VectorLike
from .fptr cimport (
    DoubleScalarFPtr, PyDoubleScalarFPtr,
    DoubleVectorFPtr, PyDoubleVectorFPtr,
)
from .scalar_derivative_approximation import DerivativeApproximation, FiniteDifference
from .ops cimport scalar_ops as sops
from .return_types cimport NewtonMethodsReturnType
from .utils._function_registering import register

__all__ = [
    'newton',
    'halley',
    'super_halley',
    'chebyshev',
    'tangent_hyperbolas',
    'householder',
]

################################################################################
# Newton
################################################################################
# noinspection DuplicatedCode
cdef NewtonMethodsReturnType newton_kernel(
        DoubleScalarFPtr f,
        DoubleScalarFPtr df,
        double x0,
        double f_x0,
        double df_x0,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_cond_scalar_initial_guess(x0, f_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return NewtonMethodsReturnType(
            x0, f_x0, df_x0, step, (f.n_f_calls, df.n_f_calls), precision, error, converged, optimal)

    cdef bint use_derivative_approximation = isinstance(df, DerivativeApproximation)
    cdef double d_x
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0 or df_x0 == 0:
            converged = False
        step += 1
        d_x = -f_x0 / df_x0
        x0 = x0 + d_x
        f_x0 = f.eval(x0)
        if use_derivative_approximation:
            df_x0 = df.eval_with_f_val(x0, f_x0)
        else:
            df_x0 = df.eval(x0)
        precision = math.fabs(d_x)
        error = math.fabs(f_x0)

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, f_x0, df_x0, step, (f.n_f_calls, df.n_f_calls), precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.newton')
@dynamic_default_args()
@cython.binding(True)
def newton(f: Callable[[float], float],
           df: Optional[Callable[[float], float]],
           x0: float,
           f_x0: Optional[float] = None,
           df_x0: Optional[float] = None,
           h: Optional[float] = named_default(FINITE_DIFF_STEP=FINITE_DIFF_STEP),
           etol: float = named_default(ETOL=ETOL),
           ertol: float = named_default(ERTOL=ERTOL),
           ptol: float = named_default(PTOL=PTOL),
           prtol: float = named_default(PRTOL=PRTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Newton method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        df (function, optional): Function returning derivative
         of ``f``.
        x0 (float): Initial point.
        f_x0 (float, optional): Value evaluated at
         initial point.
        df_x0 (float, optional): First order derivative at
         initial point.
        h (float, optional): Finite difference step size,
         ignored when ``df`` is not None. Defaults to {h}.
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
    # check params
    etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if df is None:
        df_wrapper = FiniteDifference(f_wrapper, h=h, order=1)
    else:
        df_wrapper = PyDoubleScalarFPtr.from_f(df)

    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if df_x0 is None:
        df_x0 = df_wrapper.eval(x0)

    res = newton_kernel(f_wrapper, df_wrapper, x0, f_x0, df_x0,
                        etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Halley
################################################################################
# noinspection DuplicatedCode
cdef NewtonMethodsReturnType halley_kernel(
        DoubleScalarFPtr f,
        DoubleScalarFPtr df,
        DoubleScalarFPtr d2f,
        double x0,
        double f_x0,
        double df_x0,
        double d2f_x0,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_cond_scalar_initial_guess(x0, f_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return NewtonMethodsReturnType(
            x0, f_x0, (df_x0, d2f_x0), step, (f.n_f_calls, df.n_f_calls, d2f.n_f_calls),
            precision, error, converged, optimal)

    cdef bint[2] use_derivative_approximation = [isinstance(df, DerivativeApproximation),
                                                 isinstance(d2f, DerivativeApproximation)]
    cdef double d_x, denom
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        denom = 2 * math.fabs(df_x0) ** 2 - f_x0 * d2f_x0
        if denom == 0:
            converged = False
            break
        d_x = -2 * f_x0 * df_x0 / denom
        x0 = x0 + d_x
        f_x0 = f.eval(x0)
        if use_derivative_approximation[0]:
            df_x0 = df.eval_with_f_val(x0, f_x0)
        else:
            df_x0 = df.eval(x0)
        if use_derivative_approximation[1]:
            d2f_x0 = d2f.eval_with_f_val(x0, f_x0)
        else:
            d2f_x0 = d2f.eval(x0)
        precision = math.fabs(d_x)
        error = math.fabs(f_x0)

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, f_x0, (df_x0, d2f_x0), step, (f.n_f_calls, df.n_f_calls, d2f.n_f_calls),
        precision, error, converged, optimal)

# noinspection DuplicatedCode
cdef NewtonMethodsReturnType modified_halley_kernel(
        DoubleScalarFPtr f,
        DoubleScalarFPtr df,
        DoubleScalarFPtr d2f,
        double x0,
        double f_x0,
        double df_x0,
        double d2f_x0,
        double alpha=0.5,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_cond_scalar_initial_guess(x0, f_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return NewtonMethodsReturnType(
            x0, f_x0, (df_x0, d2f_x0), step, (f.n_f_calls, df.n_f_calls, d2f.n_f_calls),
            precision, error, converged, optimal)

    cdef bint[2] use_derivative_approximation = [isinstance(df, DerivativeApproximation),
                                                 isinstance(d2f, DerivativeApproximation)]
    cdef double d_x, L_f, denom
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        if df_x0 == 0:
            converged = False
            break
        L_f = d2f_x0 * f_x0 / (df_x0 * df_x0)
        denom = 2 * (1 - alpha * L_f)
        if denom == 0:
            converged = False
            break
        d_x = -(1 + L_f / denom) * f_x0 / df_x0
        x0 = x0 + d_x
        f_x0 = f.eval(x0)
        if use_derivative_approximation[0]:
            df_x0 = df.eval_with_f_val(x0, f_x0)
        else:
            df_x0 = df.eval(x0)
        if use_derivative_approximation[1]:
            d2f_x0 = d2f.eval_with_f_val(x0, f_x0)
        else:
            d2f_x0 = d2f.eval(x0)
        precision = math.fabs(d_x)
        error = math.fabs(f_x0)

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, f_x0, (df_x0, d2f_x0), step, (f.n_f_calls, df.n_f_calls, d2f.n_f_calls),
        precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.newton')
@dynamic_default_args()
@cython.binding(True)
def halley(f: Callable[[float], float],
           df: Optional[Callable[[float], float]],
           d2f: Optional[Callable[[float], float]],
           x0: float,
           f_x0: Optional[float] = None,
           df_x0: Optional[float] = None,
           d2f_x0: Optional[float] = None,
           alpha: Optional[float] = None,
           h: Optional[float] = named_default(FINITE_DIFF_STEP=FINITE_DIFF_STEP),
           etol: float = named_default(ETOL=ETOL),
           ertol: float = named_default(ERTOL=ERTOL),
           ptol: float = named_default(PTOL=PTOL),
           prtol: float = named_default(PRTOL=PRTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Halley's method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        df (function, optional): Function returning derivative
         of ``f``.
        d2f (function, optional): Function returning second order derivative
         of ``f``.
        x0 (float): Initial point.
        f_x0 (float, optional): Value evaluated at initial point.
        df_x0 (float, optional): First order derivative at
         initial point.
        d2f_x0 (float, optional): Second order derivative at
         initial point.
        alpha (float, optional): If set, the modified halley
         formula which has parameter alpha will be used.
        h (float, optional): Finite difference step size,
         ignored when ``df`` and ``d2f`` are not None.
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
    # check params
    etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if df is None:
        df_wrapper = FiniteDifference(f_wrapper, h=h, order=1)
    else:
        df_wrapper = PyDoubleScalarFPtr.from_f(df)
    if d2f is None:
        d2f_wrapper = FiniteDifference(f_wrapper, h=h, order=2)
    else:
        d2f_wrapper = PyDoubleScalarFPtr.from_f(d2f)

    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if df_x0 is None:
        df_x0 = df_wrapper.eval(x0)
    if d2f_x0 is None:
        d2f_x0 = d2f_wrapper.eval(x0)

    if alpha is None:
        res = halley_kernel(
            f_wrapper, df_wrapper, d2f_wrapper,
            x0, f_x0, df_x0, d2f_x0, etol, ertol, ptol, prtol, max_iter)
    else:
        res = modified_halley_kernel(
            f_wrapper, df_wrapper, d2f_wrapper,
            x0, f_x0, df_x0, d2f_x0, alpha, etol, ertol, ptol, prtol, max_iter)
    return res

#------------------------
# Super-Halley
#------------------------
# noinspection DuplicatedCode
@register('cyroot.scalar.newton')
@dynamic_default_args()
@cython.binding(True)
def super_halley(f: Callable[[float], float],
                 df: Optional[Callable[[float], float]],
                 d2f: Optional[Callable[[float], float]],
                 x0: float,
                 f_x0: Optional[float] = None,
                 df_x0: Optional[float] = None,
                 d2f_x0: Optional[float] = None,
                 h: Optional[float] = named_default(FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                 etol: float = named_default(ETOL=ETOL),
                 ertol: float = named_default(ERTOL=ERTOL),
                 ptol: float = named_default(PTOL=PTOL),
                 prtol: float = named_default(PRTOL=PRTOL),
                 max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Super-Halley's method for scalar root-finding.
    This is equivalent to calling ``halley`` with ``alpha=1``.

    References:
        https://www.sciencedirect.com/science/article/abs/pii/S0096300399001757

    Args:
        f (function): Function for which the root is sought.
        df (function, optional): Function returning derivative
         of ``f``.
        d2f (function, optional): Function returning second order derivative
         of ``f``.
        x0 (float): Initial point.
        f_x0 (float, optional): Value evaluated at initial point.
        df_x0 (float, optional): First order derivative at
         initial point.
        d2f_x0 (float, optional): Second order derivative at
         initial point.
        h (float, optional): Finite difference step size,
         ignored when ``df`` and ``d2f`` are not None.
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
    return halley(f, df, d2f, x0, f_x0, df_x0, d2f_x0, 1, h,
                  etol, ertol, ptol, prtol, max_iter)

#------------------------
# Chebyshev
#------------------------
# noinspection DuplicatedCode
@register('cyroot.scalar.newton')
@dynamic_default_args()
@cython.binding(True)
def chebyshev(f: Callable[[float], float],
              df: Optional[Callable[[float], float]],
              d2f: Optional[Callable[[float], float]],
              x0: float,
              f_x0: Optional[float] = None,
              df_x0: Optional[float] = None,
              d2f_x0: Optional[float] = None,
              h: Optional[float] = named_default(FINITE_DIFF_STEP=FINITE_DIFF_STEP),
              etol: float = named_default(ETOL=ETOL),
              ertol: float = named_default(ERTOL=ERTOL),
              ptol: float = named_default(PTOL=PTOL),
              prtol: float = named_default(PRTOL=PRTOL),
              max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Chebyshev's method for scalar root-finding.
    This is equivalent to calling ``halley`` with ``alpha=0``.

    Args:
        f (function): Function for which the root is sought.
        df (function, optional): Function returning derivative
         of ``f``.
        d2f (function, optional): Function returning second order derivative
         of ``f``.
        x0 (float): Initial point.
        f_x0 (float, optional): Value evaluated at initial point.
        df_x0 (float, optional): First order derivative at
         initial point.
        d2f_x0 (float, optional): Second order derivative at
         initial point.
        h (float, optional): Finite difference step size,
         ignored when ``df`` and ``d2f`` are not None.
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
    return halley(f, df, d2f, x0, f_x0, df_x0, d2f_x0, 0, h,
                  etol, ertol, ptol, prtol, max_iter)

#------------------------
# Tangent Hyperbolas
#------------------------
# noinspection DuplicatedCode
cdef NewtonMethodsReturnType tangent_hyperbolas_kernel(
        DoubleScalarFPtr f,
        DoubleScalarFPtr df,
        DoubleScalarFPtr d2f,
        double x0,
        double f_x0,
        double df_x0,
        double d2f_x0,
        int formula=2,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_cond_scalar_initial_guess(x0, f_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return NewtonMethodsReturnType(
            x0, f_x0, (df_x0, d2f_x0), step, (f.n_f_calls, df.n_f_calls, d2f.n_f_calls),
            precision, error, converged, optimal)

    cdef bint[2] use_derivative_approximation = [isinstance(df, DerivativeApproximation),
                                                 isinstance(d2f, DerivativeApproximation)]
    cdef double d_x, a, denom
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        if df_x0 == 0:
            converged = False
            break
        a = -f_x0 / df_x0
        if formula == 2:  # more likely
            denom = df_x0 + .5 * d2f_x0 * a  # J + .5 * H.a
            if denom == 0:
                converged = False
                break
            d_x = -f_x0 / denom
        else:
            denom = 1 + .5 * d2f_x0 * a / df_x0  # I + .5 * J^-1.H.a
            if denom == 0:
                converged = False
                break
            d_x = a / denom
        x0 = x0 + d_x
        f_x0 = f.eval(x0)
        if use_derivative_approximation[0]:
            df_x0 = df.eval_with_f_val(x0, f_x0)
        else:
            df_x0 = df.eval(x0)
        if use_derivative_approximation[1]:
            d2f_x0 = d2f.eval_with_f_val(x0, f_x0)
        else:
            d2f_x0 = d2f.eval(x0)
        precision = math.fabs(d_x)
        error = math.fabs(f_x0)

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, f_x0, (df_x0, d2f_x0), step, (f.n_f_calls, df.n_f_calls, d2f.n_f_calls),
        precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.newton')
@dynamic_default_args()
@cython.binding(True)
def tangent_hyperbolas(f: Callable[[float], float],
                       df: Optional[Callable[[float], float]],
                       d2f: Optional[Callable[[float], float]],
                       x0: float,
                       f_x0: Optional[float] = None,
                       df_x0: Optional[float] = None,
                       d2f_x0: Optional[float] = None,
                       formula: Optional[int] = 2,
                       h: Optional[float] = named_default(FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                       etol: float = named_default(ETOL=ETOL),
                       ertol: float = named_default(ERTOL=ERTOL),
                       ptol: float = named_default(PTOL=PTOL),
                       prtol: float = named_default(PRTOL=PRTOL),
                       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Tangent Hyperbolas method for scalar root-finding.

    This method is the same as Halley's method.
    There are 2 formulas ``formula=1`` and ``formula=2``, the later
    (default) requires 1 less division.

    Args:
        f (function): Function for which the root is sought.
        df (function, optional): Function returning derivative
         of ``f``.
        d2f (function, optional): Function returning second order derivative
         of ``f``.
        x0 (float): Initial point.
        f_x0 (float, optional): Value evaluated at initial point.
        df_x0 (float, optional): First order derivative at
         initial point.
        d2f_x0 (float, optional): Second order derivative at
         initial point.
        formula (int): Formula 1 or 2. Defaults to 2.
        h (float, optional): Finite difference step size,
         ignored when ``df`` and ``d2f`` are not None.
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
    # check params
    etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)
    if formula not in [1, 2]:
        raise ValueError(f'Unknown formula {formula}.')

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if df is None:
        df_wrapper = FiniteDifference(f_wrapper, h=h, order=1)
    else:
        df_wrapper = PyDoubleScalarFPtr.from_f(df)
    if d2f is None:
        d2f_wrapper = FiniteDifference(f_wrapper, h=h, order=2)
    else:
        d2f_wrapper = PyDoubleScalarFPtr.from_f(d2f)

    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if df_x0 is None:
        df_x0 = df_wrapper.eval(x0)
    if d2f_x0 is None:
        d2f_x0 = d2f_wrapper.eval(x0)

    res = tangent_hyperbolas_kernel(
        f_wrapper, df_wrapper, d2f_wrapper, x0, f_x0, df_x0, d2f_x0, formula,
        etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Householder
################################################################################
# noinspection DuplicatedCode
# TODO: find other methods for __pyx_vtab error
#  https://stackoverflow.com/questions/37869945/cython-c-code-compilation-fails-with-typed-memoryviews
#  https://stackoverflow.com/questions/31119510/cython-have-sequence-of-extension-types-as-attribute-of-another-extension-type
cdef NewtonMethodsReturnType householder_kernel(
        DoubleScalarFPtr[:] fs,
        DoubleVectorFPtr nom_f,
        DoubleVectorFPtr denom_f,
        double x0_,
        double[:] fs_x0,
        unsigned int d,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    cdef vector[double] dfs_x0 = vector[double](fs_x0.shape[0] - 1)
    cdef unsigned int i
    for i in range(1, d + 1):
        dfs_x0[i - 1] = fs_x0[i]
    if _check_stop_cond_scalar_initial_guess(x0_, fs_x0[0], etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return NewtonMethodsReturnType(
            x0_, fs_x0[0], tuple(dfs_x0[i] for i in range(len(dfs_x0))), step,
            tuple(fs[i].n_f_calls for i in range(len(fs))),
            precision, error, converged, optimal)

    cdef bint[:] use_derivative_approximation = view.array(shape=(fs_x0.shape[0] - 1,),
                                                           itemsize=sizeof(int),
                                                           format='i')
    for i in range(1, d + 1):
        use_derivative_approximation[i - 1] = isinstance(fs[i], DerivativeApproximation)
    cdef double[:] x0 = view.array(shape=(1,),
                                   itemsize=sizeof(double),
                                   format='d')
    cdef double d_x
    x0[0] = x0_  # wrapped in a memory view to be able to pass into double_vector_func_type
    cdef double[:] nom_x0, denom_x0
    cdef DoubleScalarFPtr f_ptr  # __pyx_vtab error workaround
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        nom_x0 = nom_f.eval(fs_x0[:-1])
        denom_x0 = denom_f.eval(fs_x0)
        if denom_x0[0] == 0:
            converged = False
            break
        d_x = d * nom_x0[0] / denom_x0[0]
        x0[0] = x0[0] + d_x

        f_ptr = fs[0]
        fs_x0[0] = f_ptr.eval(x0[0])
        for i in range(1, d + 1):
            f_ptr = fs[i]
            if use_derivative_approximation[i - 1]:
                fs_x0[i] = f_ptr.eval_with_f_val(x0[0], fs_x0[0])
            else:
                fs_x0[i] = f_ptr.eval(x0[0])
        precision = math.fabs(d_x)
        error = math.fabs(fs_x0[0])

    optimal = sops.isclose(0, error, ertol, etol)
    for i in range(1, d + 1):
        dfs_x0[i - 1] = fs_x0[i]
    return NewtonMethodsReturnType(
        x0[0], fs_x0[0], tuple(dfs_x0[i] for i in range(len(dfs_x0))), step,
        tuple(fs[i].n_f_calls for i in range(len(fs))),
        precision, error, converged, optimal)

#########################
# Sympy Expr Evaluators
#########################
cdef class _Expr:
    cdef double eval(self, double[:] args):
        raise NotImplementedError

    def __call__(self, double[:] args) -> double:
        return self.eval(args)

cdef class _AtomicExpr(_Expr):
    pass

cdef class _Number(_AtomicExpr):
    cdef double value
    def __init__(self, number: sympy.core.numbers.Number):
        self.value = float(number.n())

    cdef inline double eval(self, double[:] args):
        return self.value

cdef class _Arg(_AtomicExpr):
    cdef unsigned long index
    def __init__(self, indexed: sympy.Indexed):
        self.index = <unsigned long> indexed.indices[0]

    cdef inline double eval(self, double[:] args):
        return args[self.index]

cdef class _ParameterizedExpr(_Expr):
    cdef readonly _Expr[:] args
    cdef _Expr _arg_i  # used for iterating args
    cdef unsigned int n_args
    def __init__(self, args):
        self.args = np.array([CyExprEvaluator.parse_symbolic_func(_) for _ in args])
        self.n_args = <unsigned int> len(self.args)

cdef class _Negate(_ParameterizedExpr):
    cdef _Expr arg
    def __init__(self, negative_one: sympy.core.numbers.NegativeOne):
        super().__init__(negative_one.args)
        self.arg = self.args[0]

    cdef inline double eval(self, double[:] args):
        return -self.arg.eval(args)

cdef class _Add(_ParameterizedExpr):
    def __init__(self, mul: sympy.core.Add):
        super().__init__(mul.args)

    cdef inline double eval(self, double[:] args):
        cdef double res = 0.0
        cdef unsigned int i
        for i in range(self.n_args):
            self._arg_i = self.args[i]
            res += self._arg_i.eval(args)
            self._arg_i = None
        return res

cdef class _Mul(_ParameterizedExpr):
    def __init__(self, mul: sympy.core.Mul):
        super().__init__(mul.args)

    cdef inline double eval(self, double[:] args):
        cdef double res = 1.0
        cdef unsigned int i
        for i in range(self.n_args):
            self._arg_i = self.args[i]
            res *= self._arg_i.eval(args)
            self._arg_i = None
        return res

cdef class _Pow(_ParameterizedExpr):
    cdef _Expr arg, exp
    def __init__(self, pow: sympy.core.Pow):
        super().__init__(pow.args)
        self.arg = self.args[0]
        self.exp = self.args[1]

    cdef inline double eval(self, double[:] args):
        return self.arg.eval(args) ** self.exp.eval(args)

#########################
# Reciprocal Derivative
#########################
from .fptr import DoubleVectorFPtr, PyDoubleVectorFPtr

# Some implementations up to 10th order
cdef class R0DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = 1/fs[0]
        return res

cdef class R1DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = -fs[1]/fs[0]**2
        return res

cdef class R2DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-fs[0]*fs[2] + 2*fs[1]**2)/fs[0]**3
        return res

cdef class R3DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-fs[0]**2*fs[3] + 6*fs[0]*fs[1]*fs[2] - 6*fs[1]**3)/fs[0]**4
        return res

cdef class R4DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((8*fs[1]*fs[3] + 6*fs[2]**2)*fs[0]**2 - fs[0]**3*fs[4] -
                  36*fs[0]*fs[1]**2*fs[2] + 24*fs[1]**4)/fs[0]**5
        return res

cdef class R5DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-30*(2*fs[1]*fs[3] + 3*fs[2]**2)*fs[0]**2*fs[1] +
                  10*(fs[1]*fs[4] + 2*fs[2]*fs[3])*fs[0]**3 - fs[0]**4*fs[5] +
                  240*fs[0]*fs[1]**3*fs[2] - 120*fs[1]**5)/fs[0]**6
        return res

cdef class R6DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((480*fs[1]*fs[3] + 1080*fs[2]**2)*fs[0]**2*fs[1]**2 +
                  (12*fs[1]*fs[5] + 30*fs[2]*fs[4] + 20*fs[3]**2)*fs[0]**4 +
                  (-90*fs[1]**2*fs[4] - 360*fs[1]*fs[2]*fs[3] - 90*fs[2]**3)*fs[0]**3 -
                  fs[0]**5*fs[6] - 1800*fs[0]*fs[1]**4*fs[2] + 720*fs[1]**6)/fs[0]**7
        return res

cdef class R7DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = (-4200*(fs[1]*fs[3] + 3*fs[2]**2)*fs[0]**2*fs[1]**3 +
                  14*(fs[1]*fs[6] + 3*fs[2]*fs[5] + 5*fs[3]*fs[4])*fs[0]**5 +
                  840*(fs[1]**2*fs[4] + 6*fs[1]*fs[2]*fs[3] + 3*fs[2]**3)*fs[0]**3*fs[1] -
                  42*(3*fs[1]**2*fs[5] + 15*fs[1]*fs[2]*fs[4] + 10*fs[1]*fs[3]**2 + 15*fs[2]**2*fs[3])*fs[0]**4 -
                  fs[0]**6*fs[7] + 15120*fs[0]*fs[1]**5*fs[2] - 5040*fs[1]**7)/fs[0]**8
        return res

cdef class R8DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((40320*fs[1]*fs[3] + 151200*fs[2]**2)*fs[0]**2*fs[1]**4 + (-8400*fs[1]**2*fs[4] -
                  67200*fs[1]*fs[2]*fs[3] - 50400*fs[2]**3)*fs[0]**3*fs[1]**2 + (16*fs[1]*fs[7] + 56*fs[2]*fs[6] +
                  112*fs[3]*fs[5] + 70*fs[4]**2)*fs[0]**6 + (-168*fs[1]**2*fs[6] - 1008*fs[1]*fs[2]*fs[5] -
                  1680*fs[1]*fs[3]*fs[4] - 1260*fs[2]**2*fs[4] - 1680*fs[2]*fs[3]**2)*fs[0]**5 +
                  (1344*fs[1]**3*fs[5] + 10080*fs[1]**2*fs[2]*fs[4] + 6720*fs[1]**2*fs[3]**2 +
                   20160*fs[1]*fs[2]**2*fs[3] + 2520*fs[2]**4)*fs[0]**4 - fs[0]**7*fs[8] -
                  141120*fs[0]*fs[1]**6*fs[2] + 40320*fs[1]**8)/fs[0]**9
        return res

cdef class R9DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((-423360*fs[1]*fs[3] - 1905120*fs[2]**2)*fs[0]**2*fs[1]**5 + (90720*fs[1]**2*fs[4] +
                  907200*fs[1]*fs[2]*fs[3] + 907200*fs[2]**3)*fs[0]**3*fs[1]**3 + (18*fs[1]*fs[8] + 72*fs[2]*fs[7] +
                  168*fs[3]*fs[6] + 252*fs[4]*fs[5])*fs[0]**7 + 2520*(-6*fs[1]**3*fs[5] - 60*fs[1]**2*fs[2]*fs[4] -
                  40*fs[1]**2*fs[3]**2 - 180*fs[1]*fs[2]**2*fs[3] - 45*fs[2]**4)*fs[0]**4*fs[1] + (2016*fs[1]**3*fs[6] +
                  18144*fs[1]**2*fs[2]*fs[5] + 30240*fs[1]**2*fs[3]*fs[4] + 45360*fs[1]*fs[2]**2*fs[4] +
                  60480*fs[1]*fs[2]*fs[3]**2 + 30240*fs[2]**3*fs[3])*fs[0]**5 + (-216*fs[1]**2*fs[7] -
                  1512*fs[1]*fs[2]*fs[6] - 3024*fs[1]*fs[3]*fs[5] - 1890*fs[1]*fs[4]**2 - 2268*fs[2]**2*fs[5] -
                  7560*fs[2]*fs[3]*fs[4] - 1680*fs[3]**3)*fs[0]**6 - fs[0]**8*fs[9] + 1451520*fs[0]*fs[1]**7*fs[2] -
                  362880*fs[1]**9)/fs[0]**10
        return res

cdef class R10DFPtr(DoubleVectorFPtr):
    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = ((4838400*fs[1]*fs[3] + 25401600*fs[2]**2)*fs[0]**2*fs[1]**6 + (-1058400*fs[1]**2*fs[4] -
                  12700800*fs[1]*fs[2]*fs[3] - 15876000*fs[2]**3)*fs[0]**3*fs[1]**4 + (20*fs[1]*fs[9] +
                  90*fs[2]*fs[8] + 240*fs[3]*fs[7] + 420*fs[4]*fs[6] + 252*fs[5]**2)*fs[0]**8 + (181440*fs[1]**3*fs[5] +
                  2268000*fs[1]**2*fs[2]*fs[4] + 1512000*fs[1]**2*fs[3]**2 + 9072000*fs[1]*fs[2]**2*fs[3] +
                  3402000*fs[2]**4)*fs[0]**4*fs[1]**2 + (-25200*fs[1]**4*fs[6] - 302400*fs[1]**3*fs[2]*fs[5] -
                  504000*fs[1]**3*fs[3]*fs[4] - 1134000*fs[1]**2*fs[2]**2*fs[4] - 1512000*fs[1]**2*fs[2]*fs[3]**2 -
                  1512000*fs[1]*fs[2]**3*fs[3] - 113400*fs[2]**5)*fs[0]**5 + (-270*fs[1]**2*fs[8] -
                  2160*fs[1]*fs[2]*fs[7] - 5040*fs[1]*fs[3]*fs[6] - 7560*fs[1]*fs[4]*fs[5] - 3780*fs[2]**2*fs[6] -
                  15120*fs[2]*fs[3]*fs[5] - 9450*fs[2]*fs[4]**2 - 12600*fs[3]**2*fs[4])*fs[0]**7 +
                  (2880*fs[1]**3*fs[7] + 30240*fs[1]**2*fs[2]*fs[6] + 60480*fs[1]**2*fs[3]*fs[5] +
                  37800*fs[1]**2*fs[4]**2 + 90720*fs[1]*fs[2]**2*fs[5] + 302400*fs[1]*fs[2]*fs[3]*fs[4] +
                  67200*fs[1]*fs[3]**3 + 75600*fs[2]**3*fs[4] + 151200*fs[2]**2*fs[3]**2)*fs[0]**6 - fs[0]**9*fs[10] -
                  16329600*fs[0]*fs[1]**8*fs[2] + 3628800*fs[1]**10)/fs[0]**11
        return res

# For functions of higher order derivatives, use this class to eval expression
# Warning: Very slow, but somehow still slightly faster than Sympy's wrapped function
# (if someone goes this far, they must be insane)
cdef class CyExprEvaluator(DoubleVectorFPtr):
    cdef _Expr cy_expr
    type_map = {
        sympy.core.Number: _Number,
        sympy.Indexed: _Arg,
        sympy.core.numbers.NegativeOne: _Negate,
        sympy.core.Add: _Add,
        sympy.core.Mul: _Mul,
        sympy.core.Pow: _Pow,
    }
    def __init__(self, expr: sympy.core.Expr):
        self.cy_expr = self.parse_symbolic_func(expr)

    @staticmethod
    def parse_symbolic_func(expr):
        if isinstance(expr, sympy.Expr):
            if isinstance(expr, sympy.core.Number):
                return _Number(expr)
            evaluator_cls = CyExprEvaluator.type_map.get(type(expr))
            if evaluator_cls is not None:
                return evaluator_cls(expr)
            else:
                raise NotImplementedError(f'No implementation found for {type(expr)}.')

    cdef inline double[:] eval(self, double[:] fs):
        cdef double[:] res = view.array(shape=(1,), itemsize=sizeof(double), format='d')
        res[0] = self.cy_expr.eval(fs)
        return res

# noinspection DuplicatedCode
class ReciprocalDerivativeFuncFactory:
    """Factory class containing compiled reciprocal derivative functions for reuse.

    References:
        https://math.stackexchange.com/questions/5357/whats-the-generalisation-of-the-quotient-rule-for-higher-derivatives
    """
    # dictionary of known functions
    rd_c_funcs: dict[int, DoubleVectorFPtr] = {
        0: R0DFPtr(),
        1: R1DFPtr(),
        2: R2DFPtr(),
        3: R3DFPtr(),
        4: R4DFPtr(),
        5: R5DFPtr(),
        6: R6DFPtr(),
        7: R7DFPtr(),
        8: R8DFPtr(),
        9: R9DFPtr(),
        10: R10DFPtr(),
    }
    rd_py_funcs: dict[int, DoubleVectorFPtr] = dict(
        enumerate(map(lambda o: PyDoubleVectorFPtr(o.__call__), rd_c_funcs.values())))
    def __init__(self):
        raise RuntimeError('Do not initialize this class.')

    @classmethod
    def get(cls, d: int, max_d: int = None, c_code: bool = False):
        if (not c_code and d not in cls.rd_py_funcs.keys()) or (c_code and d not in cls.rd_c_funcs):
            if max_d is None:
                max_d = d
            sym_x = sympy.Dummy('x')
            sym_f = sympy.Function(sympy.Dummy('f'))
            sym_fs = sympy.IndexedBase(sympy.Dummy('fs'), shape=(max_d + 1,))

            expr = 1 / sym_f(sym_x)
            sym_rd_f = expr.diff(sym_x, d).simplify()
            for i in range(d, -1, -1):
                sym_rd_f = sym_rd_f.subs(sympy.Derivative(sym_f(sym_x), (sym_x, i)), sym_fs[i])

            if not c_code:
                rd_f = sympy.lambdify(sym_fs, sympy.Array([sym_rd_f]), modules='numpy')
                cls.rd_py_funcs[d] = PyDoubleVectorFPtr(rd_f)
            else:
                rd_f = <DoubleVectorFPtr> CyExprEvaluator(sym_rd_f)
                cls.rd_c_funcs[d] = rd_f
                # sympy autowrap does not support array
                # sym_fs_mat = sympy.MatrixSymbol('fs', d + 1, 1)
                # for i in range(d, -1, -1):
                #     sym_rd_f = sym_rd_f.subs(sym_fs[i], sym_fs_mat[i])
                # rd_f = sympy.utilities.autowrap.autowrap(sym_rd_f, backend='cython')
                # cls.rd_c_funcs[d] = PyDoubleMemoryViewFPtr(
                #     lambda fs: np.reshape(rd_f(np.reshape(fs, (-1, 1))), (-1,)))
        return cls.rd_py_funcs[d] if not c_code else cls.rd_c_funcs[d]

# noinspection DuplicatedCode
@register('cyroot.scalar.newton')
@dynamic_default_args()
@cython.binding(True)
def householder(f: Callable[[float], float],
                dfs: Optional[Union[np.ndarray, Sequence[Optional[Callable[[float], float]]]]],
                x0: float,
                f_x0: Optional[float] = None,
                dfs_x0: Optional[VectorLike] = None,
                d: Optional[int] = None,
                h: Optional[float] = named_default(FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                etol: float = named_default(ETOL=ETOL),
                ertol: float = named_default(ERTOL=ERTOL),
                ptol: float = named_default(PTOL=PTOL),
                prtol: float = named_default(PRTOL=PRTOL),
                max_iter: int = named_default(MAX_ITER=MAX_ITER),
                c_code: bool = True) -> NewtonMethodsReturnType:
    """
    Householder's method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        dfs (tuple of function, optional): Tuple of derivative
         functions of ``f`` in increasing order.
        x0 (float): Initial guess.
        f_x0 (float, optional): Value evaluated at initial guess.
        dfs_x0 (tuple of float, optional): Tuple of derivatives
         in increasing order at initial guess.
        d (int, optional): Max order of derivatives, ignored
         when ``dfs`` is not None. Defaults to None.
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
        c_code (bool, optional): Use C implementation of reciprocal
         derivative function or not. Defaults to {c_code}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if dfs is None:
        if d is None:
            raise ValueError('d must be set if dfs is None.')
        dfs_wrappers = [FiniteDifference(f_wrapper, h=h, order=i + 1) for i in range(d)]
    else:
        if len(dfs) < 2:
            raise ValueError(f'Requires at least second order derivative. Got {len(dfs)}.')
        dfs_wrappers = [PyDoubleScalarFPtr.from_f(df) if df is not None else
                        FiniteDifference(f_wrapper, h=h, order=i + 1)
                        for i, df in enumerate(dfs)]
        d = len(dfs_wrappers)

    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if dfs_x0 is None:
        dfs_x0 = [df_wrapper.eval(x0) for df_wrapper in dfs_wrappers]
    fs_x0 = np.empty(d + 1, dtype=np.float64)
    fs_x0[0] = f_x0
    fs_x0[1:] = np.asarray(dfs_x0, dtype=np.float64)

    res = householder_kernel(
        np.asarray([f_wrapper] + dfs_wrappers),
        <DoubleVectorFPtr> ReciprocalDerivativeFuncFactory.get(d - 1, c_code=c_code),
        <DoubleVectorFPtr> ReciprocalDerivativeFuncFactory.get(d, c_code=c_code),
        x0, fs_x0, d, etol, ertol, ptol, prtol, max_iter)
    return res
