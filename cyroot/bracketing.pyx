# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

import cython
from cpython cimport array
from cython cimport view
from libc cimport math

from ._check_args import (
    _check_stop_condition_args,
    _check_bracket,
    _check_unique_initial_vals,
)
from ._check_args cimport _check_stop_condition_bracket
from ._defaults import ETOL, PTOL, MAX_ITER
from ._return_types import BracketingMethodsReturnType
from .fptr cimport double_scalar_func_type, DoubleScalarFPtr, PyDoubleScalarFPtr
from .utils.dynamic_default_args import dynamic_default_args, named_default
from .utils.function_tagging import tag

__all__ = [
    'bisect',
    'regula_falsi',
    'illinois',
    'pegasus',
    'anderson_bjorck',
    'dekker',
    'brent',
    'chandrupatla',
    'ridders',
    'toms748',
    'wu',
    'itp',
]

cdef inline (double, double, double, double, double, double)  _update_bracket(
        double a, double b, double c, double f_a, double f_b, double f_c):
    """Update a bracket given (c, f_c)."""
    cdef double rx, f_rx
    if math.copysign(1, f_b) * math.copysign(1, f_c) < 0:
        rx, f_rx = a, f_a
        a, f_a = c, f_c
    else:
        rx, f_rx = b, f_b
        b, f_b = c, f_c
    return a, b, rx, f_a, f_b, f_rx

################################################################################
# Bisection
################################################################################
ctypedef bint (*stop_func_type)(long, double, double, double, double)

# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) bisect_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER,
        stop_func_type extra_stop_criteria=NULL):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef double c, f_c
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        if extra_stop_criteria is not NULL and extra_stop_criteria(step, a, b, f_a, f_b):
            converged = False
            break
        step += 1
        c = (a + b) / 2
        f_c = f(c)
        error = math.fabs(f_c)
        if error <= etol:
            break
        a, b, _, f_a, f_b, _ = _update_bracket(a, b, c, f_a, f_b, f_c)
        precision = math.fabs(b - a)

    r, f_r = c, f_c
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def bisect(f: Callable[[float], float],
           a: float,
           b: float,
           f_a: Optional[float] = None,
           f_b: Optional[float] = None,
           etol: float = named_default(ETOL=ETOL),
           ptol: float = named_default(PTOL=PTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Bisection method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = bisect_kernel[DoubleScalarFPtr](f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# False position
################################################################################
ctypedef double (*scale_func_type)(double, double)

# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) regula_falsi_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER,
        scale_func_type scale_func=NULL):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef double c, f_c
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0 or f_a == f_b:
            converged = False
            break
        step += 1
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = f(c)
        precision = math.fabs(b - a)
        error = math.fabs(f_c)

        if error <= etol:
            break
        elif math.copysign(1, f_b) * math.copysign(1, f_c) < 0:
            a, f_a = b, f_b
        elif scale_func is not NULL:
            f_a *= scale_func(f_b, f_c)
        b, f_b = c, f_c

    r, f_r = c, f_c
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def regula_falsi(f: Callable[[float], float],
                 a: float,
                 b: float,
                 f_a: Optional[float] = None,
                 f_b: Optional[float] = None,
                 etol: float = named_default(ETOL=ETOL),
                 ptol: float = named_default(PTOL=PTOL),
                 max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Regula Falsi (or False Position) method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Illinois
################################################################################
cdef inline double illinois_scale(double f_b, double f_c):
    return 0.5

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def illinois(f: Callable[[float], float],
             a: float,
             b: float,
             f_a: Optional[float] = None,
             f_b: Optional[float] = None,
             etol: float = named_default(ETOL=ETOL),
             ptol: float = named_default(PTOL=PTOL),
             max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Illinois method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter, illinois_scale)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Pegasus
################################################################################
cdef inline double pegasus_scale(double f_b, double f_c):
    return f_b / (f_b + f_c)

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def pegasus(f: Callable[[float], float],
            a: float,
            b: float,
            f_a: Optional[float] = None,
            f_b: Optional[float] = None,
            etol: float = named_default(ETOL=ETOL),
            ptol: float = named_default(PTOL=PTOL),
            max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Pegasus method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter, pegasus_scale)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Anderson–Björck
################################################################################
cdef inline double anderson_bjorck_scale(double f_b, double f_c):
    m = 1 - f_c / f_b
    if m <= 0:
        return 0.5
    return m

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def anderson_bjorck(f: Callable[[float], float],
                    a: float,
                    b: float,
                    f_a: Optional[float] = None,
                    f_b: Optional[float] = None,
                    etol: float = named_default(ETOL=ETOL),
                    ptol: float = named_default(PTOL=PTOL),
                    max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Anderson–Björck method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter, anderson_bjorck_scale)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Dekker
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) dekker_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef double b_prev = a, f_b_prev = f_a, b_next, f_b_next, c, s
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        c = (a + b) / 2
        if f_b != f_b_prev:
            s = b - (b - b_prev) * f_b / (f_b - f_b_prev)
        else:
            s = math.NAN
        if math.isnan(s) or (s - b) * (s - c) >= 0:
            b_next = c
        else:
            b_next = s
        f_b_next = f(b_next)

        if math.copysign(1, f_b) * math.copysign(1, f_b_next) < 0:
            a, f_a = b, f_b
        b_prev, f_b_prev = b, f_b
        b, f_b = b_next, f_b_next
        if math.fabs(f_a) < math.fabs(f_b):  # swap
            a, f_a, b, f_b = b, f_b, a, f_a

        precision = math.fabs(b - a)
        error = math.fabs(f_b)

    r, f_r = b, f_b
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def dekker(f: Callable[[float], float],
           a: float,
           b: float,
           f_a: Optional[float] = None,
           f_b: Optional[float] = None,
           etol: float = named_default(ETOL=ETOL),
           ptol: float = named_default(PTOL=PTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Dekker's method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = dekker_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Brent
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) brent_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        int interp_method=1,
        double sigma=1e-6,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef double b_prev = a, f_b_prev = f_a, b_prev_prev = a, b_next, f_b_next, c, s
    cdef double d_ab, d_abp, d_bbp, df_ab, df_abp, df_bbp, denom
    cdef int last_method = 0  # 0 for bisect, 1 for interp
    cdef bint use_interp
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        c = (a + b) / 2

        s = math.NAN
        use_interp = False
        if interp_method == 0 and f_a != f_b != f_b_prev != f_a:
            # inverse quadratic interpolation
            df_ab = f_a - f_b
            df_abp = f_a - f_b_prev
            df_bbp = f_b - f_b_prev
            s = (a * f_b * f_b_prev / (df_ab * df_abp)
                 + b * f_a * f_b_prev / (-df_ab * df_bbp)
                 + b_prev * f_a * f_b / (df_abp * df_bbp))
            use_interp = True
        elif interp_method == 1 and a != b != b_prev:
            # hyperbolic interpolation
            d_ab = a - b
            d_bbp = b - b_prev
            df_ab = f_a - f_b
            df_abp = f_a - f_b_prev
            df_bbp = f_b - f_b_prev
            denom = f_a * df_bbp / d_bbp - f_b_prev * df_ab / d_ab
            if denom != 0:
                s = b - f_b * df_abp / denom
                use_interp = True
            # s = a - f_a * df_bbp / (f_b * df_abp / d_abp - f_b_prev * df_ab / d_ab)
        if not use_interp and f_b != f_b_prev:
            s = b - (b - b_prev) * f_b / (f_b - f_b_prev)
        if (math.isnan(s) or (s - b) * (s - (3 * a + b) / 4) >= 0 or
                (last_method == 0 and math.fabs(s - b) >= math.fabs(b - b_prev) / 2) or
                (last_method == 1 and math.fabs(s - b) >= math.fabs(b_prev - b_prev_prev) / 2) or
                (last_method == 0 and sigma > math.fabs(b - b_prev)) or
                (last_method == 1 and sigma > math.fabs(b_prev - b_prev_prev))):
            last_method = 0
            b_next = c
        else:
            last_method = 1
            b_next = s
        f_b_next = f(b_next)

        if math.copysign(1, f_b) * math.copysign(1, f_b_next) < 0:
            a, f_a = b, f_b
        b_prev_prev = b_prev
        b_prev, f_b_prev = b, f_b
        b, f_b = b_next, f_b_next
        if math.fabs(f_a) < math.fabs(f_b):  # swap
            a, f_a, b, f_b = b, f_b, a, f_a

        precision = math.fabs(b - a)
        error = math.fabs(f_b)

    r, f_r = b, f_b
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

BRENT_INTERP_METHODS: dict[str, int] = {
    'quadratic': 0,
    'hyperbolic': 1,
}

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def brent(f: Callable[[float], float],
          a: float,
          b: float,
          f_a: Optional[float] = None,
          f_b: Optional[float] = None,
          interp_method: Union[str, int] = 'hyperbolic',
          sigma: float = 1e-5,
          etol: float = named_default(ETOL=ETOL),
          ptol: float = named_default(PTOL=PTOL),
          max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Brent's method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        interp_method: Interpolation method, 'quadratic' (or 0) for Inversed
         Quadratic Interpolation and 'hyperbolic' (or 1) for Hyperbolic
         Interpolation. Defaults to 1.
        sigma: Numerical tolerance to decide which method to use.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    if ((isinstance(interp_method, int) and interp_method not in BRENT_INTERP_METHODS.values()) or
            (isinstance(interp_method, str) and interp_method not in BRENT_INTERP_METHODS.keys())):
        raise ValueError(f'interp_method can be {BRENT_INTERP_METHODS}. '
                         f'No implementation found for {interp_method}.')
    if isinstance(interp_method, str):
        interp_method = BRENT_INTERP_METHODS[interp_method]
    if sigma <= 0:
        raise ValueError(f'sigma must be positive. Got {sigma}.')

    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wraper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    res = brent_kernel[DoubleScalarFPtr](
        f_wraper, a, b, f_a, f_b, interp_method, sigma, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wraper.n_f_calls)

################################################################################
# Chandrupatla
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) chandrupatla_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double sigma=PTOL,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef double c, f_c, d, f_d, d_ab, d_fa_fb, d_ad, d_fa_fd, d_bd, d_fb_fd, tl, t = 0.5
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        c = a + (b - a) * t
        f_c = f(c)
        if math.copysign(1, f_a) * math.copysign(1, f_c) > 0:
            d, a = a, c
            f_d, f_a = f_a, f_c
        else:
            d, b, a = b, a, c
            f_d, f_b, f_a = f_b, f_a, f_c
        precision = math.fabs(b - a)
        r, f_r = (a, f_a) if math.fabs(f_a) < math.fabs(f_b) else (b, f_b)
        error = math.fabs(f_r)
        tl = (2 * ptol * math.fabs(r) + 0.5 * sigma) / precision
        if error <= etol or tl > 0.5:
            break
        if a != b != d != a and f_a != f_b != f_d != f_a:
            # inverse quadratic interpolation
            d_ab = a - b
            d_ad = a - d
            d_bd = b - d
            d_fa_fb = f_a - f_b
            d_fa_fd = f_a - f_d
            d_fb_fd = f_b - f_d

            if 1 - math.sqrt(1 + d_ab / d_bd) < -d_fa_fb / d_fb_fd < math.sqrt(-d_ab / d_bd):
                t = f_a * f_d / (-d_fa_fb * d_fb_fd) + f_a * f_b / (-d_fa_fd * -d_fb_fd) * d_ad / d_ab
            else:
                t = 0.5
        if t < tl:
            t = tl
        if t > 1 - tl:
            t = 1 - tl

    # this method set r and f_r inside loop
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def chandrupatla(f: Callable[[float], float],
                 a: float,
                 b: float,
                 f_a: Optional[float] = None,
                 f_b: Optional[float] = None,
                 sigma: float = PTOL,
                 etol: float = named_default(ETOL=ETOL),
                 ptol: float = named_default(PTOL=PTOL),
                 max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Chandrupatla's method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        sigma: Tolerance for extra stop condition. Defaults to {ptol}.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = chandrupatla_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, sigma, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Ridders
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) ridders_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef double c, f_c, d, f_d, denom
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        c = (a + b) / 2
        f_c = f(c)

        denom = math.sqrt(f_c ** 2 - f_a * f_b)
        if denom == 0:
            converged = False
            break
        d = c + (c - a) * math.copysign(1, f_a) * f_c / denom
        f_d = f(d)

        if math.copysign(1, f_c) * math.copysign(1, f_d) < 0:  # well-behaved
            a, f_a = c, f_c
        elif math.copysign(1, f_b) * math.copysign(1, f_d) < 0:
            a, f_a = b, f_b
        b, f_b = d, f_d
        precision = math.fabs(b - a)
        error = math.fabs(f_d)

    if b < a:
        a, f_a = b, f_b
    r, f_r = d, f_d
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def ridders(f: Callable[[float], float],
            a: float,
            b: float,
            f_a: Optional[float] = None,
            f_b: Optional[float] = None,
            etol: float = named_default(ETOL=ETOL),
            ptol: float = named_default(PTOL=PTOL),
            max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Ridders method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = ridders_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# TOMS748
################################################################################
# noinspection DuplicatedCode
cdef double[:, :] _divided_differences(
        double[:] xs,
        double[:] fs,
        unsigned int N=0,
        bint forward=True):
    """Return a matrix of divided differences for the xs, fs pairs

    DD[i, j] = f[x_{i-j}, ..., x_i] for 0 <= j <= i
    If forward is False, return f[c], f[b, c], f[a, b, c]."""
    cdef unsigned int i, j, M = <unsigned int> len(xs)
    if not forward:
        xs = xs[::-1]
    N = M if N == 0 else min(N, M)
    cdef double[:, :] DD = view.array(shape=(M, N), itemsize=sizeof(double), format='d')
    DD[:, 0] = fs
    for j in range(1, N):
        DD[:j, j] = 0
        for i in range(j, M):
            DD[i, j] = (DD[i, j - 1] - DD[i - 1, j - 1]) / (xs[i] - xs[i - j])
    return DD

# noinspection DuplicatedCode
cdef double[:] _diag_divided_differences(
        double[:] xs,
        double[:] fs,
        unsigned int N=0,
        bint forward=True):
    """
    Just return the main diagonal(or last row):
      f[a], f[a, b] and f[a, b, c].
    """
    cdef unsigned int j, M = <unsigned int> len(xs)
    N = M if N == 0 else min(N, M)
    cdef double[:, :] DD = _divided_differences(xs, fs, N)
    cdef double[:] dd = view.array(shape=(N,), itemsize=sizeof(double), format='d')
    if forward:
        for j in range(N):
            dd[j] = DD[j, j]
    else:
        for j in range(N):
            dd[j] = DD[-1, j]
    return dd

# noinspection DuplicatedCode
cdef double _newton_quadratic(
        double a, double b, double d,
        double f_a, double f_b, double f_d,
        long k):
    """Apply Newton-Raphson like steps, using divided differences to approximate f'

    ab is a real interval [a, b] containing a root,
    fab holds the real values of f(a), f(b)
    d is a real number outside [ab, b]
    k is the number of steps to apply
    """
    cdef double[:] dd = _diag_divided_differences(
        array.array('d', [a, b, d]),
        array.array('d', [f_a, f_b, f_d]))
    cdef double B = dd[1], A = dd[2], P, r, r1

    if A == 0:
        r = a - f_a / B
    else:
        r = a if A * f_a > 0 else b
    # Apply k Newton-Raphson steps to P(x), starting from x=r
    cdef long i
    for i in range(k):
        # P is the quadratic polynomial through the 3 points
        # Horner evaluation of fa + B * (x - a) + A * (x - a) * (x - b)
        P = (A * (r - b) + B) * (r - a) + f_a
        r1 = r - P / (B + A * (2 * r - a - b))
        if not a < r1 < b:
            if a < r < b:
                return r
            r = (a + b) / 2
            break
        r = r1
    return r

# noinspection DuplicatedCode
cdef double _inverse_poly_zero(double a, double b, double c, double d,
                               double f_a, double f_b, double f_c, double f_d):
    cdef double Q11 = (c - d) * f_c / (f_d - f_c)
    cdef double Q21 = (b - c) * f_b / (f_c - f_b)
    cdef double Q31 = (a - b) * f_a / (f_b - f_a)
    cdef double D21 = (b - c) * f_c / (f_c - f_b)
    cdef double D31 = (a - b) * f_b / (f_b - f_a)
    cdef double Q22 = (D21 - Q11) * f_b / (f_d - f_b)
    cdef double Q32 = (D31 - Q21) * f_a / (f_c - f_a)
    cdef double D32 = (D31 - Q21) * f_c / (f_c - f_a)
    cdef double Q33 = (D32 - Q22) * f_a / (f_d - f_a)
    return a + Q31 + Q32 + Q33

# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) toms748_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        long k=2,
        double mu=0.5,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef long n
    cdef double c, f_c, d, f_d, e, f_e, u, f_u, z, f_z, A
    cdef bint use_newton_quadratic
    # secant init
    c = (a - b * f_a / f_b) / (1 - f_a / f_b)
    if not a < c < b:
        c = (a + b) / 2
    f_c = f(c)
    step += 1
    error = math.fabs(f_c)
    if error <= etol:
        r, f_r = c, f_c
        return r, f_r, step, a, b, f_a, f_b, precision, error, True, True
    a, b, d, f_a, f_b, f_d = _update_bracket(a, b, c, f_a, f_b, f_c)
    precision = math.fabs(b - a)
    e = f_e = math.NAN
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        ab_width = b - a

        for n in range(2, k + 2):
            # If the f-values are sufficiently separated, perform an inverse
            # polynomial interpolation step.
            use_newton_quadratic = True
            if (not math.isnan(e) and f_a != f_b and f_a != f_d and f_a != f_e and
                    f_b != f_d and f_b != f_e and f_d != f_e):
                c = _inverse_poly_zero(a, b, d, e, f_a, f_b, f_d, f_e)
                use_newton_quadratic = False
                if not a < c < b:
                    use_newton_quadratic = True
            #  Otherwise, repeats of an approximate Newton-Raphson step.
            if use_newton_quadratic:
                c = _newton_quadratic(a, b, d, f_a, f_b, f_d, n)
            f_c = f(c)
            error = math.fabs(f_c)
            if error <= etol:
                break

            # re-bracket
            e, f_e = d, f_d
            a, b, d, f_a, f_b, f_d = _update_bracket(a, b, c, f_a, f_b, f_c)
            precision = math.fabs(b - a)
        # u is the endpoint with the smallest f-value
        if math.fabs(f_a) < math.fabs(f_b):
            u, f_u = a, f_a
            A = (f_b - f_a) / (b - a)
        else:
            u, f_u = b, f_b
            A = (f_a - f_b) / (a - b)
        c = u - 2 * f_u / A
        if math.fabs(c - u) > 0.5 * (b - a):
            c = (a + b) / 2
        f_c = f(c)
        error = math.fabs(f_c)
        if error <= etol:
            break

        # re-bracket
        e, f_e = d, f_d
        a, b, d, f_a, f_b, f_d = _update_bracket(a, b, c, f_a, f_b, f_c)
        precision = math.fabs(b - a)

        # If the width of the new interval did not decrease enough, bisect
        if precision > mu * ab_width:
            e, f_e = d, f_d
            z = (a + b) / 2
            f_z = f(z)
            error = math.fabs(f_z)
            if error <= etol:
                c, f_c = z, f_z
                break
            a, b, d, f_a, f_b, f_d = _update_bracket(a, b, z, f_a, f_b, f_z)
            precision = math.fabs(b - a)

    r, f_r = c, f_c
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def toms748(f: Callable[[float], float],
            a: float,
            b: float,
            f_a: Optional[float] = None,
            f_b: Optional[float] = None,
            k: int = 2,
            mu: float = 0.5,
            etol: float = named_default(ETOL=ETOL),
            ptol: float = named_default(PTOL=PTOL),
            max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    TOMS Algorithm 748.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        k: Defaults to 2.
        mu: Defaults to 0.5.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    _check_bracket(a, b)

    if not isinstance(k, int) or k <= 0:
        raise ValueError(f'k must be positive integer. Got {k}.')
    if not 0 < mu < 1:
        raise ValueError(f'mu must be in range (0, 1). Got {mu}.')

    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = toms748_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, k, mu, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Wu
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) wu_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef double c, f_c, a_bar, f_a_bar, b_bar, f_b_bar, c_bar
    cdef double div_diff_ab, div_diff_bc, div_diff_ac, alpha, beta, delta, s_delta, d1, d2, d
    # bisect initialize
    c = (a + b) / 2
    f_c = f(c)
    step += 1
    error = math.fabs(f_c)
    if error <= etol:
        r, f_r = c, f_c
        return r, f_r, step, a, b, f_a, f_b, precision, error, True, True
    a_bar, b_bar, _, f_a_bar, f_b_bar, _ = _update_bracket(a, b, c, f_a, f_b, f_c)
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        # Muller step for calculating c_bar
        if a == b or b == c or c == a:
            converged = False
            break
        div_diff_ab = (f_b - f_a) / (b - a)
        div_diff_ac = (f_c - f_a) / (c - a)
        div_diff_bc = (f_c - f_b) / (c - b)
        beta = div_diff_ab + div_diff_ac - div_diff_bc
        alpha = (div_diff_ab - div_diff_bc) / (a - c)
        delta = beta ** 2 - 4 * alpha * f_c
        if delta < 0:
            converged = False
            break
        s_delta = math.sqrt(delta)  # \sqrt{b^2 - 4ac}
        d1, d2 = b + s_delta, b - s_delta
        # take the higher-magnitude denominator
        d = d1 if math.fabs(d1) > math.fabs(d2) else d2
        c_bar = c - 2 * f_c / d

        a, b, f_a, f_b = a_bar, b_bar, f_a_bar, f_b_bar
        if a < c_bar < b:
            c, f_c = c_bar, f(c_bar)
        else:
            c = (a + b) / 2
            f_c = f(c)
        precision = math.fabs(b - a)
        error = math.fabs(f_c)
        a_bar, b_bar, _, f_a_bar, f_b_bar, _ = _update_bracket(a, b, c, f_a, f_b, f_c)

    r, f_r = c, f_c
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def wu(f: Callable[[float], float],
       a: float,
       b: float,
       f_a: Optional[float] = None,
       f_b: Optional[float] = None,
       etol: float = named_default(ETOL=ETOL),
       ptol: float = named_default(PTOL=PTOL),
       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Wu's (Muller-Bisection) method for root-finding presented in the paper
    "Improved Muller method and Bisection method with global and asymptotic
    superlinear convergence of both point and interval for solving nonlinear equations".

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.

    References:
        https://doi.org/10.1016/j.amc.2004.04.120
    """
    # check params
    _check_bracket(a, b)
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)

    res = wu_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# ITP
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint, bint) itp_kernel(
        double_scalar_func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double k1,
        double k2,
        long n0,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    # This slightly differs from the original paper
    # where stop condition is (b - a) <= 2 * eps
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_bracket(a, b, f_a, f_b, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

    cdef long n_1div2 = <long> math.ceil(math.log2((b - a) / ptol))
    cdef long n_max = n_1div2 + n0
    cdef double x_m, f_m, rho, delta, x_f, sigma, x_t, x_itp, f_itp
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0 or f_a == f_b:
            converged = False
            break
        step += 1
        # calculate params
        x_m = (a + b) / 2
        rho = ptol * 2 ** <double> (n_max - step) - (b - a) / 2
        delta = k1 * (b - a) ** k2
        # interpolation
        x_f = (f_b * a - f_a * b) / (f_b - f_a)
        # truncation
        sigma = math.copysign(1, x_m - x_f)
        x_t = x_f + sigma * delta if delta <= math.fabs(x_m - x_f) else x_m
        # projection
        x_itp = x_t if math.fabs(x_t - x_m) <= rho else x_m - sigma * rho
        # update interval
        f_itp = f(x_itp)
        error = math.fabs(f_itp)
        if error <= etol:
            a = b = x_itp
            f_a = f_b = f_itp
            break
        elif f_itp > 0:
            b, f_b = x_itp, f_itp
        elif f_itp < 0:
            a, f_a = x_itp, f_itp
        precision = math.fabs(b - a)
    x_m = (a + b) / 2
    f_m = f(x_m)
    error = math.fabs(f_m)

    r, f_r = x_m, f_m
    optimal = error <= etol
    return r, f_r, step, a, b, f_a, f_b, precision, error, converged, optimal

# upper bound for k2
cdef double PHI = (1 + math.sqrt(5)) / 2

# noinspection DuplicatedCode
@tag('cyroot.bracketing')
@dynamic_default_args()
@cython.binding(True)
def itp(f: Callable[[float], float],
        a: float,
        b: float,
        f_a: Optional[float] = None,
        f_b: Optional[float] = None,
        k1: Optional[float] = None,
        k2: Optional[float] = 2,
        n0: Optional[int] = 1,
        etol: float = named_default(ETOL=ETOL),
        ptol: float = named_default(PTOL=PTOL),
        max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Interpolate, Truncate, and Project (ITP) method for root-finding presented in the paper
    "An Enhancement of the Bisection Method Average Performance Preserving Minmax convergedity".

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        k1: Tuning parameter in range (1, :math:`\infty`).
            Defaults to :math:`\frac{{0.2}}{{b - a}}` as suggested in the original paper.
        k2: Tuning parameter in range [1, 1 + :math:`\phi`), where :math:`\phi` is the golden ratio.
            Defaults to 2.
        n0: Tuning parameter. Defaults to 1.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {etol}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {ptol}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to {max_iter}.

    Returns:
        solution: The solution represented as a ``RootResults`` object.

    References:
        https://dl.acm.org/doi/10.1145/3423597
    """
    # check params
    _check_bracket(a, b)
    if k1 is None:
        k1 = 0.2 / (b - a)
    elif k1 <= 0:
        raise ValueError(f'k1 must be in range (1, inf). Got {k1}.')
    if not 1 <= k2 < 1 + PHI:
        raise ValueError(f'k2 must be in range [1, 1 + phi), where '
                         f'phi is the golden ratio. Got {k2}.')
    if n0 < 0:
        raise ValueError(f'n0 must be non-negative integer. Got {n0}.')

    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_a is None:
        f_a = f_wrapper(a)
    if f_b is None:
        f_b = f_wrapper(b)
    _check_unique_initial_vals(f_a, f_b)

    res = itp_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, k1, k2, n0, etol, ptol, max_iter)
    return BracketingMethodsReturnType.from_results(res, f_wrapper.n_f_calls)
