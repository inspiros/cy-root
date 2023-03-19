# distutils: language=c++
# cython: cdivision = True
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

from cpython cimport array
from cython cimport view
from libc cimport math

from ._defaults cimport ETOL, PTOL, PHI
from ._return_types import BracketingMethodsReturnType
from .fptr cimport func_type, DoubleScalarFPtr, PyDoubleScalarFPtr

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
    'itp',
]

cdef inline (double, double, double, double, double, double)  _update_bracket(
        double a, double b, double c, double f_a, double f_b, double f_c):
    """Update a bracket given (c, f_c)."""
    cdef double rx, f_rx
    if math.copysign(1, f_a) * math.copysign(1, f_c) > 0:
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
cdef (double, double, long, double, double, double, double, double, double, bint) bisect_kernel(
        func_type f,
        double a,
        double b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0,
        stop_func_type extra_stop_criteria=NULL):
    cdef double precision = math.INFINITY, error = math.INFINITY
    cdef long step = 0
    cdef double c, f_c, f_a = math.NAN, f_b = math.NAN
    cdef bint converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
            converged = False
            break
        if extra_stop_criteria is not NULL and extra_stop_criteria(step, a, b, f_a, f_b):
            converged = False
            break
        step += 1
        c = (a + b) / 2
        f_c = f(c)
        precision = math.fabs(c - a)
        error = math.fabs(f_c)
        if f_c == 0:
            break
        elif f_c > 0:
            b, f_b = c, f_c
        else:
            a, f_a = c, f_c
    return c, f_c, step, a, b, f_a, f_b, precision, error, converged

# noinspection DuplicatedCode
def bisect(f: Callable[[float], float],
           a: float,
           b: float,
           etol: float = ETOL,
           ptol: float = PTOL,
           max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Bisection method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    a, b = min(a, b), max(a, b)
    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = bisect_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# False position
################################################################################
ctypedef double (*scale_func_type)(double, double)

# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint) regula_falsi_kernel(
        func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0,
        scale_func_type scale_func=NULL):
    cdef double precision = math.INFINITY, error = math.INFINITY
    cdef long step = 0
    cdef double c, f_c, c_prev, m = 1
    cdef bint converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0 or f_a == f_b:
            converged = False
            break
        step += 1
        c = (a * f_b - b * f_a) / (f_b - f_a)
        f_c = f(c)
        precision = math.fabs(f_c)
        error = math.fabs(c - c_prev)
        c_prev = c

        if scale_func is not NULL:
            m = scale_func(f_b, f_c)

        if f_c == 0:
            break
        elif math.copysign(1, f_b) * math.copysign(1, f_c) < 0:
            a, f_a = b, f_b
        else:
            f_a = m * f_a
        b, f_b = c, f_c
    return c, f_c, step, a, b, f_a, f_b, precision, error, converged

# noinspection DuplicatedCode
def regula_falsi(f: Callable[[float], float],
                 a: float,
                 b: float,
                 f_a: Optional[float] = None,
                 f_b: Optional[float] = None,
                 etol: float = ETOL,
                 ptol: float = PTOL,
                 max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Regula Falsi (or False Position) method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# Illinois
################################################################################
cdef inline double illinois_scale(double f_b, double f_c):
    return 0.5

# noinspection DuplicatedCode
def illinois(f: Callable[[float], float],
             a: float,
             b: float,
             f_a: Optional[float] = None,
             f_b: Optional[float] = None,
             etol: float = ETOL,
             ptol: float = PTOL,
             max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Illinois method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter, illinois_scale)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# Pegasus
################################################################################
cdef inline double pegasus_scale(double f_b, double f_c):
    return f_b / (f_b + f_c)

# noinspection DuplicatedCode
def pegasus(f: Callable[[float], float],
            a: float,
            b: float,
            f_a: Optional[float] = None,
            f_b: Optional[float] = None,
            etol: float = ETOL,
            ptol: float = PTOL,
            max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Pegasus method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter, pegasus_scale)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# Anderson–Björck
################################################################################
cdef inline double anderson_bjorck_scale(double f_b, double f_c):
    m = 1 - f_c / f_b
    if m <= 0:
        return 0.5
    return m

# noinspection DuplicatedCode
def anderson_bjorck(f: Callable[[float], float],
                    a: float,
                    b: float,
                    f_a: Optional[float] = None,
                    f_b: Optional[float] = None,
                    etol: float = ETOL,
                    ptol: float = PTOL,
                    max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Anderson–Björck method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = regula_falsi_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter, anderson_bjorck_scale)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# Dekker
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint) dekker_kernel(
        func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.fabs(b - a), error = math.INFINITY
    cdef long step = 0
    cdef double b_prev = a, f_b_prev = f_a, b_next, f_b_next, c, s
    cdef bint converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
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

        precision = math.fabs(a - b)
        error = math.fabs(f_b)
    return b, f_b, step, a, b, f_a, f_b, precision, error, converged

# noinspection DuplicatedCode
def dekker(f: Callable[[float], float],
           a: float,
           b: float,
           f_a: Optional[float] = None,
           f_b: Optional[float] = None,
           etol: float = ETOL,
           ptol: float = PTOL,
           max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Dekker's method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = dekker_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# Brent
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint) brent_kernel(
        func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        int interp_method=1,
        double sigma=1e-6,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.fabs(b - a), error = math.INFINITY
    cdef long step = 0
    cdef double b_prev = a, f_b_prev = f_a, b_prev_prev = a, b_next, f_b_next, c, s
    cdef double d_ab, d_abp, d_bbp, df_ab, df_abp, df_bbp, denom
    cdef int last_method = 0  # 0 for bisect, 1 for interp
    cdef bint use_interp, converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
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
            s = a * f_b * f_b_prev / (df_ab * df_abp)
            s += b * f_a * f_b_prev / (-df_ab * df_bbp)
            s += b_prev * f_a * f_b / (df_abp * df_bbp)
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

        precision = math.fabs(a - b)
        error = math.fabs(f_b)
    return b, f_b, step, a, b, f_a, f_b, precision, error, converged

BRENT_INTERP_METHODS: dict[str, int] = {
    'quadratic': 0,
    'hyperbolic': 1,
}

# noinspection DuplicatedCode
def brent(f: Callable[[float], float],
          a: float,
          b: float,
          f_a: Optional[float] = None,
          f_b: Optional[float] = None,
          interp_method: Union[str, int] = 'hyperbolic',
          sigma: float = 1e-5,
          etol: float = ETOL,
          ptol: float = PTOL,
          max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
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
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if ((isinstance(interp_method, int) and interp_method not in BRENT_INTERP_METHODS.values()) or
            (isinstance(interp_method, str) and interp_method not in BRENT_INTERP_METHODS.keys())):
        raise ValueError(f'interp_method can be {BRENT_INTERP_METHODS}. '
                         f'No implementation found for {interp_method}.')
    if isinstance(interp_method, str):
        interp_method = BRENT_INTERP_METHODS[interp_method]
    if sigma <= 0:
        raise ValueError(f'sigma must be positive. Got {sigma}.')

    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wraper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = brent_kernel[DoubleScalarFPtr](
        f_wraper, a, b, f_a, f_b, interp_method, sigma, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wraper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# Chandrupatla
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint) chandrupatla_kernel(
        func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double sigma=1e-5,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.fabs(b - a), error = math.INFINITY
    cdef long step = 0
    cdef double c, f_c, d, f_d, d_ab, d_fa_fb, d_ad, d_fa_fd, d_bd, d_fb_fd, tl, t = 0.5
    cdef double r, f_r
    cdef bint converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
            converged = False
            break
        step += 1
        c = a + (b - a) * t
        f_c = f(c)
        if math.copysign(1, f_c) * math.copysign(1, f_a) > 0:
            d, a = a, c
            f_d, f_a = f_a, f_c
        else:
            d, b, a = b, a, c
            f_d, f_b, f_a = f_b, f_a, f_c
        precision = math.fabs(a - b)
        r, f_r = (a, f_a) if math.fabs(f_a) < math.fabs(f_b) else (b, f_b)
        error = math.fabs(f_r)
        tl = (2 * ptol * math.fabs(r) + 0.5 * sigma) / precision
        if f_r == 0 or tl > 0.5:
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

    return r, f_r, step, a, b, f_a, f_b, precision, error, converged

# noinspection DuplicatedCode
def chandrupatla(f: Callable[[float], float],
                 a: float,
                 b: float,
                 f_a: Optional[float] = None,
                 f_b: Optional[float] = None,
                 sigma: float = 1e-5,
                 etol: float = ETOL,
                 ptol: float = PTOL,
                 max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Chandrupatla's method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        sigma: Tolerance (not sure what it's for). Defaults to 1e-5.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = chandrupatla_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, sigma, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# Ridders
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint) ridders_kernel(
        func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.fabs(b - a), error = math.INFINITY
    cdef long step = 0
    cdef double c, f_c, d, f_d, denom
    cdef bint converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
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
    return d, f_d, step, a, b, f_a, f_b, precision, error, converged

# noinspection DuplicatedCode
def ridders(f: Callable[[float], float],
            a: float,
            b: float,
            f_a: Optional[float] = None,
            f_b: Optional[float] = None,
            etol: float = ETOL,
            ptol: float = PTOL,
            max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Ridders method for root-finding.

    Args:
        f: Function for which the root is sought.
        a: Lower bound of the interval to be searched.
        b: Upper bound of the interval to be searched.
        f_a: Value evaluated at lower bound.
        f_b: Value evaluated at upper bound.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)
    if f_a * f_b >= 0:
        raise ValueError('f_a and f_b must have opposite sign.')

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = ridders_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

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
cdef (double, double, long, double, double, double, double, double, double, bint) toms748_kernel(
        func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        long k=2,
        double mu=0.5,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.fabs(b - a)
    cdef long step = 0, n
    cdef double c, f_c, d, f_d, e, f_e, u, f_u, z, f_z, A
    cdef bint use_newton_quadratic, converged = True
    # secant init
    c = (a - b * f_a / f_b) / (1 - f_a / f_b)
    if not a < c < b:
        c = (a + b) / 2
    f_c = f(c)
    step += 1
    cdef double error = math.fabs(f_c)
    if f_c == 0:
        return c, f_c, step, a, b, f_a, f_b, precision, error, True
    a, b, d, f_a, f_b, f_d = _update_bracket(a, b, c, f_a, f_b, f_c)
    e = f_e = math.NAN
    while error > etol and precision > ptol:
        if step > max_iter > 0:
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
            if f_c == 0:
                break

            # re-bracket
            e, f_e = d, f_d
            a, b, d, f_a, f_b, f_d = _update_bracket(a, b, c, f_a, f_b, f_c)
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
        if f_c == 0:
            break

        # re-bracket
        e, f_e = d, f_d
        a, b, d, f_a, f_b, f_d = _update_bracket(a, b, c, f_a, f_b, f_c)

        # If the width of the new interval did not decrease enough, bisect
        if b - a > mu * ab_width:
            e, f_e = d, f_d
            z = (a + b) / 2
            f_z = f(z)
            if f_z == 0:
                c, f_c = z, f_z
                break
            a, b, d, f_a, f_b, f_d = _update_bracket(a, b, z, f_a, f_b, f_z)

        precision = math.fabs(b - a)
        error = math.fabs(f_c)
    return c, f_c, step, a, b, f_a, f_b, precision, error, converged

# noinspection DuplicatedCode
def toms748(f: Callable[[float], float],
            a: float,
            b: float,
            f_a: Optional[float] = None,
            f_b: Optional[float] = None,
            k: int = 2,
            mu: float = 0.5,
            etol: float = ETOL,
            ptol: float = PTOL,
            max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
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
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f'k must be positive integer. Got {k}.')
    if not 0 < mu < 1:
        raise ValueError(f'mu must be in range (0, 1). Got {mu}.')

    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)
    if f_a * f_b >= 0:
        raise ValueError('f_a and f_b must have opposite sign.')

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = toms748_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, k, mu, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)

################################################################################
# ITP
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, double, double, double, double, bint) itp_kernel(
        func_type f,
        double a,
        double b,
        double f_a,
        double f_b,
        double k1,
        double k2,
        long n0,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    # preprocessing
    cdef double precision = (b - a) / 2, error = math.INFINITY
    cdef long n_1div2 = <long> math.ceil(math.log2((b - a) / (2 * ptol)))
    cdef long n_max = n_1div2 + n0
    cdef long step = 0
    cdef double x_m, f_m, r, delta, x_f, sigma, x_t, x_itp, f_itp
    cdef bint converged = True
    while precision > ptol:
        if step > max_iter > 0:
            converged = False
            break
        step += 1
        # calculate params
        x_m = (a + b) / 2
        r = ptol * 2 ** <double> (n_max - step) - (b - a) / 2
        delta = k1 * (b - a) ** k2
        # interpolation
        x_f = (f_b * a - f_a * b) / (f_b - f_a)
        # truncation
        sigma = math.copysign(1, x_m - x_f)
        x_t = x_f + sigma * delta if delta <= math.fabs(x_m - x_f) else x_m
        # projection
        x_itp = x_t if math.fabs(x_t - x_m) <= r else x_m - sigma * r
        # update interval
        f_itp = f(x_itp)
        if f_itp > 0:
            b = x_itp
            f_b = f_itp
        elif f_itp < 0:
            a = x_itp
            f_a = f_itp
        else:
            a = b = x_itp
        precision = (b - a) / 2
        # error = math.fabs(f_itp)
    x_m = (a + b) / 2
    f_m = f(x_m)
    precision = (b - a) / 2
    error = math.fabs(f_m)
    return x_m, f_m, step, a, b, f_a, f_b, precision, error, converged

# noinspection DuplicatedCode
def itp(f: Callable[[float], float],
        a: float,
        b: float,
        f_a: Optional[float] = None,
        f_b: Optional[float] = None,
        k1: Optional[float] = None,
        k2: Optional[float] = 2,
        n0: Optional[int] = 1,
        etol: float = ETOL,
        ptol: float = PTOL,
        max_iter: int = 0) -> BracketingMethodsReturnType:
    f"""
    Implementation of the ITP (Interpolate Truncate and Project) method for root-finding
    presented in the paper "An Enhancement of the Bisection Method Average Performance Preserving Minmax convergedity".

    References:
        https://dl.acm.org/doi/10.1145/3423597

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
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if k1 is None:
        k1 = 0.2 / (b - a)
    elif k1 <= 0:
        raise ValueError(f'k1 must be in range (1, inf). Got {k1}.')
    if k2 is None:
        k2 = 2
    elif not 1 <= k2 < 1 + PHI:
        raise ValueError(f'k2 must be in range [1, 1 + phi), where '
                         f'phi is the golden ratio. Got {k2}.')
    if n0 is None:
        n0 = 1
    elif n0 < 0:
        raise ValueError(f'n0 must be non-negative integer. Got {n0}.')

    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if f_a is None:
        f_a = f(a)
    if f_b is None:
        f_b = f(b)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, a, b, f_a, f_b, precision, error, converged = itp_kernel[DoubleScalarFPtr](
        f_wrapper, a, b, f_a, f_b, k1, k2, n0, etol, ptol, max_iter)
    return BracketingMethodsReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       a, b, f_a, f_b, precision, error, converged)
