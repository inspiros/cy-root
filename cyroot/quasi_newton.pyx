# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Sequence, Optional

import cython
import numpy as np
cimport numpy as np
from cython cimport view
from libc cimport math

from .utils.array_ops cimport fabs, fabs_width, cabs_width, argsort, permute
from ._check_args cimport (
    _check_stop_condition_initial_guess,
    _check_stop_condition_initial_guesses,
    _check_stop_condition_initial_guesses_complex,
)
from ._check_args import (
    _check_stop_condition_args,
    _check_unique_initial_guesses,
    _check_unique_initial_vals,
)
from ._defaults import ETOL, PTOL, MAX_ITER
from ._return_types import QuasiNewtonMethodReturnType
from .fptr cimport (
    double_scalar_func_type, DoubleScalarFPtr, PyDoubleScalarFPtr,
    complex_scalar_func_type, ComplexScalarFPtr, PyComplexScalarFPtr
)
from .utils.dynamic_default_args import dynamic_default_args, named_default
from .utils.function_tagging import tag

cdef extern from '<complex>':
    double complex sqrt(double complex x) nogil
    double abs(double complex) nogil

__all__ = [
    'secant',
    'sidi',
    'steffensen',
    'inverse_quadratic_interp',
    'hyperbolic_interp',
    'muller',
]

################################################################################
# Secant
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint, bint) secant_kernel(
        double_scalar_func_type f,
        double x0,
        double x1,
        double f_x0,
        double f_x1,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    cdef double[2] xs = [x0, x1], f_xs = [f_x0, f_x1]
    if _check_stop_condition_initial_guesses(xs, f_xs, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, precision, error, converged, optimal

    cdef double x2, df_01
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        df_01 = f_x0 - f_x1
        if df_01 == 0:
            converged = False
            break
        x2 = x0 - f_x0 * (x0 - x1) / df_01
        x0, f_x0 = x1, f_x1
        x1, f_x1 = x2, f(x2)

        precision = math.fabs(x1 - x0)
        error = math.fabs(f_x1)

    r, f_r = x1, f_x1
    optimal = error <= etol
    return r, f_r, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def secant(f: Callable[[float], float],
           x0: float,
           x1: float,
           f_x0: Optional[float] = None,
           f_x1: Optional[float] = None,
           etol: float = named_default(ETOL=ETOL),
           ptol: float = named_default(PTOL=PTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Secant method for root-finding.

    Args:
        f: Function for which the root is sought.
        x0: First initial point.
        x1: Second initial point.
        f_x0: Value evaluated at first initial point.
        f_x1: Value evaluated at second initial point.
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
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)
    _check_unique_initial_guesses(x0, x1)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_x0 is None:
        f_x0 = f_wrapper(x0)
    if f_x1 is None:
        f_x1 = f_wrapper(x1)
    _check_unique_initial_vals(f_x0, f_x1)

    res = secant_kernel[DoubleScalarFPtr](
        f_wrapper, x0, x1, f_x0, f_x1, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Sidi
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint, bint) sidi_kernel(
        double_scalar_func_type f,
        double[:] x0s,
        double[:] f_x0s,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    # sort by absolute value of f
    cdef long[:] inds = argsort(fabs(f_x0s), reverse=<bint> True)
    cdef double[:] xs = permute(x0s, inds)
    cdef double[:] f_xs = permute(f_x0s, inds)

    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guesses(xs, f_xs, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, precision, error, converged, optimal

    cdef double xn, f_xn, dp_xn
    cdef NewtonPolynomial poly
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        poly = NewtonPolynomial(xs, f_xs)

        dp_xn = poly.dnf(xs[-1], 1)
        if dp_xn == 0:
            converged = False
            break
        xn = xs[-1] - f_xs[-1] / dp_xn
        f_xn = f(xn)
        # remove x0 and add xn
        xs[:-1] = xs[1:]
        xs[-1] = xn
        f_xs[:-1] = f_xs[1:]
        f_xs[-1] = f_xn

        precision = fabs_width(xs)
        error = math.fabs(f_xn)

    r, f_r = xn, f_xn
    optimal = error <= etol
    return r, f_r, step, precision, error, converged, optimal

cdef class NewtonPolynomial:
    cdef unsigned int n
    cdef double[:] x, a

    def __cinit__(self, xs: double[:], ys: double[:]):
        self.n = <unsigned int> xs.shape[0]
        self.x = view.array(shape=(self.n - 1,),
                            itemsize=sizeof(double),
                            format='d')
        self.x[:] = xs[:-1]
        self.a = view.array(shape=(self.n,),
                            itemsize=sizeof(double),
                            format='d')

        cdef double[:, :] DD = view.array(shape=(self.n, self.n),
                                          itemsize=sizeof(double),
                                          format='d')
        cdef int i, j
        # Fill in divided differences
        with nogil:
            DD[:, 0] = ys
            for j in range(1, self.n):
                DD[:j, j] = 0
                for i in range(j, self.n):
                    DD[i, j] = (DD[i, j - 1] - DD[i - 1, j - 1]) / (xs[i] - xs[i - j])
            # Copy diagonal elements into array for returning
            for j in range(self.n):
                self.a[j] = DD[j, j]

    def __call__(self, double x):
        return self.f(x)

    cdef inline double f(self, double x) nogil:
        cdef double f_x = self.a[-1]
        cdef unsigned int k
        for k in range(self.n - 2, -1, -1):
            f_x = f_x * (x - self.x[k]) + self.a[k]
        return f_x

    cdef inline double df(self, double x) nogil:
        return self.dnf(x, 1)

    cdef inline double dnf(self, double x, int order=1) nogil:
        cdef double[:] dfs
        with gil:
            dfs = view.array(shape=(order + 1,), itemsize=sizeof(double), format='d')
        dfs[0] = self.a[-1]
        dfs[1:] = 0
        cdef unsigned int i, k
        cdef double v
        for k in range(self.n - 2, -1, -1):
            v = x - self.x[k]
            for i in range(order, 0, -1):
                dfs[i] = dfs[i] * v + dfs[i - 1]
            dfs[0] = dfs[0] * v + self.a[k]
        return dfs[-1]

# noinspection DuplicatedCode
@tag('cyroot.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def sidi(f: Callable[[float], float],
         xs: Sequence[float],
         f_xs: Sequence[float] = None,
         etol: float = named_default(ETOL=ETOL),
         ptol: float = named_default(PTOL=PTOL),
         max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Sidi's Generalized Secant method for root-finding.

    Args:
        f: Function for which the root is sought.
        xs: List of initial points.
        f_xs: Values evaluated at initial points.
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
    if not isinstance(xs, Sequence):
        raise ValueError(f'xs must be a sequence. Got {type(xs)}.')
    if len(xs) < 2:
        raise ValueError(f'Requires at least 2 initial guesses. Got {len(xs)}.')

    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_xs is None:
        f_xs = [f_wrapper(x) for x in xs]
    elif not isinstance(f_xs, Sequence):
        raise ValueError(f'f_xs must be a sequence. Got {type(f_xs)}.')
    elif len(f_xs) != len(xs):
        raise ValueError(f'xs and f_xs must have same size. Got {len(xs)} and {len(f_xs)}.')

    xs = np.array(xs, dtype=np.float64)
    f_xs = np.array(f_xs, dtype=np.float64)
    res = sidi_kernel[DoubleScalarFPtr](f_wrapper, xs, f_xs, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Steffensen
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint, bint) steffensen_kernel(
        double_scalar_func_type f,
        double x0,
        double f_x0,
        bint adsp=True,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess(x0, f_x0, etol, ptol,
                            &precision, &error, &converged, &optimal):
        return x0, f_x0, step, precision, error, converged, optimal

    cdef double x1, x2, x3, denom
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        x1 = x0 + f_x0
        x2 = x1 + f(x1)
        denom = x2 - 2 * x1 + x0
        if denom == 0:
            converged = False
            break
        # Use Aitken's delta-squared method to find a better approximation
        if adsp:
            x3 = x0 - (x1 - x0) ** 2 / denom
        else:
            x3 = x2 - (x2 - x1) ** 2 / denom
        precision = math.fabs(x3 - x0)
        x0, f_x0 = x3, f(x3)
        error = math.fabs(f_x0)

    optimal = error <= etol
    return x0, f_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def steffensen(f: Callable[[float], float],
               x0: float,
               f_x0: Optional[float] = None,
               adsp: bool = True,
               etol: float = named_default(ETOL=ETOL),
               ptol: float = named_default(PTOL=PTOL),
               max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Steffensen's method for root-finding.

    Args:
        f: Function for which the root is sought.
        x0: First initial point.
        f_x0: Value evaluated at first initial point.
        adsp: Use Aitken's delta-squared process or not.
         Defaults to True.
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
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_x0 is None:
        f_x0 = f_wrapper(x0)

    res = steffensen_kernel[DoubleScalarFPtr](
        f_wrapper, x0, f_x0, adsp, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Inverse Quadratic Interpolation
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint, bint) inverse_quadratic_interp_kernel(
        double_scalar_func_type f,
        double x0,
        double x1,
        double x2,
        double f_x0,
        double f_x1,
        double f_x2,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    cdef double[3] x_arr = [x0, x1, x2], f_arr = [f_x0, f_x1, f_x2]
    cdef double[:] xs = x_arr, f_xs = f_arr
    if _check_stop_condition_initial_guesses(xs, f_xs, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, precision, error, converged, optimal

    cdef double x3, df_01, df_02, df_12
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        df_01 = f_xs[0] - f_xs[1]
        df_02 = f_xs[0] - f_xs[2]
        df_12 = f_xs[1] - f_xs[2]
        if df_01 == 0 or df_02 == 0 or df_12 == 0:
            converged = False
            break
        x3 = (xs[0] * f_xs[1] * f_xs[2] / (df_01 * df_02)
              + xs[1] * f_xs[0] * f_xs[2] / (-df_01 * df_12)
              + xs[2] * f_xs[0] * f_xs[1] / (df_02 * df_12))
        xs[0], f_xs[0] = xs[1], f_xs[1]
        xs[1], f_xs[1] = xs[2], f_xs[2]
        xs[2], f_xs[2] = x3, f(x3)

        precision = fabs_width(xs)
        error = math.fabs(f_xs[2])

    r, f_r = xs[2], f_xs[2]
    optimal = error <= etol
    return r, f_r, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def inverse_quadratic_interp(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        x2: float,
        f_x0: Optional[float] = None,
        f_x1: Optional[float] = None,
        f_x2: Optional[float] = None,
        etol: float = named_default(ETOL=ETOL),
        ptol: float = named_default(PTOL=PTOL),
        max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Inverse Quadratic Interpolation method for root-finding.

    Args:
        f: Function for which the root is sought.
        x0: First initial point.
        x1: Second initial point.
        x2: Third initial point.
        f_x0: Value evaluated at first initial point.
        f_x1: Value evaluated at second initial point.
        f_x2: Value evaluated at third initial point.
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
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)
    _check_unique_initial_guesses(x0, x1, x2)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_x0 is None:
        f_x0 = f_wrapper(x0)
    if f_x1 is None:
        f_x1 = f_wrapper(x1)
    if f_x2 is None:
        f_x2 = f_wrapper(x2)
    _check_unique_initial_vals(f_x0, f_x1, f_x2)

    res = inverse_quadratic_interp_kernel[DoubleScalarFPtr](
        f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Hyperbolic Interpolation
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint, bint) hyperbolic_interp_kernel(
        double_scalar_func_type f,
        double x0,
        double x1,
        double x2,
        double f_x0,
        double f_x1,
        double f_x2,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    cdef double[3] x_arr = [x0, x1, x2], f_arr = [f_x0, f_x1, f_x2]
    cdef double[:] xs = x_arr, f_xs = f_arr
    if _check_stop_condition_initial_guesses(xs, f_xs, etol, ptol,
                              &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, precision, error, converged, optimal

    cdef double x3, d_01, d_12, df_01, df_02, df_12
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        d_01 = xs[0] - xs[1]
        d_12 = xs[1] - xs[2]
        df_01 = f_xs[0] - f_xs[1]
        df_02 = f_xs[0] - f_xs[2]
        df_12 = f_xs[1] - f_xs[2]
        if d_01 == 0 or d_12 == 0:
            converged = False
            break
        denom = f_xs[0] * df_12 / d_12 - f_xs[2] * df_01 / d_01
        if denom == 0:
            converged = False
            break
        x3 = xs[1] - f_xs[1] * df_02 / denom
        xs[0], f_xs[0] = xs[1], f_xs[1]
        xs[1], f_xs[1] = xs[2], f_xs[2]
        xs[2], f_xs[2] = x3, f(x3)

        precision = fabs_width(xs)
        error = math.fabs(f_xs[2])

    r, f_r = xs[2], f_xs[2]
    optimal = error <= etol
    return r, f_r, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def hyperbolic_interp(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        x2: float,
        f_x0: Optional[float] = None,
        f_x1: Optional[float] = None,
        f_x2: Optional[float] = None,
        etol: float = named_default(ETOL=ETOL),
        ptol: float = named_default(PTOL=PTOL),
        max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Hyperbolic Interpolation method for root-finding.

    Args:
        f: Function for which the root is sought.
        x0: First initial point.
        x1: Second initial point.
        x2: Third initial point.
        f_x0: Value evaluated at first initial point.
        f_x1: Value evaluated at second initial point.
        f_x2: Value evaluated at third initial point.
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
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)
    _check_unique_initial_guesses(x0, x1, x2)

    f_wrapper = PyDoubleScalarFPtr(f)
    if f_x0 is None:
        f_x0 = f_wrapper(x0)
    if f_x1 is None:
        f_x1 = f_wrapper(x1)
    if f_x2 is None:
        f_x2 = f_wrapper(x2)
    _check_unique_initial_vals(f_x0, f_x1, f_x2)

    res = hyperbolic_interp_kernel[DoubleScalarFPtr](
        f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, f_wrapper.n_f_calls)

################################################################################
# Muller
################################################################################
# noinspection DuplicatedCode
cdef (double complex, double complex, long, double, double, bint, bint) muller_kernel(
        complex_scalar_func_type f,
        double complex x0,
        double complex x1,
        double complex x2,
        double complex f_x0,
        double complex f_x1,
        double complex f_x2,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=MAX_ITER):
    cdef long step = 0
    cdef double complex r, f_r
    cdef double precision, error
    cdef bint converged, optimal
    cdef double complex[3] x_arr = [x0, x1, x2], f_arr = [f_x0, f_x1, f_x2]
    cdef double complex[:] xs = x_arr, f_xs = f_arr
    if _check_stop_condition_initial_guesses_complex(xs, f_xs, etol, ptol,
                                      &r, &f_r, &precision, &error, &converged, &optimal):
        return r, f_r, step, precision, error, converged, optimal

    cdef double complex div_diff_01, div_diff_12, div_diff_02, a, b, s_delta, d1, d2, d, x3
    cdef double complex d_01, d_02, d_12
    converged = True
    while error > etol and precision > ptol:
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        d_01 = xs[0] - xs[1]
        d_02 = xs[0] - xs[2]
        d_12 = xs[1] - xs[2]
        if d_01 == 0 or d_02 == 0 or d_12 == 0:
            converged = False
            break

        div_diff_01 = (f_xs[0] - f_xs[1]) / d_01
        div_diff_02 = (f_xs[0] - f_xs[2]) / d_02
        div_diff_12 = (f_xs[1] - f_xs[2]) / d_12
        b = div_diff_01 + div_diff_02 - div_diff_12
        a = (div_diff_01 - div_diff_12) / d_02
        s_delta = sqrt(b ** 2 - 4 * a * f_xs[2])  # \sqrt{b^2 - 4ac}
        d1, d2 = b + s_delta, b - s_delta
        # take the higher-magnitude denominator
        d = d1 if abs(d1) > abs(d2) else d2

        x3 = xs[2] - 2 * f_xs[2] / d
        xs[0], f_xs[0] = xs[1], f_xs[1]
        xs[1], f_xs[1] = xs[2], f_xs[2]
        xs[2], f_xs[2] = x3, f(x3)

        precision = cabs_width(xs)
        error = abs(f_xs[2])

    r, f_r = xs[2], f_xs[2]
    optimal = error <= etol
    return r, f_r, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def muller(f: Callable[[complex], complex],
           x0: complex,
           x1: complex,
           x2: complex,
           f_x0: Optional[complex] = None,
           f_x1: Optional[complex] = None,
           f_x2: Optional[complex] = None,
           etol: float = named_default(ETOL=ETOL),
           ptol: float = named_default(PTOL=PTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Muller's method for root-finding.

    Args:
        f: Function for which the root is sought.
        x0: First initial point.
        x1: Second initial point.
        x2: Third initial point.
        f_x0: Value evaluated at first initial point.
        f_x1: Value evaluated at second initial point.
        f_x2: Value evaluated at third initial point.
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
    etol, ptol, max_iter = _check_stop_condition_args(etol, ptol, max_iter)
    _check_unique_initial_guesses(x0, x1, x2)

    f_wrapper = PyComplexScalarFPtr(f)
    if f_x0 is None:
        f_x0 = f_wrapper(x0)
    if f_x1 is None:
        f_x1 = f_wrapper(x1)
    if f_x2 is None:
        f_x2 = f_wrapper(x2)
    _check_unique_initial_vals(f_x0, f_x1, f_x2)

    res = muller_kernel[ComplexScalarFPtr](
        f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, f_wrapper.n_f_calls)
