# distutils: language=c++
# cython: cdivision = True
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Sequence, Optional

import numpy as np
cimport numpy as np
from cython cimport view
from libc cimport math

from ._return_types import QuasiNewtonMethodReturnType
from ._defaults cimport ETOL, PTOL
from .fptr cimport (
    func_type, DoubleScalarFPtr, PyDoubleScalarFPtr,
    complex_func_type, ComplexScalarFPtr, PyComplexScalarFPtr
)

cdef extern from '<complex.h>':
    double complex sqrt(double complex x) nogil
    double cabs(double complex x) nogil
# from numpy.lib.scimath import sqrt

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
cdef (double, double, long, double, double, bint) secant_kernel(
        func_type f,
        double x0,
        double x1,
        double f_x0,
        double f_x1,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.fabs(x1 - x0), error = math.INFINITY
    cdef long step = 0
    cdef double x2, df_10
    cdef converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
            converged = False
            break
        step += 1
        df_10 = f_x1 - f_x0
        if df_10 == 0:
            converged = False
            break
        x2 = x0 - (x1 - x0) * f_x0 / df_10
        x0, f_x0 = x1, f_x1
        x1, f_x1 = x2, f(x2)

        precision = math.fabs(x1 - x0)
        error = math.fabs(f_x1)
    return x1, f_x1, step, precision, error, converged

# noinspection DuplicatedCode
def secant(f: Callable[[float], float],
           x0: float,
           x1: float,
           f_x0: Optional[float] = None,
           f_x1: Optional[float] = None,
           etol: float = ETOL,
           ptol: float = PTOL,
           max_iter: int = 0) -> QuasiNewtonMethodReturnType:
    """
    Secant method for root-finding.

    Args:
        f: Function for which the root is sought.
        x0: First initial point.
        x1: Second initial point.
        f_x0: Value evaluated at first initial point.
        f_x1: Value evaluated at second initial point.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to 0.

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

    if x0 == x1:
        raise ValueError(f'x0 and x1 must be different. Got {x0} and {x1}.')
    if f_x0 is None:
        f_x0 = f(x0)
    if f_x1 is None:
        f_x1 = f(x1)
    if f_x0 == f_x1:
        raise ValueError(f'f_x0 and f_x1 must be different. Got {f_x0} and {f_x1}.')

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, precision, error, converged = secant_kernel[DoubleScalarFPtr](
        f_wrapper, x0, x1, f_x0, f_x1, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       precision, error, converged)

################################################################################
# Sidi
################################################################################
cdef class NewtonPolynomial:
    cdef int n
    cdef double[:] x, a

    def __init__(self, xs: double[:], ys: double[:]):
        self.n = <int> len(xs)
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
        DD[:, 0] = ys
        cdef int i, j
        # Fill in divided differences
        for j in range(1, self.n):
            DD[:j, j] = 0
            for i in range(j, self.n):
                DD[i, j] = (DD[i, j - 1] - DD[i - 1, j - 1]) / (xs[i] - xs[i - j])
        # Copy diagonal elements into array for returning
        for j in range(self.n):
            self.a[j] = DD[j, j]

    def __call__(self, double x):
        return self.f(x)

    cdef double f(self, double x):
        cdef double f_x = self.a[-1]
        cdef int k
        for k in range(self.n - 2, -1, -1):
            f_x = f_x * (x - self.x[k]) + self.a[k]
        return f_x

    cdef inline double df(self, double x):
        return self.dnf(x, 1)

    cdef double dnf(self, double x, int order=1):
        cdef double[:] dfs = view.array(shape=(order + 1,), itemsize=sizeof(double), format='d')
        dfs[0] = self.a[-1]
        dfs[1:] = 0
        cdef int i, k
        cdef double v
        for k in range(self.n - 2, -1, -1):
            v = x - self.x[k]
            for i in range(order, 0, -1):
                dfs[i] = dfs[i] * v + dfs[i - 1]
            dfs[0] = dfs[0] * v + self.a[k]
        return dfs[-1]

# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint) sidi_kernel(
        func_type f,
        double[:] x0s,
        double[:] f_x0s,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    # sort by absolute value of f
    cdef np.ndarray[np.int64_t, ndim=1] inds = np.flip(np.argsort(np.abs(f_x0s)))
    cdef double[:] xs = np.take_along_axis(np.array(x0s), inds, 0)
    cdef double[:] f_xs = np.take_along_axis(np.array(f_x0s), inds, 0)

    cdef double precision = math.fabs(f_xs[-1] - f_xs[-2]), error = math.fabs(f_xs[-1])
    cdef long step = 0
    cdef double xn, f_xn, dp_xn
    cdef converged = True
    cdef NewtonPolynomial poly
    while error > etol and precision > ptol:
        if step > max_iter > 0:
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

        precision = math.fabs(f_xn - f_xs[-2])
        error = math.fabs(f_xn)
    return xn, f_xn, step, precision, error, converged

# noinspection DuplicatedCode
def sidi(f: Callable[[float], float],
         xs: Sequence[float],
         f_xs: Sequence[float] = None,
         etol: float = ETOL,
         ptol: float = PTOL,
         max_iter: int = 0) -> QuasiNewtonMethodReturnType:
    """
    Sidi's Generalized Secant method for root-finding.

    Args:
        f: Function for which the root is sought.
        xs: List of initial points.
        f_xs: Values evaluated at initial points.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to 0.

    Returns:
        solution: The solution represented as a ``RootResults`` object.
    """
    # check params
    if not isinstance(xs, Sequence):
        raise ValueError(f'xs must be a sequence. Got {type(xs)}.')
    if len(xs) < 2:
        raise ValueError(f'Requires at least 2 initial guesses. Got {len(xs)}.')
    if f_xs is None:
        f_xs = [f(x) for x in xs]
    elif not isinstance(f_xs, Sequence):
        raise ValueError(f'f_xs must be a sequence. Got {type(f_xs)}.')
    elif len(f_xs) != len(xs):
        raise ValueError(f'xs and f_xs must have same size. Got {len(xs)} and {len(f_xs)}.')

    if etol <= 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol <= 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    f_wrapper = PyDoubleScalarFPtr(f)
    xs = np.array(xs, dtype=np.float64)
    f_xs = np.array(f_xs, dtype=np.float64)
    r, f_r, step, precision, error, converged = sidi_kernel[DoubleScalarFPtr](
        f_wrapper, xs, f_xs, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       precision, error, converged)

################################################################################
# Steffensen
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint) steffensen_kernel(
        func_type f,
        double x0,
        double f_x0,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.INFINITY, error = math.INFINITY
    cdef long step = 0
    cdef double x1, x2, x3, denom
    cdef converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
            converged = False
            break
        step += 1
        x1 = x0 + f_x0
        x2 = x1 + f(x1)
        # Use Aitken's delta squared method to find a better approximation to x0.
        denom = x2 - 2 * x1 + x0
        if denom == 0:
            converged = False
            break
        x3 = x0 - (x1 - x0) ** 2 / (x2 - 2 * x1 + x0)
        # x3 = x2 - (x2 - x1) ** 2 / (x2 - 2 * x1 + x0)
        precision = math.fabs(x3 - x0)
        error = math.fabs(f_x0)
        x0, f_x0 = x3, f(x3)
    return x0, f_x0, step, precision, error, converged

# noinspection DuplicatedCode
def steffensen(f: Callable[[float], float],
               x0: float,
               f_x0: Optional[float] = None,
               etol: float = ETOL,
               ptol: float = PTOL,
               max_iter: int = 0) -> QuasiNewtonMethodReturnType:
    """
    Steffensen's method for root-finding.

    Args:
        f: Function for which the root is sought.
        x0: First initial point.
        f_x0: Value evaluated at first initial point.
        etol: Error tolerance, indicating the desired precision
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to 0.

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

    if f_x0 is None:
        f_x0 = f(x0)

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, precision, error, converged = steffensen_kernel[DoubleScalarFPtr](
        f_wrapper, x0, f_x0, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       precision, error, converged)

################################################################################
# Inverse Quadratic Interpolation
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint) inverse_quadratic_interp_kernel(
        func_type f,
        double x0,
        double x1,
        double x2,
        double f_x0,
        double f_x1,
        double f_x2,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.INFINITY, error = math.INFINITY
    cdef long step = 0
    cdef double x3, df_01, df_02, df_12
    cdef converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
            converged = False
            break
        step += 1
        df_01 = f_x0 - f_x1
        df_02 = f_x0 - f_x2
        df_12 = f_x1 - f_x2
        if df_01 == 0 or df_02 == 0 or df_12 == 0:
            converged = False
            break
        x3 = x0 * f_x1 * f_x2 / (df_01 * df_02)
        x3 += x1 * f_x0 * f_x2 / (-df_01 * df_12)
        x3 += x2 * f_x0 * f_x1 / (df_02 * df_12)
        x0, f_x0 = x1, f_x1
        x1, f_x1 = x2, f_x2
        x2, f_x2 = x3, f(x3)

        precision = math.fabs(x2 - x1)
        error = math.fabs(f_x2)
    return x2, f_x2, step, precision, error, converged

# noinspection DuplicatedCode
def inverse_quadratic_interp(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        x2: float,
        f_x0: Optional[float] = None,
        f_x1: Optional[float] = None,
        f_x2: Optional[float] = None,
        etol: float = ETOL,
        ptol: float = PTOL,
        max_iter: int = 0) -> QuasiNewtonMethodReturnType:
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
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to 0.

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

    if any((x0 == x1, x1 == x2, x0 == x2)):
        raise ValueError(f'x0, x1, and x2 must be different. '
                         f'Got {x0}, {x1}, and {x2}.')
    if f_x0 is None:
        f_x0 = f(x0)
    if f_x1 is None:
        f_x1 = f(x1)
    if f_x2 is None:
        f_x2 = f(x2)
    if f_x0 == f_x1 == f_x2:
        raise ValueError(f'f_x0, f_x1, and f_x2 must be different. '
                         f'Got {f_x0}, {f_x1}, and {f_x2}.')

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, precision, error, converged = inverse_quadratic_interp_kernel[DoubleScalarFPtr](
        f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       precision, error, converged)

################################################################################
# Hyperbolic Interpolation
################################################################################
# noinspection DuplicatedCode
cdef (double, double, long, double, double, bint) hyperbolic_interp_kernel(
        func_type f,
        double x0,
        double x1,
        double x2,
        double f_x0,
        double f_x1,
        double f_x2,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = math.INFINITY, error = math.INFINITY
    cdef long step = 0
    cdef double x3, d_01, d_12, df_01, df_02, df_12
    cdef converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0:
            converged = False
            break
        step += 1

        d_01 = x0 - x1
        d_12 = x1 - x2
        df_01 = f_x0 - f_x1
        df_02 = f_x0 - f_x2
        df_12 = f_x1 - f_x2
        if d_01 == 0 or d_12 == 0:
            converged = False
            break
        denom = f_x0 * df_12 / d_12 - f_x2 * df_01 / d_01
        if denom == 0:
            converged = False
            break
        x3 = x1 - f_x1 * df_02 / denom
        x0, f_x0 = x1, f_x1
        x1, f_x1 = x2, f_x2
        x2, f_x2 = x3, f(x3)

        precision = math.fabs(x2 - x1)
        error = math.fabs(f_x2)
    return x2, f_x2, step, precision, error, converged

# noinspection DuplicatedCode
def hyperbolic_interp(
        f: Callable[[float], float],
        x0: float,
        x1: float,
        x2: float,
        f_x0: Optional[float] = None,
        f_x1: Optional[float] = None,
        f_x2: Optional[float] = None,
        etol: float = ETOL,
        ptol: float = PTOL,
        max_iter: int = 0) -> QuasiNewtonMethodReturnType:
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
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to 0.

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

    if any((x0 == x1, x1 == x2, x0 == x2)):
        raise ValueError(f'x0, x1, and x2 must be different. '
                         f'Got {x0}, {x1}, and {x2}.')
    if f_x0 is None:
        f_x0 = f(x0)
    if f_x1 is None:
        f_x1 = f(x1)
    if f_x2 is None:
        f_x2 = f(x2)
    if f_x0 == f_x1 == f_x2:
        raise ValueError(f'f_x0, f_x1, and f_x2 must be different. '
                         f'Got {f_x0}, {f_x1}, and {f_x2}.')

    f_wrapper = PyDoubleScalarFPtr(f)
    r, f_r, step, precision, error, converged = hyperbolic_interp_kernel[DoubleScalarFPtr](
        f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       precision, error, converged)

################################################################################
# Muller
################################################################################
# noinspection DuplicatedCode
cdef (double complex, double complex, long, double, double, bint) muller_kernel(
        complex_func_type f,
        double complex x0,
        double complex x1,
        double complex x2,
        double complex f_x0,
        double complex f_x1,
        double complex f_x2,
        double etol=ETOL,
        double ptol=PTOL,
        long max_iter=0):
    cdef double precision = cabs(x2 - x0), error = math.INFINITY
    cdef long step = 0
    cdef double complex div_diff_01, div_diff_02, div_diff_12, a, b, c, s_delta, d1, d2, d, x3
    cdef converged = True
    while error > etol and precision > ptol:
        if step > max_iter > 0 or x0 == x1 or x1 == x2 or x0 == x2:
            converged = False
            break
        step += 1
        div_diff_01 = (f_x1 - f_x0) / (x1 - x0)
        div_diff_02 = (f_x2 - f_x0) / (x2 - x0)
        div_diff_12 = (f_x2 - f_x1) / (x2 - x1)
        c = f_x2
        b = div_diff_01 + div_diff_02 - div_diff_12
        a = (div_diff_01 - div_diff_12) / (x0 - x2)
        s_delta = sqrt(b ** 2 - 4 * a * c)  # \sqrt{b^2 - 4ac}
        d1, d2 = b + s_delta, b - s_delta
        # take the higher-magnitude denominator
        d = d1 if cabs(d1) > cabs(d2) else d2

        x3 = x2 - 2 * f_x2 / d
        x0, f_x0 = x1, f_x1
        x1, f_x1 = x2, f_x2
        x2, f_x2 = x3, f(x3)

        precision = cabs(x2 - x0)
        error = cabs(f_x2)
    return x2, f_x2, step, precision, error, converged

# noinspection DuplicatedCode
def muller(f: Callable[[complex], complex],
           x0: complex,
           x1: complex,
           x2: complex,
           f_x0: Optional[complex] = None,
           f_x1: Optional[complex] = None,
           f_x2: Optional[complex] = None,
           etol: float = ETOL,
           ptol: float = PTOL,
           max_iter: int = 0) -> QuasiNewtonMethodReturnType:
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
         of the root. Defaults to {ETOL}.
        ptol: Precision tolerance, indicating the minimum change
         of root approximations or width of brackets (in bracketing
         methods) after each iteration. Defaults to {PTOL}.
        max_iter: Maximum number of iterations. If set to 0, the
         procedure will run indefinitely until stopping condition is
         met. Defaults to 0.

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

    if any((x0 == x1, x1 == x2, x0 == x2)):
        raise ValueError(f'x0, x1, and x2 must be different. '
                         f'Got {x0}, {x1}, and {x2}.')
    if f_x0 is None:
        f_x0 = f(x0)
    if f_x1 is None:
        f_x1 = f(x1)
    if f_x2 is None:
        f_x2 = f(x2)
    if f_x0 == f_x1 == f_x2:
        raise ValueError(f'f_x0, f_x1, and f_x2 must be different. '
                         f'Got {f_x0}, {f_x1}, and {f_x2}.')

    f_wrapper = PyComplexScalarFPtr(f)
    r, f_r, step, precision, error, converged = muller_kernel[ComplexScalarFPtr](
        f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2, etol, ptol, max_iter)
    return QuasiNewtonMethodReturnType(r, f_r, step, f_wrapper.n_f_calls,
                                       precision, error, converged)
