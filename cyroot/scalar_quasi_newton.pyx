# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional

import cython
import numpy as np
cimport numpy as np
from cython cimport view
from dynamic_default_args import dynamic_default_args, named_default
from libc cimport math

from ._check_args cimport (
    _check_stop_cond_scalar_initial_guess,
    _check_stop_cond_scalar_initial_guesses,
    _check_stop_cond_complex_scalar_initial_guesses,
)
from ._check_args import (
    _check_stop_cond_args,
    _check_initial_guesses_uniqueness,
    _check_initial_vals_uniqueness,
)
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from ._types import VectorLike
from .fptr cimport (
    DoubleScalarFPtr, PyDoubleScalarFPtr,
    ComplexScalarFPtr, PyComplexScalarFPtr
)
from .ops cimport scalar_ops as sops, vector_ops as vops
from .return_types cimport RootReturnType
from .utils._function_registering import register

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
cdef RootReturnType secant_kernel(
        DoubleScalarFPtr f,
        double x0,
        double x1,
        double f_x0,
        double f_x1,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    cdef double[2] xs = [x0, x1], f_xs = [f_x0, f_x1]
    if _check_stop_cond_scalar_initial_guesses(xs, f_xs, etol, ertol, ptol, prtol,
                                               &r, &f_r, &precision, &error, &converged, &optimal):
        return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

    cdef double x2, df_01
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
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
        x1, f_x1 = x2, f.eval(x2)

        precision = math.fabs(x1 - x0)
        error = math.fabs(f_x1)

    r, f_r = x1, f_x1
    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def secant(f: Callable[[float], float],
           x0: float,
           x1: float,
           f_x0: Optional[float] = None,
           f_x1: Optional[float] = None,
           etol: float = named_default(ETOL=ETOL),
           ertol: float = named_default(ERTOL=ERTOL),
           ptol: float = named_default(PTOL=PTOL),
           prtol: float = named_default(PRTOL=PRTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Secant method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        x0 (float): First initial point.
        x1 (float): Second initial point.
        f_x0 (float, optional): Value evaluated at first initial point.
        f_x1 (float, optional): Value evaluated at second initial point.
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
    _check_initial_guesses_uniqueness((x0, x1))

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if f_x1 is None:
        f_x1 = f_wrapper.eval(x1)
    _check_initial_vals_uniqueness((f_x0, f_x1))

    res = secant_kernel(f_wrapper, x0, x1, f_x0, f_x1,
                        etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Sidi
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType sidi_kernel(
        DoubleScalarFPtr f,
        double[:] x0s,
        double[:] f_x0s,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    if _check_stop_cond_scalar_initial_guesses(x0s, f_x0s, etol, ertol, ptol, prtol,
                                               &r, &f_r, &precision, &error, &converged, &optimal):
        return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

    # sort by error of f
    cdef unsigned long[:] inds = vops.argsort(vops.fabs(f_x0s), reverse=<bint> True)
    cdef double[:] xs = vops.permute(x0s, inds)
    cdef double[:] f_xs = vops.permute(f_x0s, inds)

    cdef double xn, f_xn, dp_xn
    cdef double[:] dfs = view.array(shape=(1 + 1,), itemsize=sizeof(double), format='d')
    cdef NewtonPolynomial poly = NewtonPolynomial(x0s.shape[0])
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        poly.update(xs, f_xs)

        dp_xn = poly.dnf(xs[-1], 1, dfs)
        if dp_xn == 0:
            converged = False
            break
        xn = xs[-1] - f_xs[-1] / dp_xn
        f_xn = f.eval(xn)
        # remove x0 and add xn
        xs[:-1] = xs[1:]
        xs[-1] = xn
        f_xs[:-1] = f_xs[1:]
        f_xs[-1] = f_xn

        precision = vops.width(xs)
        error = math.fabs(f_xn)

    r, f_r = xn, f_xn
    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

cdef class NewtonPolynomial:
    cdef unsigned int n
    cdef double[:] x, a
    cdef double[:, :] DD

    def __cinit__(self, unsigned int n):
        self.n = n
        self.x = view.array(shape=(self.n - 1,),
                            itemsize=sizeof(double),
                            format='d')
        self.a = view.array(shape=(self.n,),
                            itemsize=sizeof(double),
                            format='d')

        self.DD = view.array(shape=(self.n, self.n),
                             itemsize=sizeof(double),
                             format='d')

    cdef void update(self, double[:] xs, double[:] ys) nogil:
        self.x[:] = xs[:-1]

        cdef unsigned int i, j
        # Fill in divided differences
        self.DD[:, 0] = ys
        for j in range(1, self.n):
            self.DD[:j, j] = 0
            for i in range(j, self.n):
                self.DD[i, j] = (self.DD[i, j - 1] - self.DD[i - 1, j - 1]) / (xs[i] - xs[i - j])
        # Copy diagonal elements into array for returning
        for j in range(self.n):
            self.a[j] = self.DD[j, j]

    def __call__(self, double x):
        return self.f(x)

    cdef inline double f(self, double x) nogil:
        cdef double f_x = self.a[-1]
        cdef unsigned int k
        for k in range(self.n - 2, -1, -1):
            f_x = f_x * (x - self.x[k]) + self.a[k]
        return f_x

    cdef inline double df(self, double x, double[:] out) nogil:
        return self.dnf(x, 1, out)

    cdef inline double dnf(self, double x, int order, double[:] out) nogil:
        out[0] = self.a[-1]
        out[1:] = 0
        cdef unsigned int i, k
        cdef double v
        for k in range(self.n - 2, -1, -1):
            v = x - self.x[k]
            for i in range(order, 0, -1):
                out[i] = out[i] * v + out[i - 1]
            out[0] = out[0] * v + self.a[k]
        return out[-1]

# noinspection DuplicatedCode
@register('cyroot.scalar.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def sidi(f: Callable[[float], float],
         x0s: VectorLike,
         f_x0s: Optional[VectorLike] = None,
         etol: float = named_default(ETOL=ETOL),
         ertol: float = named_default(ERTOL=ERTOL),
         ptol: float = named_default(PTOL=PTOL),
         prtol: float = named_default(PRTOL=PRTOL),
         max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Sidi's Generalized Secant method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        x0s (tuple of float): Tuple of initial points.
        f_x0s (tuple of float, optional): Tuple of values evaluated at initial points.
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

    x0s = np.asarray(x0s, dtype=np.float64)
    if x0s.shape[0] < 2:
        raise ValueError('Requires at least 2 initial guesses. '
                         f'Got {x0s.shape[0]}.')
    _check_initial_guesses_uniqueness(x0s)

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if f_x0s is None:
        f_x0s = np.array([f_wrapper.eval(x) for x in x0s], dtype=np.float64)
    else:
        f_x0s = np.asarray(f_x0s, dtype=np.float64)
        if x0s.shape[0] != f_x0s.shape[0]:
            raise ValueError('xs and f_xs must have same size. '
                             f'Got {x0s.shape[0]} and {f_x0s.shape[0]}.')
    _check_initial_guesses_uniqueness(f_x0s)

    res = sidi_kernel(f_wrapper, x0s, f_x0s, etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Steffensen
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType steffensen_kernel(
        DoubleScalarFPtr f,
        double x0,
        double f_x0,
        bint aitken=True,
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
        return RootReturnType(x0, f_x0, step, f.n_f_calls, precision, error, converged, optimal)

    cdef double x1, x2, x3, denom
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        x1 = x0 + f_x0
        x2 = x1 + f.eval(x1)
        denom = x2 - 2 * x1 + x0
        if denom == 0:
            converged = False
            break
        # Use Aitken's delta-squared method to find a better approximation
        if aitken:
            x3 = x0 - (x1 - x0) ** 2 / denom
        else:
            x3 = x2 - (x2 - x1) ** 2 / denom
        precision = math.fabs(x3 - x0)
        x0, f_x0 = x3, f.eval(x3)
        error = math.fabs(f_x0)

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(x0, f_x0, step, f.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def steffensen(f: Callable[[float], float],
               x0: float,
               f_x0: Optional[float] = None,
               aitken: bool = True,
               etol: float = named_default(ETOL=ETOL),
               ertol: float = named_default(ERTOL=ERTOL),
               ptol: float = named_default(PTOL=PTOL),
               prtol: float = named_default(PRTOL=PRTOL),
               max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Steffensen's method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        x0 (float): First initial point.
        f_x0 (float, optional): Value evaluated at first initial point.
        aitken (bool, optional): Use Aitken's delta-squared process or not.
         Defaults to True.
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
    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)

    res = steffensen_kernel(f_wrapper, x0, f_x0, aitken,
                            etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Inverse Quadratic Interpolation
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType inverse_quadratic_interp_kernel(
        DoubleScalarFPtr f,
        double x0,
        double x1,
        double x2,
        double f_x0,
        double f_x1,
        double f_x2,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    cdef double[3] x_arr = [x0, x1, x2], f_arr = [f_x0, f_x1, f_x2]
    cdef double[:] xs = x_arr, f_xs = f_arr
    if _check_stop_cond_scalar_initial_guesses(xs, f_xs, etol, ertol, ptol, prtol,
                                               &r, &f_r, &precision, &error, &converged, &optimal):
        return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

    cdef double x3, df_01, df_02, df_12
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
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
        xs[2], f_xs[2] = x3, f.eval(x3)

        precision = vops.width(xs)
        error = math.fabs(f_xs[2])

    r, f_r = xs[2], f_xs[2]
    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.quasi_newton')
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
        ertol: float = named_default(ERTOL=ERTOL),
        ptol: float = named_default(PTOL=PTOL),
        prtol: float = named_default(PRTOL=PRTOL),
        max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Inverse Quadratic Interpolation method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        x0 (float): First initial point.
        x1 (float): Second initial point.
        x2 (float): Third initial point.
        f_x0 (float, optional): Value evaluated at first initial point.
        f_x1 (float, optional): Value evaluated at second initial point.
        f_x2 (float, optional): Value evaluated at third initial point.
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
    _check_initial_guesses_uniqueness((x0, x1, x2))

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if f_x1 is None:
        f_x1 = f_wrapper.eval(x1)
    if f_x2 is None:
        f_x2 = f_wrapper.eval(x2)
    _check_initial_vals_uniqueness((f_x0, f_x1, f_x2))

    res = inverse_quadratic_interp_kernel(f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2,
                                          etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Hyperbolic Interpolation
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType hyperbolic_interp_kernel(
        DoubleScalarFPtr f,
        double x0,
        double x1,
        double x2,
        double f_x0,
        double f_x1,
        double f_x2,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double r, f_r, precision, error
    cdef bint converged, optimal
    cdef double[3] x_arr = [x0, x1, x2], f_arr = [f_x0, f_x1, f_x2]
    cdef double[:] xs = x_arr, f_xs = f_arr
    if _check_stop_cond_scalar_initial_guesses(xs, f_xs, etol, ertol, ptol, prtol,
                                               &r, &f_r, &precision, &error, &converged, &optimal):
        return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

    cdef double x3, d_01, d_12, df_01, df_02, df_12
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
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
        xs[2], f_xs[2] = x3, f.eval(x3)

        precision = vops.width(xs)
        error = math.fabs(f_xs[2])

    r, f_r = xs[2], f_xs[2]
    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.quasi_newton')
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
        ertol: float = named_default(ERTOL=ERTOL),
        ptol: float = named_default(PTOL=PTOL),
        prtol: float = named_default(PRTOL=PRTOL),
        max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Hyperbolic Interpolation method for scalar root-finding.

    Args:
        f (function): Function for which the root is sought.
        x0 (float): First initial point.
        x1 (float): Second initial point.
        x2 (float): Third initial point.
        f_x0 (float, optional): Value evaluated at first initial point.
        f_x1 (float, optional): Value evaluated at second initial point.
        f_x2 (float, optional): Value evaluated at third initial point.
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
    _check_initial_guesses_uniqueness((x0, x1, x2))

    f_wrapper = PyDoubleScalarFPtr.from_f(f)
    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if f_x1 is None:
        f_x1 = f_wrapper.eval(x1)
    if f_x2 is None:
        f_x2 = f_wrapper.eval(x2)
    _check_initial_vals_uniqueness((f_x0, f_x1, f_x2))

    res = hyperbolic_interp_kernel(f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2,
                                   etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Muller
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType muller_kernel(
        ComplexScalarFPtr f,
        double complex x0,
        double complex x1,
        double complex x2,
        double complex f_x0,
        double complex f_x1,
        double complex f_x2,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double complex r, f_r
    cdef double precision, error
    cdef bint converged, optimal
    cdef double complex[3] x_arr = [x0, x1, x2], f_arr = [f_x0, f_x1, f_x2]
    cdef double complex[:] xs = x_arr, f_xs = f_arr
    if _check_stop_cond_complex_scalar_initial_guesses(xs, f_xs, etol, ertol, ptol, prtol,
                                                       &r, &f_r, &precision, &error, &converged, &optimal):
        return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

    cdef double complex div_diff_01, div_diff_12, div_diff_02, a, b, s_delta, d1, d2, d, x3
    cdef double complex d_01, d_02, d_12
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
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
        s_delta = sops.csqrt(b ** 2 - 4 * a * f_xs[2])  # \sqrt{b^2 - 4ac}
        d1, d2 = b + s_delta, b - s_delta
        # take the higher-magnitude denominator
        d = d1 if sops.cabs(d1) > sops.cabs(d2) else d2

        x3 = xs[2] - 2 * f_xs[2] / d
        xs[0], f_xs[0] = xs[1], f_xs[1]
        xs[1], f_xs[1] = xs[2], f_xs[2]
        xs[2], f_xs[2] = x3, f.eval(x3)

        precision = vops.cwidth(xs)
        error = sops.cabs(f_xs[2])

    r, f_r = xs[2], f_xs[2]
    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(r, f_r, step, f.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.scalar.quasi_newton')
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
           ertol: float = named_default(ERTOL=ERTOL),
           ptol: float = named_default(PTOL=PTOL),
           prtol: float = named_default(PRTOL=PRTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Muller's method for scalar root-finding.

    References:
        https://www.ams.org/journals/mcom/1956-10-056/S0025-5718-1956-0083822-0/

    Args:
        f (function): Function for which the root is sought.
        x0 (complex): First initial point.
        x1 (complex): Second initial point.
        x2 (complex): Third initial point.
        f_x0 (complex, optional): Value evaluated at first
         initial point.
        f_x1 (complex, optional): Value evaluated at second
         initial point.
        f_x2 (complex, optional): Value evaluated at third
         initial point.
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
    _check_initial_guesses_uniqueness((x0, x1, x2))

    f_wrapper = PyComplexScalarFPtr.from_f(f)
    if f_x0 is None:
        f_x0 = f_wrapper.eval(x0)
    if f_x1 is None:
        f_x1 = f_wrapper.eval(x1)
    if f_x2 is None:
        f_x2 = f_wrapper.eval(x2)
    _check_initial_vals_uniqueness((f_x0, f_x1, f_x2))

    res = muller_kernel(f_wrapper, x0, x1, x2, f_x0, f_x1, f_x2,
                        etol, ertol, ptol, prtol, max_iter)
    return res
