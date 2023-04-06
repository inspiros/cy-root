# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

import cython
from dynamic_default_args import dynamic_default_args, named_default
import numpy as np
cimport numpy as np

from ._check_args cimport (
    _check_stop_condition_initial_guess_vector,
    _check_stop_condition_initial_guesses_vector,
)
from ._check_args import (
    _check_stop_condition_args,
)
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from ._return_types import QuasiNewtonMethodReturnType
from .fptr cimport (
    double_np_ndarray_func_type, DoubleNpNdArrayFPtr, PyDoubleNpNdArrayFPtr,
)
from .utils.scalar_ops cimport fisclose
from .utils.vector_ops cimport fabs, fmax
from .utils.function_tagging import tag

__all__ = [
    'generalized_secant',
    'broyden',
    'klement',
]

################################################################################
# Klement
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef wolfe_bittner_secant_kernel(
        double_np_ndarray_func_type F,
        np.ndarray[np.float64_t, ndim=2] x0s,
        np.ndarray[np.float64_t, ndim=2] F_x0s,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef np.ndarray[np.float64_t, ndim=1] r = np.empty(x0s.shape[1], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] F_r = np.empty(F_x0s.shape[1], dtype=np.float64)
    cdef bint converged, optimal
    if _check_stop_condition_initial_guesses_vector(x0s, F_x0s, etol, ertol, ptol, prtol,
                                                    r, F_r, &precision, &error, &converged, &optimal):
        return r, F_r, step, precision, error, converged, optimal

    cdef unsigned long d = x0s.shape[1], i
    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, ps
    cdef np.ndarray[np.float64_t, ndim=2] A = np.empty((d + 1, d + 1), dtype=np.float64)
    A[0, :] = 1
    A[1:] = F_x0s.transpose()
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(d + 1, dtype=np.float64)
    b[0] = 1
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        ps = np.linalg.lstsq(A, b, rcond=None)[0]
        x1 = ps.dot(x0s)
        F_x1 = F(x1)

        x0s[:-1] = x0s[1:]
        F_x0s[:-1] = F_x0s[1:]
        x0s[-1] = x1
        F_x0s[-1] = F_x1
        A[1:, :-1] = A[1:, 1:]
        A[1:, -1] = F_x1

        precision = (F_x0s.min(0) - F_x0s.max(0)).max()
        error = fmax(fabs(F_x1))

    optimal = fisclose(0, error, ertol, etol)
    return x0s[-1], F_x0s[-1], step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_secant(F: Callable[[np.ndarray], np.ndarray],
                       x0s: np.ndarray,
                       F_x0s: Optional[np.ndarray] = None,
                       etol: float = named_default(ETOL=ETOL),
                       ertol: float = named_default(ERTOL=ERTOL),
                       ptol: float = named_default(PTOL=PTOL),
                       prtol: float = named_default(PRTOL=PRTOL),
                       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Wolfe-Bittner's Secant method for vector root-finding.

    Args:
        F (function): Function for which the root is sought.
        x0s (np.ndarray): First initial points. There requires
         to be `d + 1` points, where `d` is the dimension.
        F_x0s (np.ndarray, optional): Values evaluated at initial
         points.
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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)

    x0s = np.asarray(x0s).astype(np.float64)
    d = x0s.shape[1]
    if x0s.shape[0] != d + 1:
        raise ValueError('Number of initial points must be d + 1. '
                         f'Got n_points={x0s.shape[0]}, d={d}.')

    F_wrapper = PyDoubleNpNdArrayFPtr(F)
    if F_x0s is None:
        F_x0s = np.stack([F_wrapper(x0s[i]) for i in range(d + 1)])

    res = wolfe_bittner_secant_kernel[DoubleNpNdArrayFPtr](
        F_wrapper, x0s, F_x0s, etol, ertol, ptol, prtol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)

################################################################################
# Broyden
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef broyden1_kernel(
        double_np_ndarray_func_type F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess_vector(x0, F_x0, etol, ertol, ptol, prtol,
                                                  &precision, &error, &converged, &optimal):
        return x0, F_x0, step, precision, error, converged, optimal

    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, d_F
    cdef double denom
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        d_x = np.linalg.solve(J_x0, -F_x0)

        x1 = x0 + d_x
        F_x1 = F(x1)
        d_F = F_x1 - F_x0

        denom = np.dot(d_x, d_x)  # ||x1 - x0||^2
        if denom == 0:
            converged = False
            break
        J_x0 += np.outer(d_F - np.dot(J_x0, d_x), d_x) / denom

        x0, F_x0 = x1, F_x1
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x0))

    optimal = fisclose(0, error, ertol, etol)
    return x0, F_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef broyden2_kernel(
        double_np_ndarray_func_type F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess_vector(x0, F_x0, etol, ertol, ptol, prtol,
                                                  &precision, &error, &converged, &optimal):
        return x0, F_x0, step, precision, error, converged, optimal

    cdef np.ndarray[np.float64_t, ndim=2] J_x0_inv = np.linalg.inv(J_x0)
    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, d_F, u
    cdef double denom
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        d_x = -J_x0_inv.dot(F_x0)  # Newton step

        x1 = x0 + d_x
        F_x1 = F(x1)
        d_F = F_x1 - F_x0

        u = J_x0_inv.dot(d_F)  # J^-1.[f_x1 - f_x0]

        denom = np.dot(d_x, u)  # [x1 - x0].J^-1.[f_x1 - f_x0]
        if denom == 0:
            converged = False
            break
        # Sherman-Morrison formula
        J_x0_inv += ((d_x - u).dot(d_x) * J_x0_inv) / denom

        x0, F_x0 = x1, F_x1
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x0))

    optimal = fisclose(0, error, ertol, etol)
    return x0, F_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def broyden(F: Callable[[np.ndarray], np.ndarray],
            x0: np.ndarray,
            F_x0: Optional[np.ndarray] = None,
            J_x0: Optional[np.ndarray] = None,
            algo: Union[int, str] = 2,
            etol: float = named_default(ETOL=ETOL),
            ertol: float = named_default(ERTOL=ERTOL),
            ptol: float = named_default(PTOL=PTOL),
            prtol: float = named_default(PRTOL=PRTOL),
            max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Broyden's method for vector root-finding.

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
         point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
        algo (int, str): Broyden's 1/'good' or 2/'bad' method.
         Defaults to {algo}.
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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)

    x0 = np.asarray(x0).astype(np.float64)
    if J_x0 is None:
        raise ValueError('J_x0 must be explicitly set.')
    J_x0 = np.asarray(J_x0).astype(np.float64)

    F_wrapper = PyDoubleNpNdArrayFPtr(F)
    if F_x0 is None:
        F_x0 = F_wrapper(x0)

    if str(algo) in ['1', 'good']:
        res = broyden1_kernel[DoubleNpNdArrayFPtr](
            F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    elif str(algo) in ['2', 'bad']:
        res = broyden2_kernel[DoubleNpNdArrayFPtr](
            F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    else:
        raise ValueError(f'algo should be 1/\'good\' or 2/\'bad\'. '
                         f'Got unknown algo {algo}.')
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)

################################################################################
# Klement
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef klement_kernel(
        double_np_ndarray_func_type F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess_vector(x0, F_x0, etol, ertol, ptol, prtol,
                                                  &precision, &error, &converged, &optimal):
        return x0, F_x0, step, precision, error, converged, optimal

    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, d_F, denom, k
    cdef np.ndarray[np.float64_t, ndim=2] U
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        d_x = np.linalg.solve(J_x0, -F_x0)

        x1 = x0 + d_x
        F_x1 = F(x1)
        d_F = F_x1 - F_x0

        U = J_x0 * d_x
        denom = np.power(U, 2).sum(1)  # Σ_j(J_{i,j} * d_x_{j})^2
        if np.any(denom == 0):
            converged = False
            break
        k = (d_F - U.sum(1)) / denom
        J_x0 += k.reshape(-1, 1) * U * J_x0  # (1 + k_i * J_{i,j} * d_x_{j}) * J_{i, j}

        x0, F_x0 = x1, F_x1
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x0))

    optimal = fisclose(0, error, ertol, etol)
    return x0, F_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def klement(F: Callable[[np.ndarray], np.ndarray],
            x0: np.ndarray,
            F_x0: Optional[np.ndarray] = None,
            J_x0: Optional[np.ndarray] = None,
            etol: float = named_default(ETOL=ETOL),
            ertol: float = named_default(ERTOL=ERTOL),
            ptol: float = named_default(PTOL=PTOL),
            prtol: float = named_default(PRTOL=PRTOL),
            max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Klement's method (a modified Broyden's Good method) for vector root-finding presented
    in the paper "On Using Quasi-Newton Algorithms of the Broyden Class for Model-to-Test Correlation".

    References:
        https://jatm.com.br/jatm/article/view/373

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
         point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)

    x0 = np.asarray(x0).astype(np.float64)
    if J_x0 is None:
        raise ValueError('J_x0 must be explicitly set.')
    J_x0 = np.asarray(J_x0).astype(np.float64)

    F_wrapper = PyDoubleNpNdArrayFPtr(F)
    if F_x0 is None:
        F_x0 = F_wrapper(x0)

    res = klement_kernel[DoubleNpNdArrayFPtr](
        F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)
