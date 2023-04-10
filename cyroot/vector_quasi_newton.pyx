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
    _check_unique_initial_guesses,
)
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from ._return_types import QuasiNewtonMethodReturnType
from .fptr cimport (
    ndarray_func_type, NdArrayFPtr, PyNdArrayFPtr,
)
from .ops.scalar_ops cimport fisclose, sqrt
from .ops.vector_ops cimport fabs, fmax, norm
from .utils.function_tagging import tag

__all__ = [
    'wolfe_bittner',
    'robinson',
    'barnes',
    'traub_steffensen',
    'broyden',
    'klement',
]

################################################################################
# Generalized Secant
################################################################################
#------------------------
# Wolfe-Bittner
#------------------------
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef wolfe_bittner_kernel(
        ndarray_func_type F,
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

    # sort by error of F
    cdef np.ndarray[np.int64_t, ndim=1] inds = np.abs(F_x0s).max(1).argsort()[::-1]
    cdef np.ndarray[np.float64_t, ndim=2] xs = x0s[inds]
    cdef np.ndarray[np.float64_t, ndim=2] F_xs = F_x0s[inds]

    cdef unsigned long d = xs.shape[1], i
    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, ps
    cdef np.ndarray[np.float64_t, ndim=2] A = np.empty((d + 1, d + 1), dtype=np.float64)
    A[0, :] = 1
    A[1:] = F_xs.transpose()
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(d + 1, dtype=np.float64)
    b[0] = 1
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        ps = np.linalg.lstsq(A, b, rcond=None)[0]
        x1 = ps.dot(xs)
        F_x1 = F(x1)

        xs[:-1] = xs[1:]
        F_xs[:-1] = F_xs[1:]
        xs[-1] = x1
        F_xs[-1] = F_x1
        A[1:, :-1] = A[1:, 1:]
        A[1:, -1] = F_x1

        precision = (F_xs.min(0) - F_xs.max(0)).max()
        error = fmax(fabs(F_x1))

    optimal = fisclose(0, error, ertol, etol)
    return xs[-1], F_xs[-1], step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def wolfe_bittner(F: Callable[[np.ndarray], np.ndarray],
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

    x0s = np.asarray(x0s, dtype=np.float64)
    d = x0s.shape[1]
    if x0s.shape[0] != d + 1:
        raise ValueError('Number of initial points must be d + 1. '
                         f'Got n_points={x0s.shape[0]}, d={d}.')

    F_wrapper = PyNdArrayFPtr(F)
    if F_x0s is None:
        F_x0s = np.stack([F_wrapper(x0s[i]) for i in range(d + 1)])
    else:
        F_x0s = np.asarray(F_x0s, dtype=np.float64)

    res = wolfe_bittner_kernel[NdArrayFPtr](
        F_wrapper, x0s, F_x0s, etol, ertol, ptol, prtol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)

#------------------------
# Robinson
#------------------------
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef robinson_kernel(
        ndarray_func_type F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] x1,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=1] F_x1,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef np.ndarray[np.float64_t, ndim=1] r = np.empty(x0.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] F_r = np.empty(F_x0.shape[0], dtype=np.float64)
    cdef bint converged, optimal
    if _check_stop_condition_initial_guesses_vector(np.stack([x0, x1]), np.stack([F_x0, F_x1]),
                                                    etol, ertol, ptol, prtol,
                                                    r, F_r, &precision, &error, &converged, &optimal):
        return r, F_r, step, precision, error, converged, optimal

    cdef unsigned long i
    cdef np.ndarray[np.float64_t, ndim=2] I = np.eye(x0.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] J = np.empty((x0.shape[0], x0.shape[0]), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] H
    cdef np.ndarray[np.float64_t, ndim=1] d_x = x1 - x0
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        H = norm(d_x) * I  # simple
        # H = norm(d_x) * _compute_H(d_x)  # efficient
        # H = norm(d_x) * scipy.stats.ortho_group.rvs(dim=x0.shape[0])  # random
        for i in range(x0.shape[0]):
            J[i] = F(x1 + H[i]) - F_x1
        J = J.dot(np.linalg.solve(H, I))
        d_x = np.linalg.solve(J, -F_x1)  # -J^-1.F_x_n
        r = x1 + d_x
        F_r = F(r)

        x0, x1 = x1, r
        F_x0, F_x1 = F_x1, F_r
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_r))

    optimal = fisclose(0, error, ertol, etol)
    return r, F_r, step, precision, error, converged, optimal

# noinspection DuplicatedCode
cdef np.ndarray[np.float64_t, ndim=2] _compute_H(np.ndarray[np.float64_t, ndim=1] d_x):
    cdef np.ndarray[np.float64_t, ndim=2] H = np.empty((d_x.shape[0], d_x.shape[0]), dtype=np.float64)
    cdef unsigned long i, j, k
    cdef double[:] d_x_squared = d_x * d_x
    cdef double scale
    with nogil:
        for j in range(d_x.shape[0]):
            scale = 0.
            for i in range(d_x.shape[0]):
                if j == d_x.shape[0] - 1:
                    H[i, j] = -d_x[i]
                elif i == j + 1:
                    H[i, j] = 0
                    for k in range(j + 1):
                        H[i, j] -= d_x_squared[k]
                elif i <= j:
                    H[i, j] = d_x[i] * d_x[j + 1]
                else:
                    H[i, j] = 0
                scale += H[i, j] * H[i, j]
            scale = sqrt(scale)
            if scale != 0:
                for i in range(d_x.shape[0]):  # row normalize
                    H[i, j] /= scale
    return H

# noinspection DuplicatedCode
@tag('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def robinson(F: Callable[[np.ndarray], np.ndarray],
             x0: np.ndarray,
             x1: np.ndarray,
             F_x0: Optional[np.ndarray] = None,
             F_x1: Optional[np.ndarray] = None,
             etol: float = named_default(ETOL=ETOL),
             ertol: float = named_default(ERTOL=ERTOL),
             ptol: float = named_default(PTOL=PTOL),
             prtol: float = named_default(PRTOL=PRTOL),
             max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Robinson's Secant method for vector root-finding.

    References:
        https://epubs.siam.org/doi/abs/10.1137/0703057

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        x1 (np.ndarray): Second initial point.
        F_x0 (np.ndarray, optional): Value evaluated at first
         initial point.
        F_x1 (np.ndarray, optional): Value evaluated at second
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
    # TODO: the method for computing the efficient H following the paper
    #  is bugged and does not converge.
    # check params
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)
    _check_unique_initial_guesses(x0, x1)

    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr(F)
    if F_x0 is None:
        F_x0 = F(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if F_x1 is None:
        F_x1 = F(x1)
    else:
        F_x1 = np.asarray(F_x1, dtype=np.float64)

    res = robinson_kernel[NdArrayFPtr](
        F_wrapper, x0, x1, F_x0, F_x1, etol, ertol, ptol, prtol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)

#------------------------
# Barnes
#------------------------
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef barnes_kernel(
        ndarray_func_type F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        int formula=2,
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

    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, z
    cdef np.ndarray[np.float64_t, ndim=2] D
    cdef np.ndarray[np.float64_t, ndim=2] d_xs = np.zeros((x0.shape[0] - 1, x0.shape[0]), dtype=np.float64)
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        d_x = np.linalg.solve(J_x0, -F_x0)
        if step > 1:
            z = _orthogonal_vector(d_xs[:step - 1])
            if step > x0.shape[0] - 1:
                d_xs[:-1] = d_xs[1:]
                d_xs[-1] = d_x
            else:
                d_xs[step - 1] = d_x
        else:
            z = d_x

        x1 = x0 + d_x
        F_x1 = F(x1)

        if formula == 2:
            D = np.outer(F_x1 - F_x0 - J_x0.dot(d_x), z) / z.dot(d_x) # (F_x1 - F_x0 - J.d_x).z^T / z^T.d_x
        else:
            D = np.outer(F_x1, z) / z.dot(d_x)  # F.z^T / z^T.d_x
        J_x0 = J_x0 + D

        x0 = x1
        F_x0 = F_x1
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x1))

    optimal = fisclose(0, error, ertol, etol)
    return x1, F_x1, step, precision, error, converged, optimal

# noinspection DuplicatedCode
cdef np.ndarray[np.float64_t, ndim=1] _orthogonal_vector(
        np.ndarray[np.float64_t, ndim=2] v,
        int method = 1,
        double eps = 1e-15):
    cdef unsigned int n = v.shape[0]
    cdef unsigned int d = v.shape[1]
    cdef np.ndarray[np.float64_t, ndim=2] basis = np.empty((d, d), dtype=np.float64)

    cdef unsigned int i, k = 0
    cdef np.ndarray[np.float64_t, ndim=1] b, s
    cdef np.ndarray[np.float64_t, ndim=2] q, r
    if method == 0:  # svd
        u, s, _ = np.linalg.svd(v.T, full_matrices=False)
        k = np.sum(s > eps)
        basis[:k] = u.T[:k]
    elif method == 1:  # qr
        q, r = np.linalg.qr(v.T)
        k = np.sum(np.abs(r).max(1) > eps)
        basis[:k] = q.T[:k]
    elif method == 2:  # gs
        for i in range(v.shape[0]):
            b = v[i] - np.sum(v[i].dot(basis[:k].T).reshape(-1, 1) * basis[:k], 0)
            if not np.allclose(b, 0, atol=eps, rtol=0):
                basis[k] = b / norm(b)
                k += 1
    else:
        raise ValueError(f'Unsupported orthogonalize method {method}.')

    # use gram schmidt process to find the next basis
    cdef np.ndarray[np.float64_t, ndim=1] v_rand
    cdef unsigned int k_final = k
    while k == k_final:
        v_rand = np.random.randn(d)
        b = v_rand - np.sum(v_rand.dot(basis[:k].T).reshape(-1, 1) * basis[:k], 0)
        if not np.allclose(b, 0, atol=eps, rtol=0):
            basis[k] = b / norm(b)
            k += 1
            break
    return basis[k_final]

# noinspection DuplicatedCode
@tag('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def barnes(F: Callable[[np.ndarray], np.ndarray],
           x0: np.ndarray,
           F_x0: Optional[np.ndarray] = None,
           J_x0: np.ndarray = None,
           formula: int = 2,
           etol: float = named_default(ETOL=ETOL),
           ertol: float = named_default(ERTOL=ERTOL),
           ptol: float = named_default(PTOL=PTOL),
           prtol: float = named_default(PRTOL=PRTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Barnes's Secant method for vector root-finding.
    Setting ``formula=1`` will use Eq.(9) while setting ``formula=2``
    will use Eq.(44) (recommended) in the paper to update the Jacobian.

    References:
        https://academic.oup.com/comjnl/article/8/1/66/489886

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at
         initial point.
        J_x0 (np.ndarray, optional): Jacobian guess at initial
         point.
        formula (int, optional): Formula to update the Jacobian,
         which can be either the base ``1`` or its extended form
         ``2``. Defaults to {formula}
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

    x0 = np.asarray(x0, dtype=np.float64)

    if J_x0 is None:  # TODO: estimate initial guess with finite difference
        raise ValueError('J_x0 must be explicitly set.')
    J_x0 = np.asarray(J_x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr(F)
    if F_x0 is None:
        F_x0 = F(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)

    res = barnes_kernel[NdArrayFPtr](
        F_wrapper, x0, F_x0, J_x0, formula, etol, ertol, ptol, prtol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)

################################################################################
# Traub-Steffensen
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef traub_steffensen_kernel(
        ndarray_func_type F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
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

    cdef unsigned long i
    cdef np.ndarray[np.float64_t, ndim=2] J = np.empty((x0.shape[0], x0.shape[0]), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] H
    cdef np.ndarray[np.float64_t, ndim=1] d_x
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        H = np.diag(F_x0)
        for i in range(x0.shape[0]):
            J[i] = F(x0 + H[i]) - F_x0
        J = np.matmul(J, np.linalg.inv(H))
        d_x = np.linalg.solve(J, -F_x0)  # -J^-1.F_x_n
        x0 = x0 + d_x
        F_x0 = F(x0)

        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x0))

    optimal = fisclose(0, error, ertol, etol)
    return x0, F_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def traub_steffensen(F: Callable[[np.ndarray], np.ndarray],
                     x0: np.ndarray,
                     F_x0: Optional[np.ndarray] = None,
                     etol: float = named_default(ETOL=ETOL),
                     ertol: float = named_default(ERTOL=ERTOL),
                     ptol: float = named_default(PTOL=PTOL),
                     prtol: float = named_default(PRTOL=PRTOL),
                     max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> QuasiNewtonMethodReturnType:
    """
    Traub-Steffensen method for vector root-finding.

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        x1 (np.ndarray): Second initial point.
        F_x0 (np.ndarray, optional): Value evaluated at first
         initial point.
        F_x1 (np.ndarray, optional): Value evaluated at second
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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr(F)
    if F_x0 is None:
        F_x0 = F(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)

    res = traub_steffensen_kernel[NdArrayFPtr](
        F_wrapper, x0, F_x0, etol, ertol, ptol, prtol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)

################################################################################
# Broyden
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef broyden1_kernel(
        ndarray_func_type F,
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
        ndarray_func_type F,
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
    Broyden's methods for vector root-finding.
    This will use Broyden's Good method if ``algo='good'`` or Broyden's Bad
    method if ``algo='bad'``.

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
         point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
        algo (int, str): Broyden's ``1``/``'good'`` or ``2``/``'bad'``
         version. Defaults to {algo}.
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

    x0 = np.asarray(x0, dtype=np.float64)
    if J_x0 is None:
        raise ValueError('J_x0 must be explicitly set.')
    J_x0 = np.asarray(J_x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr(F)
    if F_x0 is None:
        F_x0 = F_wrapper(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)

    if algo in [1, 'good']:
        res = broyden1_kernel[NdArrayFPtr](
            F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    elif algo in [2, 'bad']:
        res = broyden2_kernel[NdArrayFPtr](
            F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    else:
        raise ValueError(f'algo must be either 1/\'good\' or 2/\'bad\'. '
                         f'Got unknown algo {algo}.')
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)

################################################################################
# Klement
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef klement_kernel(
        ndarray_func_type F,
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

    x0 = np.asarray(x0, dtype=np.float64)
    if J_x0 is None:
        raise ValueError('J_x0 must be explicitly set.')
    J_x0 = np.asarray(J_x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr(F)
    if F_x0 is None:
        F_x0 = F_wrapper(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)

    res = klement_kernel[NdArrayFPtr](
        F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    return QuasiNewtonMethodReturnType.from_results(res, F_wrapper.n_f_calls)
