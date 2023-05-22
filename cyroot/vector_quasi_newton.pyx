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
    _check_stop_cond_vector_initial_guess,
    _check_stop_cond_vector_initial_guesses,
)
from ._check_args import (
    _check_stop_cond_args,
    _check_initial_guesses_uniqueness,
    _check_initial_vals_uniqueness,
)
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER, FINITE_DIFF_STEP
from ._types import VectorLike, Array2DLike
from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .ops cimport scalar_ops as sops, vector_ops as vops, matrix_ops as mops
from .return_types cimport RootReturnType
from .utils._function_registering import register
from .utils._warnings import warn_value
from .vector_derivative_approximation import generalized_finite_difference

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
cdef RootReturnType wolfe_bittner_kernel(
        NdArrayFPtr F,
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
    if _check_stop_cond_vector_initial_guesses(x0s, F_x0s, etol, ertol, ptol, prtol,
                                               r, F_r, &precision, &error, &converged, &optimal):
        return RootReturnType(r, F_r, step, F.n_f_calls, precision, error, converged, optimal)

    # sort by error of F
    cdef np.ndarray[np.int64_t, ndim=1] inds = np.abs(F_x0s).max(1).argsort()[::-1]
    cdef np.ndarray[np.float64_t, ndim=2] xs = x0s[inds]
    cdef np.ndarray[np.float64_t, ndim=2] F_xs = F_x0s[inds]

    cdef unsigned long i
    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, ps
    cdef np.ndarray[np.float64_t, ndim=2] A = np.empty((F_xs.shape[1] + 1, xs.shape[0]), dtype=np.float64)
    A[0, :] = 1
    A[1:] = F_xs.transpose()
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(F_xs.shape[1] + 1, dtype=np.float64)
    b[0] = 1
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        ps = mops.inv(A, b, method=4)  # lstsq
        x1 = ps.dot(xs)
        F_x1 = F.eval(x1)

        xs[:-1] = xs[1:]
        F_xs[:-1] = F_xs[1:]
        xs[-1] = x1
        F_xs[-1] = F_x1
        A[1:, :-1] = A[1:, 1:]
        A[1:, -1] = F_x1

        precision = (F_xs.min(0) - F_xs.max(0)).max()
        error = vops.max(vops.fabs(F_x1))

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(xs[-1], F_xs[-1], step, F.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def wolfe_bittner(F: Callable[[VectorLike], VectorLike],
                  x0s: Array2DLike,
                  F_x0s: Optional[Array2DLike] = None,
                  etol: float = named_default(ETOL=ETOL),
                  ertol: float = named_default(ERTOL=ERTOL),
                  ptol: float = named_default(PTOL=PTOL),
                  prtol: float = named_default(PRTOL=PRTOL),
                  max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Wolfe-Bittner's Secant method for vector root-finding.

    References:
        https://doi.org/10.1145/368518.368542

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
    etol, ertol, ptol, prtol, max_iter = _check_stop_cond_args(etol, ertol, ptol, prtol, max_iter)

    x0s = np.asarray(x0s, dtype=np.float64)
    if x0s.ndim != 2:
        raise ValueError(f'x0s must be a 2D array. Got x0s.shape={x0s.shape}.')

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if F_x0s is None:
        F_x0s = np.stack([F_wrapper.eval(x0s[i]) for i in range(x0s.shape[0])])
    else:
        F_x0s = np.asarray(F_x0s, dtype=np.float64)
        if F_x0s.ndim != 2 or F_x0s.shape[0] != x0s.shape[0]:
            raise ValueError('x0s and F_x0s must have same length.')

    warn_msgs = []
    if x0s.shape[1] < F_x0s.shape[1]:
        warn_msgs.append('Input dimension is smaller than output dimension. '
                         f'Got d_in={x0s.shape[0]}, d_out={F_x0s.shape[0]}.')
    if x0s.shape[0] < F_x0s.shape[1] + 1:
        warn_msgs.append('Number of initial points should be at least d_out + 1, '
                         'where d_out is dimension of output. '
                         f'Got len(x0s)={x0s.shape[0]}, d_out={F_x0s.shape[1]}.')
    if len(warn_msgs):
        warn_value('\n'.join(warn_msgs))

    res = wolfe_bittner_kernel(
        F_wrapper, x0s, F_x0s, etol, ertol, ptol, prtol, max_iter)
    return res

#------------------------
# Robinson
#------------------------
# noinspection DuplicatedCode
cdef RootReturnType robinson_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] x1,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=1] F_x1,
        int orth=2,
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
    if _check_stop_cond_vector_initial_guesses(np.stack([x0, x1]), np.stack([F_x0, F_x1]),
                                               etol, ertol, ptol, prtol,
                                               r, F_r, &precision, &error, &converged, &optimal):
        return RootReturnType(r, F_r, step, F.n_f_calls, precision, error, converged, optimal)

    cdef unsigned long i
    cdef np.ndarray[np.float64_t, ndim=2] I = np.eye(x0.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] J = np.empty((F_x0.shape[0], x0.shape[0]), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] H
    cdef np.ndarray[np.float64_t, ndim=1] d_x = x1 - x0
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        if orth == 1:  # simple
            H = vops.norm(d_x) * I
        elif orth == 2:  # efficient
            H = vops.norm(d_x) * _efficient_H(d_x)
        else:  # random
            H = vops.norm(d_x) * _random_H(x0.shape[0])
        for i in range(x0.shape[0]):
            J[:, i] = F.eval(x1 + H[i]) - F_x1
        J = J.dot(mops.inv(H))
        d_x = mops.inv(J, -F_x1)  # -J^-1.F_x_n
        r = x1 + d_x
        F_r = F.eval(r)

        x0, x1 = x1, r
        F_x0, F_x1 = F_x1, F_r
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_r))

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(r, F_r, step, F.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
cdef np.ndarray[np.float64_t, ndim=2] _efficient_H(np.ndarray[np.float64_t, ndim=1] d_x):
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
            scale = sops.sqrt(scale)
            if scale != 0:
                for i in range(d_x.shape[0]):  # row normalize
                    H[i, j] /= scale
    return H

cdef np.ndarray[np.float64_t, ndim=2] _random_H(unsigned int d):
    cdef np.ndarray[np.float64_t, ndim=2] q, r
    cdef np.ndarray[np.float64_t, ndim=1] s
    q, r = np.linalg.qr(np.random.normal(size=(d, d)))
    s = r.diagonal()
    q *= (s / np.abs(s)).reshape(1, d)
    return q

# noinspection DuplicatedCode
@register('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def robinson(F: Callable[[VectorLike], VectorLike],
             x0: VectorLike,
             x1: VectorLike,
             F_x0: Optional[VectorLike] = None,
             F_x1: Optional[VectorLike] = None,
             orth: Union[str, int] = 'efficient',
             etol: float = named_default(ETOL=ETOL),
             ertol: float = named_default(ERTOL=ERTOL),
             ptol: float = named_default(PTOL=PTOL),
             prtol: float = named_default(PRTOL=PRTOL),
             max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
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
        orth (int, optional): Strategy to compute the orthogonal
         matrix, can be either ``1``/``'identity'`` for identity,
         ``2``/``'efficient'`` for efficient method proposed in
         the original paper, and ``3``/``'random'`` for random.
         Defaults to {orth}.
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

    if orth == 'identity':
        orth = 1
    elif orth == 'efficient':
        orth = 2
    elif orth == 'random':
        orth = 3
    elif orth not in [1, 2, 3]:
        raise ValueError('orth must be either 1/\'identity\', 2/\'efficient\', or '
                         f'3/\'random\'. Got {orth}.')

    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    _check_initial_guesses_uniqueness((x0, x1))

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if F_x1 is None:
        F_x1 = F_wrapper.eval(x1)
    else:
        F_x1 = np.asarray(F_x1, dtype=np.float64)
    _check_initial_vals_uniqueness((F_x0, F_x1))

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got n={x0.shape[0]}, m={F_x0.shape[0]}.')

    res = robinson_kernel(
        F_wrapper, x0, x1, F_x0, F_x1, orth, etol, ertol, ptol, prtol, max_iter)
    return res

#------------------------
# Barnes
#------------------------
# noinspection DuplicatedCode
cdef RootReturnType barnes_kernel(
        NdArrayFPtr F,
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
    if _check_stop_cond_vector_initial_guess(x0, F_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

    cdef double denom
    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, z
    cdef np.ndarray[np.float64_t, ndim=2] D
    cdef np.ndarray[np.float64_t, ndim=2] d_xs = np.zeros((x0.shape[0] - 1, x0.shape[0]), dtype=np.float64)
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        d_x = mops.inv(J_x0, -F_x0)
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
        F_x1 = F.eval(x1)

        denom = z.dot(d_x)
        if denom == 0:
            converged = False
            break
        if formula == 2:
            D = np.outer(F_x1 - F_x0 - J_x0.dot(d_x), z) / denom # (F_x1 - F_x0 - J.d_x).z^T / z^T.d_x
        else:
            D = np.outer(F_x1, z) / denom  # F.z^T / z^T.d_x
        J_x0 = J_x0 + D

        x0 = x1
        F_x0 = F_x1
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x1))

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(x1, F_x1, step, F.n_f_calls, precision, error, converged, optimal)

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
                basis[k] = b / vops.norm(b)
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
            basis[k] = b / vops.norm(b)
            k += 1
            break
    return basis[k_final]

# noinspection DuplicatedCode
@register('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def barnes(F: Callable[[VectorLike], VectorLike],
           x0: VectorLike,
           F_x0: Optional[VectorLike] = None,
           J_x0: Optional[Array2DLike] = None,
           formula: int = 2,
           h: Optional[Union[float, VectorLike]] = named_default(
               FINITE_DIFF_STEP=FINITE_DIFF_STEP),
           etol: float = named_default(ETOL=ETOL),
           ertol: float = named_default(ERTOL=ERTOL),
           ptol: float = named_default(PTOL=PTOL),
           prtol: float = named_default(PRTOL=PRTOL),
           max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
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
         ``2``. Defaults to {formula}.
        h (float, np.ndarray, optional): Finite difference step size,
         ignored when ``J_x0`` is not None. Defaults to {h}.
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

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = generalized_finite_difference(F_wrapper, x0, F_x0, h=h, order=1)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got n={x0.shape[0]}, m={F_x0.shape[0]}.')

    res = barnes_kernel(
        F_wrapper, x0, F_x0, J_x0, formula, etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Traub-Steffensen
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType traub_steffensen_kernel(
        NdArrayFPtr F,
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
    if _check_stop_cond_vector_initial_guess(x0, F_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

    cdef unsigned long i
    cdef np.ndarray[np.float64_t, ndim=2] J = np.empty((F_x0.shape[0], x0.shape[0]), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] H = np.zeros((x0.shape[0], x0.shape[0]), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] d_x
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        for i in range(min(x0.shape[0], F_x0.shape[0])):  # H = diag(F_x0)
            H[i, i] = F_x0[i]

        for i in range(x0.shape[0]):
            J[:, i] = F.eval(x0 + H[i]) - F_x0
        J = np.matmul(J, mops.inv(H))
        d_x = mops.inv(J, -F_x0)  # -J^-1.F_x_n
        x0 = x0 + d_x
        F_x0 = F.eval(x0)

        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def traub_steffensen(F: Callable[[VectorLike], VectorLike],
                     x0: VectorLike,
                     F_x0: Optional[VectorLike] = None,
                     etol: float = named_default(ETOL=ETOL),
                     ertol: float = named_default(ERTOL=ERTOL),
                     ptol: float = named_default(PTOL=PTOL),
                     prtol: float = named_default(PRTOL=PRTOL),
                     max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Traub-Steffensen method for vector root-finding.

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at first
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

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got n={x0.shape[0]}, m={F_x0.shape[0]}.')

    res = traub_steffensen_kernel(
        F_wrapper, x0, F_x0, etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Broyden
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType broyden1_kernel(
        NdArrayFPtr F,
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
    if _check_stop_cond_vector_initial_guess(x0, F_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, d_F
    cdef double denom
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        d_x = mops.inv(J_x0, -F_x0)

        x1 = x0 + d_x
        F_x1 = F.eval(x1)
        d_F = F_x1 - F_x0

        denom = d_x.dot(d_x)  # ||x1 - x0||^2
        if denom == 0:
            converged = False
            break
        J_x0 += np.outer(d_F - J_x0.dot(d_x), d_x) / denom

        x0, F_x0 = x1, F_x1
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
cdef RootReturnType broyden2_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        bint inversed=False,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_cond_vector_initial_guess(x0, F_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

    cdef np.ndarray[np.float64_t, ndim=2] J_x0_inv = J_x0 if inversed else mops.inv(J_x0, None, method=3)
    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, d_F, u
    cdef double denom
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        d_x = -J_x0_inv.dot(F_x0)  # Newton step

        x1 = x0 + d_x
        F_x1 = F.eval(x1)
        d_F = F_x1 - F_x0

        u = J_x0_inv.dot(d_F)  # J^-1.[f_x1 - f_x0]

        denom = d_x.dot(u)  # [x1 - x0].J^-1.[f_x1 - f_x0]
        if denom == 0:
            converged = False
            break
        # Sherman-Morrison formula
        J_x0_inv += ((d_x - u).dot(d_x) * J_x0_inv) / denom

        x0, F_x0 = x1, F_x1
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def broyden(F: Callable[[VectorLike], VectorLike],
            x0: VectorLike,
            F_x0: Optional[VectorLike] = None,
            J_x0: Optional[Array2DLike] = None,
            algo: Union[int, str] = 2,
            h: Optional[Union[float, VectorLike]] = named_default(
                FINITE_DIFF_STEP=FINITE_DIFF_STEP),
            etol: float = named_default(ETOL=ETOL),
            ertol: float = named_default(ERTOL=ERTOL),
            ptol: float = named_default(PTOL=PTOL),
            prtol: float = named_default(PRTOL=PRTOL),
            max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
    """
    Broyden's methods for vector root-finding.
    This will use Broyden's Good method if ``algo='good'`` or Broyden's Bad
    method if ``algo='bad'``.

    References:
        https://doi.org/10.2307/2003941

    Args:
        F (function): Function for which the root is sought.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
         point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
        algo (int, str): Broyden's ``1``/``'good'`` or ``2``/``'bad'``
         version. Defaults to {algo}.
        h (float, np.ndarray, optional): Finite difference step size,
         ignored when ``J_x0`` is not None. Defaults to {h}.
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

    if algo == 'good':
        algo = 1
    elif algo == 'bad':
        algo = 2
    elif algo not in [1, 2]:
        raise ValueError('algo must be either 1/\'good\' or 2/\'bad\'. '
                         f'Got {algo}.')

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = generalized_finite_difference(F_wrapper, x0, F_x0, h=h, order=1)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got n={x0.shape[0]}, m={F_x0.shape[0]}.')

    if algo == 1:
        res = broyden1_kernel(
            F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    else:
        res = broyden2_kernel(
            F_wrapper, x0, F_x0, J_x0, False, etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Klement
################################################################################
# noinspection DuplicatedCode
cdef RootReturnType klement_kernel(
        NdArrayFPtr F,
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
    if _check_stop_cond_vector_initial_guess(x0, F_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

    cdef np.ndarray[np.float64_t, ndim=1] x1, F_x1, d_x, d_F, denom, k
    cdef np.ndarray[np.float64_t, ndim=2] U
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        d_x = mops.inv(J_x0, -F_x0)

        x1 = x0 + d_x
        F_x1 = F.eval(x1)
        d_F = F_x1 - F_x0

        U = J_x0 * d_x
        denom = np.power(U, 2).sum(1)  # Σ_j(J_{i,j} * d_x_{j})^2
        if np.any(denom == 0):
            converged = False
            break
        k = (d_F - U.sum(1)) / denom
        J_x0 += k.reshape(-1, 1) * U * J_x0  # (1 + k_i * J_{i,j} * d_x_{j}) * J_{i, j}

        x0, F_x0 = x1, F_x1
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return RootReturnType(x0, F_x0, step, F.n_f_calls, precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.vector.quasi_newton')
@dynamic_default_args()
@cython.binding(True)
def klement(F: Callable[[VectorLike], VectorLike],
            x0: VectorLike,
            F_x0: Optional[VectorLike] = None,
            J_x0: Optional[Array2DLike] = None,
            h: Optional[Union[float, VectorLike]] = named_default(
                FINITE_DIFF_STEP=FINITE_DIFF_STEP),
            etol: float = named_default(ETOL=ETOL),
            ertol: float = named_default(ERTOL=ERTOL),
            ptol: float = named_default(PTOL=PTOL),
            prtol: float = named_default(PRTOL=PRTOL),
            max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> RootReturnType:
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
        h (float, np.ndarray, optional): Finite difference step size,
         ignored when ``J_x0`` is not None. Defaults to {h}.
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

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = generalized_finite_difference(F_wrapper, x0, F_x0, h=h, order=1)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got n={x0.shape[0]}, m={F_x0.shape[0]}.')

    res = klement_kernel(
        F_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    return res
