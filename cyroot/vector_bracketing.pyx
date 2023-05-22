# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union, Tuple

import cython
import numpy as np
cimport numpy as np
from cython cimport view
from dynamic_default_args import dynamic_default_args, named_default
from libc cimport math

from .scalar_bracketing import bisect
from ._check_args import (
    _check_stop_cond_args,
)
from ._check_args cimport _check_stop_cond_vector_bracket
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from ._types import VectorLike, Array2DLike
from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .ops cimport scalar_ops as sops, vector_ops as vops
from .return_types cimport BracketingMethodsReturnType
from .utils._function_registering import register
from .utils._warnings import warn_value

__all__ = [
    'vrahatis',
]

################################################################################
# Bisection
################################################################################
#------------------------
# Vrahatis
#------------------------
# noinspection DuplicatedCode
cdef BracketingMethodsReturnType vrahatis_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t, ndim=2] x0s,
        np.ndarray[np.float64_t, ndim=2] F_x0s,
        np.ndarray[np.float64_t, ndim=2] S_x0s,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned int d_in = <unsigned int> x0s.shape[1], d_out = <unsigned int> F_x0s.shape[1]
    cdef unsigned long step = 0
    cdef double precision, error
    cdef np.ndarray[np.float64_t, ndim=1] r = np.empty(d_in, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] F_r = np.empty(d_out, dtype=np.float64)
    cdef bint converged, optimal
    if _check_stop_cond_vector_bracket(x0s, F_x0s, etol, ertol, ptol, prtol,
                                       r, F_r, &precision, &error, &converged, &optimal):
        return BracketingMethodsReturnType(
            r, F_r, step, F.n_f_calls, x0s, F_x0s, precision, error, converged, optimal)

    cdef unsigned long[:, :] vertices = _get_1_simplexes(S_x0s)
    cdef np.ndarray[np.float64_t, ndim=1] S_r = np.empty(d_out, dtype=np.float64)
    cdef unsigned long i, p0, p1
    cdef long r_id = -1
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        for i in range(vertices.shape[0]):
            if vertices[i, 2] == 1:
                p0 = vertices[i, 0]
                p1 = vertices[i, 1]
                r = (x0s[p0] + x0s[p1]) / 2
                F_r[:] = F.eval(r)
                S_r[:] = np.sign(F_r)
                if np.array_equal(S_r, S_x0s[p0]):
                    x0s[p0] = r
                    F_x0s[p0] = F_r
                    r_id = p0
                else:
                    x0s[p1] = r
                    F_x0s[p1] = F_r
                    r_id = p1
                error = vops.max(vops.fabs(F_r))
                precision = vops.max(x0s.max(0) - x0s.min(0))
                if not sops.isclose(0, error, ertol, etol):
                    r_id = -1
                else:
                    break
        if r_id >= 0:
            break

    if r_id >= 0:
        # return the vertex with small enough error
        optimal = sops.isclose(0, error, ertol, etol)
        return BracketingMethodsReturnType(
            x0s[r_id], F_x0s[r_id], step, F.n_f_calls, x0s, F_x0s, precision, error, converged, optimal)
    elif step == 0:
        # if the precision tol is satisfied without running into the loop,
        # just return the vertex with the smallest error
        optimal = sops.isclose(0, error, ertol, etol)
        return BracketingMethodsReturnType(
            r, F_r, step, F.n_f_calls, x0s, F_x0s, precision, error, converged, optimal)
    # otherwise, find the diagonal with the longest length
    cdef unsigned long best_i
    cdef double longest_len_squared = -math.INFINITY, diag_len_squared
    for i in range(vertices.shape[0]):
        if vertices[i, 2] == d_in - 1:
            diag_len_squared = np.power(x0s[vertices[i, 0]] - x0s[vertices[i, 1]], 2).sum()
            if diag_len_squared > longest_len_squared:
                best_i = i
                longest_len_squared = diag_len_squared
    r = (x0s[vertices[best_i, 0]] + x0s[vertices[best_i, 1]]) / 2
    F_r = F.eval(r)

    error = vops.max(vops.fabs(F_r))
    optimal = sops.isclose(0, error, ertol, etol)
    return BracketingMethodsReturnType(
        r, F_r, step, F.n_f_calls, x0s, F_x0s, precision, error, converged, optimal)

# noinspection DuplicatedCode
cpdef unsigned long[:, :] _get_1_simplexes(double[:, :] S):
    cdef unsigned long[:, :] vertices = view.array(
        shape=(S.shape[0] * (S.shape[0] - 1) // 2, 3), itemsize=sizeof(long), format='L')
    vertices[:, 2] = 0
    cdef unsigned int v_id = 0, i, j, k
    with nogil:
        for i in range(S.shape[0]):
            for j in range(i + 1, S.shape[0]):
                vertices[v_id, 0] = i
                vertices[v_id, 1] = j
                for k in range(S.shape[1]):
                    vertices[v_id, 2] += S[i, k] == S[j, k]
                v_id += 1
    return vertices

# noinspection DuplicatedCode
cpdef np.ndarray[np.float64_t, ndim=2] get_M(unsigned int n,
                                             bint sign=False):
    cdef np.ndarray[np.float64_t, ndim=2] M = np.empty((2 ** n, n), dtype=np.float64)
    cdef unsigned int i, j, rate = 1
    for j in range(<unsigned int> M.shape[1] - 1, -1, -1):
        for i in range(M.shape[0]):
            M[i, j] = (i // rate) % 2
        rate *= 2
    if sign:
        M[M == 0] = -1
    return M

# noinspection DuplicatedCode
def compute_admissible_n_polygon(
        F: Callable[[VectorLike], VectorLike],
        x: VectorLike,
        h: Optional[Union[VectorLike, float]] = None,
        eps: float=1e-5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find an admissible n-polygon from an initial point.

    Args:
        F (function): Function for which the root is sought.
        x (np.ndarray): Initial point.
        h (np.ndarray, optional): Search direction.
        eps: (float): ``etol`` for internal bisection.

    Returns:
        V (np.ndarray): Vertices of the n-polygon.
        S (np.ndarray): Signs of ``V``.
    """
    if eps <= 0:
        raise ValueError(f'eps must be positive. Got {eps}.')
    if isinstance(F, NdArrayFPtr):
        F = F.eval

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    F_x = F(x)
    d_in, d_out = x.shape[0], F_x.shape[0]
    n = 2 ** d_out

    if h is None:
        h = np.full(d_in, np.abs(x).mean(), dtype=np.float64)
    elif isinstance(h, np.ndarray):
        h = h.astype(np.float64).reshape(-1)
        if h.shape[0] != d_in:
            raise ValueError('h must be of the same dimension as x. '
                             f'Got d_in={d_in}, h.shape={h.shape[0]}.')
    else:
        h = np.full(d_in, h, dtype=np.float64)

    E = np.tile(x, [2 ** d_in, 1])
    cdef M = get_M(d_in)
    V = E + M.dot(np.diag(h))
    F_V = np.empty((V.shape[0], d_out), dtype=np.float64)
    F_V[0] = F_x
    for i in range(1, F_V.shape[0]):
        F_V[i] = F(V[i])
    S = np.sign(F_V)

    if np.unique(S, axis=0).shape[0] == n:
        if d_in == d_out:
            return V, S
        else:
            V_out = np.empty((n, d_in), dtype=np.float64)
            S_out = np.empty((n, d_out), dtype=np.float64)
            V_out[0] = V[0]
            S_out[0] = S[0]
            k = 1
            for i in range(V.shape[0]):
                if not np.equal(S[i], S_out[:k]).all(1).any():
                    V_out[k] = V[i]
                    S_out[k] = S[i]
                    k += 1
                if k == n:
                    return V_out, S_out

    vertices = _get_1_simplexes(M)
    mask1 = np.ones(V.shape[0], dtype=np.bool_)
    mask2 = np.ones(V.shape[0], dtype=np.bool_)
    for i in range(vertices.shape[0]):
        if vertices[i, 2] != 1:
            continue
        pair = vertices[i, :2]
        simplex = V[pair]
        signs = S[pair]

        mask1[:] = True
        mask2[:] = True
        mask1[pair[0]] = False
        mask2[pair[1]] = False
        if not (np.equal(signs[0], S[mask1]).all(1).any() and
                np.equal(signs[1], S[mask2]).all(1).any()):
            continue

        coef = simplex[1] - simplex[0]
        intercept = simplex[0]
        length = np.linalg.norm(coef)
        if length == 0:
            continue

        for j in range(d_out):
            res = bisect(lambda r: F(r * coef + intercept)[j], 0, 1,
                         algo=2, etol=eps, ertol=0, ptol=0, prtol=0, max_iter=0)
            if res.optimal:  # found an intersection between simplex and F_j
                step = 2 * eps / length
                new_simplex = np.empty_like(simplex)
                new_simplex[0] = max(0, res.root - step) * coef + intercept
                new_simplex[1] = min(1, res.root + step) * coef + intercept
                new_signs = np.empty_like(signs)
                new_signs[0] = np.sign(F(new_simplex[0]))
                new_signs[1] = np.sign(F(new_simplex[1]))
                # new_signs[1, j] = -new_signs[0, j]

                if np.array_equal(signs, new_signs):
                    continue
                elif (np.equal(new_signs[0], S[mask1]).all(1).any() and
                        np.equal(new_signs[1], S[mask2]).all(1).any()):
                    continue

                V[pair] = new_simplex
                S[pair] = new_signs
            if np.unique(S, axis=0).shape[0] == n:
                if d_in == d_out:
                    return V, S
                else:
                    V_out = np.empty((n, d_in), dtype=np.float64)
                    S_out = np.empty((n, d_out), dtype=np.float64)
                    V_out[0] = V[0]
                    S_out[0] = S[0]
                    k = 1
                    for i in range(V.shape[0]):
                        if not np.equal(S[i], S_out[:k]).all(1).any():
                            V_out[k] = V[i]
                            S_out[k] = S[i]
                            k += 1
                        if k == n:
                            return V_out, S_out
    raise ValueError('Unable to find an admissible n-polygon. '
                     'Try to modify initial point x or search direction h. '
                     'Best unique signs:\n' + repr(np.unique(S)))

# noinspection DuplicatedCode
def sorted_by_vertices(*mats, S):
    sorted_inds = np.lexsort(
        tuple(S[:, j] for j in range(S.shape[1] - 1, -1, -1)))
    return *(m[sorted_inds] for m in mats), S[sorted_inds]

# noinspection DuplicatedCode
@register('cyroot.vector.bracketing')
@dynamic_default_args()
@cython.binding(True)
def vrahatis(F: Callable[[VectorLike], VectorLike],
             x0s: Union[Array2DLike, VectorLike],
             F_x0s: Optional[Array2DLike] = None,
             h: Optional[Union[VectorLike, float]] = None,
             etol: float = named_default(ETOL=ETOL),
             ertol: float = named_default(ERTOL=ERTOL),
             ptol: float = named_default(PTOL=PTOL),
             prtol: float = named_default(PRTOL=PRTOL),
             max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> BracketingMethodsReturnType:
    """
    Vrahatis's Generalized Bisection method for vector root-finding.
    This method requires the initial brackets to form an admissible
    n-polygon, which is an n-sided polygon with the sign of the function
    ``F`` on each of its corner is unique.

    The parameter ``h`` can be used to set a search direction to construct
    the admissible n-polygon from a single input guess following the
    algorithm proposed in the original paper. However, it is usually not
    very helpful.

    Another shortcoming is that the initial search region should be
    well divided into n parts by sign of the function ``F``. This means
    the bracket should be close enough for complex functions.
    Otherwise, the convergence is not guaranteed.

    References:
        https://doi.org/10.1007/BF01389620

    Args:
        F (function): Function for which the root is sought.
        x0s (np.ndarray): Initial bounds or a single initial point.
        F_x0s (np.ndarray, optional): Values evaluated at initial
         bounds.
        h (np.ndarray, optional): Search direction used to construct
         the admissible n-polygon from a single initial point.
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

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if x0s.ndim == 1 or x0s.shape[0] == 1:
        x0s, S_x0s = compute_admissible_n_polygon(F_wrapper, x0s.reshape(-1), h)
    elif x0s.ndim != 2:
        raise ValueError('Initial bounds must be 2D array. '
                         f'Got {x0s.shape}.')

    if F_x0s is None:
        F_x0s = np.stack([F_wrapper.eval(x0s[i]) for i in range(x0s.shape[0])])
    else:
        F_x0s = np.asarray(F_x0s, dtype=np.float64)
        if F_x0s.ndim != 2 or F_x0s.shape[0] != x0s.shape[0]:
            raise ValueError('x0s and F_x0s must have same length.')
    if x0s.shape[0] != 2 ** F_x0s.shape[1]:
        raise ValueError('Initial bounds do not form an admissible n-polygon. '
                         f'Expected {2 ** F_x0s.shape[1]} points, got {x0s.shape[0]}.')

    if x0s.shape[1] < F_x0s.shape[1]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got n={x0s.shape[1]}, m={F_x0s.shape[1]}.')

    S_x0s = np.sign(F_x0s)
    if np.unique(S_x0s, axis=0).shape[0] != x0s.shape[0]:
        raise ValueError('Initial bounds do not form an admissible n-polygon. '
                         f'Got signs:\n' + repr(S_x0s))

    # sort by order of M
    x0s, F_x0s, S_x0s = sorted_by_vertices(x0s, F_x0s, S=S_x0s)
    res = vrahatis_kernel(
        F_wrapper, x0s, F_x0s, S_x0s, etol, ertol, ptol, prtol, max_iter)
    return res

#------------------------
# Eiger-Sikorski-Stenger
#------------------------
# TODO: Add Eiger-Sikorski-Stenger' Bisection method using simplex
#  as presented in https://dl.acm.org/doi/10.1145/2701.2705.
