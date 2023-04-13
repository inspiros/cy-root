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
    _check_stop_condition_args,
)
from ._check_args cimport _check_stop_condition_bracket_vector
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from ._return_types import BracketingMethodsReturnType
from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .ops.scalar_ops cimport isclose
from .ops.vector_ops cimport fabs, max
from .typing import *
from .utils.function_tagging import tag

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
@cython.returns((np.ndarray, np.ndarray, cython.unsignedlong, np.ndarray, np.ndarray, double, double, bint, bint))
cdef vrahatis_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t, ndim=2] x0s,
        np.ndarray[np.float64_t, ndim=2] F_x0s,
        np.ndarray[np.float64_t, ndim=2] S_x0s,
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
    if _check_stop_condition_bracket_vector(x0s, F_x0s, etol, ertol, ptol, prtol,
                                            r, F_r, &precision, &error, &converged, &optimal):
        return r, F_r, step, x0s, F_x0s, precision, error, converged, optimal

    cdef unsigned int d = x0s.shape[1]
    cdef unsigned long[:, :] vertices = _get_1_simplexes(S_x0s)
    cdef np.ndarray[np.float64_t, ndim=1] S_r = np.empty(F_x0s.shape[1], dtype=np.float64)
    cdef unsigned long i, p0, p1
    cdef long r_id = -1
    converged = True
    while not (isclose(0, error, ertol, etol) or isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1
        for i in range(vertices.shape[0]):
            if vertices[i, 2] == 1:
                p0 = vertices[i, 0]
                p1 = vertices[i, 1]
                r = (x0s[p0] + x0s[p1]) / 2
                F_r[:] = F(r)
                S_r[:] = np.sign(F_r)
                if np.array_equal(S_r, S_x0s[p0]):
                    x0s[p0] = r
                    F_x0s[p0] = F_r
                    r_id = p0
                else:
                    x0s[p1] = r
                    F_x0s[p1] = F_r
                    r_id = p1
                error = max(fabs(F_r))
                precision = max(x0s.max(0) - x0s.min(0))
                if not isclose(0, error, ertol, etol):
                    r_id = -1
                else:
                    break
        if r_id >= 0:
            break

    if r_id >= 0:
        # return the vertex with small enough error
        optimal = isclose(0, error, ertol, etol)
        return x0s[r_id], F_x0s[r_id], step, x0s, F_x0s, precision, error, converged, optimal
    elif step == 0:
        # if the precision tol is satisfied without running into the loop,
        # just return the vertex with the smallest error
        optimal = isclose(0, error, ertol, etol)
        return r, F_r, step, x0s, F_x0s, precision, error, converged, optimal
    # otherwise, find the diagonal with the longest length
    cdef unsigned long best_i
    cdef double longest_len_squared = -math.INFINITY, diag_len_squared
    for i in range(vertices.shape[0]):
        if vertices[i, 2] == d - 1:
            diag_len_squared = np.power(x0s[vertices[i, 0]] - x0s[vertices[i, 1]], 2).sum()
            if diag_len_squared > longest_len_squared:
                best_i = i
                longest_len_squared = diag_len_squared
    r = (x0s[vertices[best_i, 0]] + x0s[vertices[best_i, 1]]) / 2
    F_r = F(r)

    error = max(fabs(F_r))
    optimal = isclose(0, error, ertol, etol)
    return r, F_r, step, x0s, F_x0s, precision, error, converged, optimal

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
    for j in range(M.shape[1] - 1, -1, -1):
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
        eps: float=1e-3) -> Tuple[np.ndarray, np.ndarray]:
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
    x = x.reshape(-1)
    cdef unsigned int d = x.shape[0]
    if h is None:
        h = np.full(d, np.abs(x).mean(), dtype=np.float64)
    elif isinstance(h, np.ndarray):
        h = h.astype(np.float64).reshape(-1)
        if h.shape[0] != d:
            raise ValueError('x and h must be of the same dimension.')
    else:
        h = np.full(d, h, dtype=np.float64)

    cdef:
        np.ndarray[np.float64_t, ndim=2] E = np.tile(x, [2 ** x.shape[0], 1])
        np.ndarray[np.float64_t, ndim=2] M = get_M(d)

        np.ndarray[np.float64_t, ndim=2] V = E + M @ np.diag(h)

        np.ndarray[np.float64_t, ndim=2] F_V = np.stack([F(V[i]) for i in range(V.shape[0])])
        np.ndarray[np.float64_t, ndim=2] S = np.sign(F_V)
    if np.unique(S, axis=0).shape[0] == M.shape[0]:
        return V, S

    cdef unsigned long[:, :] vertices = _get_1_simplexes(M)
    for i in range(vertices.shape[0]):
        if vertices[i, 2] != 1:
            continue
        pair = vertices[i, :2]
        simplex = V[pair]
        signs = S[pair]
        coef = simplex[1] - simplex[0]
        intercept = simplex[0]
        for j in range(d):
            res = bisect(lambda r: F(r * coef + intercept)[j], 0, 1,
                         algo=2, etol=eps, ertol=0, ptol=0, prtol=0, max_iter=0)
            if res.optimal:  # found an intersection between simplex and F_j
                new_simplex = np.empty_like(simplex)
                new_simplex[0] = res.root * (1 - eps * 10) * coef + intercept
                new_simplex[1] = res.root * (1 + eps * 10) * coef + intercept
                new_signs = np.empty_like(signs)
                new_signs[:] = np.sign(F(new_simplex[0]))
                new_signs[1, j] = -new_signs[0, j]
                if np.array_equal(signs, new_signs):
                    continue
                V[pair] = new_simplex
                S[pair] = new_signs
            if np.unique(S, axis=0).shape[0] == M.shape[0]:
                return V, S
    raise ValueError('Unable to find an admissible n-polygon. '
                     'Try to modify initial point x or search direction h.')

# noinspection DuplicatedCode
def sorted_by_vertices(*mats, S):
    sorted_inds = np.lexsort(
        tuple(S[:, j] for j in range(S.shape[1] - 1, -1, -1)))
    return *(m[sorted_inds] for m in mats), S[sorted_inds]

# noinspection DuplicatedCode
@tag('cyroot.vector.bracketing')
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
        https://link.springer.com/article/10.1007/BF01389620

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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)

    x0s = np.asarray(x0s, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if x0s.ndim == 1 or x0s.shape[0] == 1:
        x0s, S_x0s = compute_admissible_n_polygon(F_wrapper, x0s.reshape(-1), h)
    elif x0s.shape[0] != 2 ** x0s.shape[1]:
        raise ValueError('Initial bounds do not form an admissible n-polygon.')

    if F_x0s is None:
        F_x0s = np.stack([F_wrapper(x0s[i]) for i in range(x0s.shape[0])])
    else:
        F_x0s = np.asarray(F_x0s, dtype=np.float64)

    S_x0s = np.sign(F_x0s)
    if np.unique(S_x0s, axis=0).shape[0] != x0s.shape[0]:
        raise ValueError('Initial bounds do not form an admissible n-polygon.')

    # sort by order of M
    x0s, F_x0s, S_x0s = sorted_by_vertices(x0s, F_x0s, S=S_x0s)
    res = vrahatis_kernel(
        F_wrapper, x0s, F_x0s, S_x0s, etol, ertol, ptol, prtol, max_iter)
    return BracketingMethodsReturnType.from_results(res, F_wrapper.n_f_calls)

#------------------------
# Eiger-Sikorski-Stenger
#------------------------
# TODO: Add Eiger-Sikorski-Stenger' Bisection method using simplex
#  as presented in https://dl.acm.org/doi/10.1145/2701.2705.
