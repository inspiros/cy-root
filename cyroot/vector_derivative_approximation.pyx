# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

import cython
from cython cimport view
import numpy as np
cimport numpy as np

from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .scalar_derivative_approximation import _finite_diference_args_check
from .typing import ArrayLike

__all__ = [
    'VectorDerivativeApproximation',
    'generalized_finite_difference',
    'GeneralizedFiniteDifference',
]

################################################################################
# Base Class
################################################################################
# noinspection DuplicatedCode
cdef class VectorDerivativeApproximation(NdArrayFPtr):
    def __init__(self, F: Union[NdArrayFPtr, Callable[[ArrayLike], ArrayLike]]):
        if isinstance(F, NdArrayFPtr):
            self.F = F
        else:
            self.F = PyNdArrayFPtr(F)

    cpdef np.ndarray eval(self, np.ndarray x):
        raise NotImplementedError

    cpdef np.ndarray eval_with_f_val(self, np.ndarray x, np.ndarray F_x):
        raise NotImplementedError

################################################################################
# Finite Difference
################################################################################
# noinspection DuplicatedCode
cdef np.ndarray generalized_finite_difference_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=1] F_x,
        double h=1.,
        int order=1,
        int kind=1):
    cdef unsigned int d = x.shape[0], i
    cdef double x_, diff = 0.
    cdef int bin_coef, sgn = (-1) ** order if kind == 1 else 1
    cdef unsigned int[:] dims = view.array(shape=(order + 1,),
                                           itemsize=sizeof(int),
                                           format='I')
    dims[:] = d
    cdef np.ndarray[np.float64_t, ndim=1] x_h = x.copy(), f_i
    cdef np.ndarray[np.float64_t, ndim=2] D = np.empty(dims, dtype=np.float64)
    for i in range(d):
        x_ = x_h[i]
        x_h[i] += h
        D[:, i] = (F.eval(x_h) - F_x) / h
        x_h[i] = x_
    # for i in range(order + 1):
    #     bin_coef = binomial_coef(order, i)
    #     if kind == 1:  # forward
    #         f_i = F(x + i * h) if i > 0 else F_x
    #         diff += sgn * bin_coef * f_i
    #     elif kind == -1:  # backward
    #         f_i = F(x - i * h) if i > 0 else F_x
    #         diff += sgn * bin_coef * f_i
    #     else:  # central
    #         f_i = F(x + (<double> order / 2 - i) * h) if 2 * i != order else F_x
    #         diff += sgn * bin_coef * F(x + (order / 2 - i) * h)
    #     sgn = -sgn
    return D

# noinspection DuplicatedCode
cdef class GeneralizedFiniteDifference(VectorDerivativeApproximation):
    def __init__(self,
                 F: Union[NdArrayFPtr, Callable[[ArrayLike], ArrayLike]],
                 h: float = 1.,
                 order: int = 1,
                 kind: int = 1):
        super().__init__(F)
        # check args
        _finite_diference_args_check(h, order, kind)
        self.h = h
        self.order = order
        self.kind = kind

    cpdef np.ndarray eval(self, np.ndarray x):
        self.n_f_calls += 1
        cdef np.ndarray[np.float64_t, ndim=1] F_x = self.F.eval(x)
        return generalized_finite_difference_kernel(
            <NdArrayFPtr> self.F, x, F_x, self.h, self.order, self.kind)

    cpdef np.ndarray eval_with_f_val(self, np.ndarray x, np.ndarray F_x):
        self.n_f_calls += 1
        return generalized_finite_difference_kernel(
            <NdArrayFPtr> self.F, x, F_x, self.h, self.order, self.kind)

# noinspection DuplicatedCode
@cython.binding(True)
def generalized_finite_difference(F: Callable[[ArrayLike], ArrayLike],
                                  x: ArrayLike,
                                  F_x: Optional[ArrayLike] = None,
                                  h: float = 1.,
                                  order: int = 1,
                                  kind: int = 1):
    """
    Generalized finite difference method.

    Args:
        F (function): Function for which the derivative is sought.
        x (np.ndarray): Point at which the derivative is sought.
        F_x (np.ndarray, optional): Value evaluated at point ``x``.
        h (float):  Finite difference step. Defaults to 1.
        order (int): Order of derivative to be estimated.
         Defaults to 1.
        kind (int): Type of finite difference, including ``1``
         for forward, ``-1`` for backward, and ``0`` for central.
         Defaults to 1.

    Returns:
        diff: Estimated derivative.
    """
    # check args
    _finite_diference_args_check(h, order, kind)

    x = np.asarray(x, dtype=np.float64)

    if isinstance(F, NdArrayFPtr):
        F_wrapper = F
    else:
        F_wrapper = PyNdArrayFPtr(F)

    if F_x is None:
        F_x = F_wrapper(x)
    else:
        F_x = np.asarray(F_x, dtype=np.float64)
    return generalized_finite_difference_kernel(<NdArrayFPtr> F_wrapper, x, F_x, h, order, kind)
