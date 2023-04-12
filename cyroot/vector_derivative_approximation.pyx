# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

from cpython cimport array
import array
import numpy as np
cimport numpy as np
import cython
from cython cimport view
from libcpp.algorithm cimport sort
from libcpp.vector cimport vector

from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .ops.scalar_ops cimport binomial_coef
from .ops.vector_ops cimport prod
from .scalar_derivative_approximation import _finite_diference_args_check
from .typing import *
from .utils.itertools cimport product

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

    @staticmethod
    cdef object from_f(object F):
        if F is not None:
            return F
        cdef GeneralizedFiniteDifference wrapper = GeneralizedFiniteDifference.__new__(GeneralizedFiniteDifference)
        wrapper.f = F
        return wrapper

    cpdef np.ndarray eval(self, np.ndarray x):
        raise NotImplementedError

    cpdef np.ndarray eval_with_f_val(self, np.ndarray x, np.ndarray F_x):
        raise NotImplementedError

################################################################################
# Finite Difference
################################################################################
# noinspection DuplicatedCode
cdef unsigned int[:, :] _vector_derivative_indices(unsigned int[:] ns, bint[:] unique_mask):
    cdef unsigned int[:, :] inds = product(ns)
    unique_mask[:] = True
    cdef unsigned int i, j, k
    with nogil:
        for i in range(inds.shape[0]):
            for j in range(inds.shape[1] - 1):
                for k in range(j + 1, inds.shape[1]):
                    if inds[i, j] > inds[i, k]:
                        unique_mask[i] = False
                        break
                if not unique_mask[i]:
                    break
    return inds

cdef np.ndarray[np.float64_t, ndim=2] _vector_perturbations(unsigned int dim, int order):
    cdef unsigned int[:] ns = view.array(shape=(dim,),
                                         itemsize=sizeof(int),
                                         format='I')
    ns[:] = order + 1
    cdef unsigned int[:, :] inds = product(ns)
    cdef array.array perturb_inds = array.array('I')
    for i in range(inds.shape[0]):
        steps = inds[i]
        if sum(steps) <= order:
            perturb_inds.append(i)
    return np.ascontiguousarray(np.asarray(inds, dtype=np.float64)[perturb_inds])

cdef int[:] _finite_difference_coefs(int order, int kind):
    cdef int[:] out = view.array(shape=(order + 1,),
                                 itemsize=sizeof(int),
                                 format='i')
    cdef long i
    if kind == 1:
        for i in range(order + 1):
            out[i] = (-1) ** (order - i) * binomial_coef(order, i)
    else:
        for i in range(order + 1):
            out[i] = (-1) ** i * binomial_coef(order, i)
    return out

cdef unsigned int[:] _index_to_grad_comb(unsigned int[:] index, unsigned int dim):
    cdef unsigned int[:] comb = view.array(shape=(dim,),
                                           itemsize=sizeof(int),
                                           format='I')
    comb[:] = 0
    cdef unsigned int i
    for i in range(index.shape[0]):
        comb[index[i]] += 1
    return comb

# noinspection DuplicatedCode
cdef np.ndarray generalized_finite_difference_kernel(
        NdArrayFPtr F,
        np.ndarray[np.float64_t, ndim=1] x,
        np.ndarray[np.float64_t, ndim=1] F_x,
        np.ndarray[np.float64_t, ndim=1] h,
        int order=1,
        int kind=1):
    cdef unsigned int i, j, k, ii, eq_ii
    cdef unsigned int[:] dims = view.array(shape=(order + 1,),
                                           itemsize=sizeof(int),
                                           format='I')
    dims[0] = F_x.shape[0]
    dims[1:] = x.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] D = np.zeros(dims, dtype=np.float64).reshape(F_x.shape[0], -1)

    # TODO: efficient implementations for Jacobian and Hessian
    # cdef np.ndarray[np.float64_t, ndim=1] x_perturb = x.copy()
    # for j in range(x.shape[0]):
    #     x_perturb[j] += h[j]
    #     D[:, j] = (F.eval(x_perturb) - F_x) / h[j]
    #     x_perturb[j] = x[j]
    # return D.reshape(dims)

    cdef unsigned int[:, :] indices
    cdef unsigned int[:] index
    cdef bint[:] unique_mask = view.array(shape=(prod[np.uint32_t](dims[1:]),),
                                          itemsize=sizeof(int),
                                          format='i')
    indices = _vector_derivative_indices(dims[1:], unique_mask)

    cdef bint zero_step
    cdef np.ndarray[np.float64_t, ndim=2] perturbation_steps = _vector_perturbations(x.shape[0], order)
    cdef np.ndarray[np.float64_t, ndim=1] perturbation_step
    cdef np.ndarray[np.float64_t, ndim=2] F_perturbations = np.empty(
        (perturbation_steps.shape[0], F_x.shape[0]), dtype=np.float64)
    for i in range(perturbation_steps.shape[0]):
        if kind == 1:
            perturbation_step = perturbation_steps[i]
        elif kind == -1:
            perturbation_step = -perturbation_steps[i]
        else:
            perturbation_step = <double> order / 2. - perturbation_steps[i]

        if kind != 0:
            if i == 0:
                F_perturbations[0] = F_x
            else:
                F_perturbations[i] = F(x + perturbation_step * h)
        else:
            zero_step = True
            for j in range(perturbation_step.shape[0]):
                if perturbation_step[j] != 0:
                    zero_step = False
                    break
            if zero_step:
                F_perturbations[i] = F_x
            else:
                F_perturbations[i] = F(x + perturbation_step * h)

    cdef double scale
    cdef int coef
    cdef int[:] coefs
    cdef vector[vector[int]] all_coefs = vector[vector[int]](x.shape[0])
    for j in range(all_coefs.size()):
        all_coefs[j].reserve(order + 1)
    cdef unsigned int[:] grad_comb, len_coefs = view.array(shape=(x.shape[0],),
                                                           itemsize=sizeof(int),
                                                           format='I')
    cdef unsigned int[:, :] perturbs
    cdef unsigned int[:] perturb
    for ii in range(indices.shape[0]):
        if unique_mask[ii]:
            index = indices[ii]
            grad_comb = _index_to_grad_comb(index, x.shape[0])
            scale = 1
            for j in range(x.shape[0]):
                scale *= h[j] ** grad_comb[j]
                all_coefs[j].clear()
                coefs = _finite_difference_coefs(grad_comb[j], kind)
                for k in range(coefs.shape[0]):
                    all_coefs[j].push_back(coefs[k])
                len_coefs[j] = all_coefs[j].size()
            perturbs = product(len_coefs)
            for j in range(perturbs.shape[0]):
                perturb = perturbs[j]
                coef = 1
                for k in range(all_coefs.size()):
                    coef *= all_coefs[k][perturb[k]]
                for k in range(perturbation_steps.shape[0]):
                    if np.array_equal(perturbation_steps[k], perturb):
                        break
                D[:, ii] += coef * F_perturbations[k] / scale

    cdef unsigned int[:] eq_index = view.array(shape=(indices.shape[1],),
                                               itemsize=sizeof(int),
                                               format='I')
    for ii in range(indices.shape[0]):
        if not unique_mask[ii]:
            eq_index[:] = indices[ii]
            sort(&eq_index[0], (&eq_index[0]) + eq_index.shape[0])
            eq_ii = 0
            for i in range(order - 1, -1, -1):
                eq_ii += eq_index[i] * x.shape[0] ** (order - 1 - i)
            D[:, ii] = D[:, eq_ii]
    return D.reshape(dims)

# noinspection DuplicatedCode
cdef class GeneralizedFiniteDifference(VectorDerivativeApproximation):
    def __init__(self,
                 F: Union[NdArrayFPtr, Callable[[ArrayLike], ArrayLike]],
                 h: Union[float, VectorLike] = 1.,
                 order: int = 1,
                 kind: int = 1):
        super().__init__(F)
        # check args
        _finite_diference_args_check(h, order, kind)
        if isinstance(h, float):
            self.h = np.full(1, h)
        else:
            self.h = np.asarray(h, dtype=np.float64)
        self.order = order
        self.kind = kind

    cpdef np.ndarray eval(self, np.ndarray x):
        self.n_f_calls += 1
        cdef np.ndarray[np.float64_t, ndim=1] F_x = self.F.eval(x)
        cdef np.ndarray[np.float64_t, ndim=1] h
        if self.h.shape[0] == x.shape[0]:
            h = self.h
        elif self.h.shape[0] == 1:
            h = np.full(x.shape[0], self.h[0])
        else:
            raise ValueError(f'x.shape[0]={x.shape[0]} while h.shape[0]={self.h.shape[0]}.')
        return generalized_finite_difference_kernel(
            <NdArrayFPtr> self.F, x, F_x, h, self.order, self.kind)

    cpdef np.ndarray eval_with_f_val(self, np.ndarray x, np.ndarray F_x):
        self.n_f_calls += 1
        cdef np.ndarray[np.float64_t, ndim=1] h
        if self.h.shape[0] == x.shape[0]:
            h = self.h
        elif self.h.shape[0] == 1:
            h = np.full(x.shape[0], self.h[0])
        else:
            raise ValueError(f'x.shape[0]={x.shape[0]} while h.shape[0]={self.h.shape[0]}.')
        return generalized_finite_difference_kernel(
            <NdArrayFPtr> self.F, x, F_x, h, self.order, self.kind)

# noinspection DuplicatedCode
@cython.binding(True)
def generalized_finite_difference(F: Callable[[ArrayLike], ArrayLike],
                                  x: ArrayLike,
                                  F_x: Optional[ArrayLike] = None,
                                  h: Union[float, VectorLike] = 1.,
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
    if isinstance(h, float):
        h = np.full(x.shape[0], h)
    else:
        h = np.asarray(h, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if F_x is None:
        F_x = F_wrapper(x)
    else:
        F_x = np.asarray(F_x, dtype=np.float64)
    return generalized_finite_difference_kernel(<NdArrayFPtr> F_wrapper, x, F_x, h, order, kind)
