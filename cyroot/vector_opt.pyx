# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

import numpy as np
cimport numpy as np
import cython
from libcpp.limits cimport numeric_limits

from ._check_args import _check_stop_cond_args
from libc.math cimport INFINITY, isfinite
from ._stopping cimport (
    StopState, RUNNING, RESIDUAL_TOLERANCE, MAX_ITERATIONS, BREAKDOWN, NONFINITE,
    init_stop, advance_x, check_stop, finish_stop, successful, norm_inf,
)
from ._types import VectorLike, Array2DLike, Array3DLike
from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .ops cimport scalar_ops as sops, vector_ops as vops, matrix_ops as mops
from .return_types cimport NewtonMethodsReturnType
from .utils._function_registering import register
from .utils._warnings import warn_value
from .vector_derivative_approximation import GeneralizedFiniteDifference, VectorDerivativeApproximation

cdef extern from 'defaults.hpp' namespace 'cyroot' nogil:
    double ETOL, ERTOL, PTOL, PRTOL, FINITE_DIFF_STEP
    size_t MAX_ITER

__all__ = [
    'generalized_mishchenko',
]

cdef class VectorOptFunctionWrapper:
    cdef NdArrayFPtr F, J, H
    # np.ndarray[np.float64_t, ndim=1] x0
    # np.ndarray[np.float64_t, ndim=1] F_x0
    # np.ndarray[np.float64_t, ndim=2] J_x0
    # np.ndarray[np.float64_t, ndim=3] H_x0

    def __init__(self,
                 F: Union[NdArrayFPtr, Callable[[VectorLike], VectorLike]],
                 J: Union[NdArrayFPtr, Callable[[VectorLike], VectorLike]],
                 H: Optional[Union[NdArrayFPtr, Callable[[VectorLike], VectorLike]]] = None):
        if isinstance(F, NdArrayFPtr):
            self.F = F
        else:
            self.F = PyNdArrayFPtr(F)
        if isinstance(J, NdArrayFPtr):
            self.J = J
        else:
            self.J = PyNdArrayFPtr(J)
        if H is not None:
            if isinstance(H, NdArrayFPtr):
                self.H = H
            else:
                self.H = PyNdArrayFPtr(H)

    cdef np.ndarray eval_G(self, np.ndarray x):
        return self.F.eval(x)

    cdef np.ndarray eval_J(self, np.ndarray x):
        cdef np.ndarray J_G = 2 * self.J.eval(x).transpose() @ self.F.eval(x)
        return np.expand_dims(J_G, 0)

    cdef np.ndarray eval_H(self, np.ndarray x):
        cdef np.ndarray H_G = self.J.eval(x).transpose() @ self.J.eval(x)  # J_f(x)^T @ J_f(x)
        cdef np.ndarray F_x, H_x
        cdef size_t i
        if self.H is not None:
            F_x = self.F.eval(x)
            H_x = self.H.eval(x)
            for i in range(F_x.shape[0]):
                H_G += F_x[i] * H_x[i]  # \sum_i f_i(x).H_f_i(x)
        H_G *= 2
        return np.expand_dims(H_G, 0)

################################################################################
# Generalized Mishchenko
################################################################################
cdef inline double empirical_hess_lip(
        np.ndarray[np.float64_t, ndim=1] delta_prev,
        np.ndarray[np.float64_t, ndim=1] J_curr,
        np.ndarray[np.float64_t, ndim=1] J_prev,
        np.ndarray[np.float64_t, ndim=2] H_prev):
    cdef double r2 = vops.norm(delta_prev) ** 2
    if r2 > 0:
        return 2 * vops.norm(J_curr - J_prev - H_prev @ delta_prev) / 2
    return numeric_limits[double].epsilon()

# noinspection DuplicatedCode
cdef NewtonMethodsReturnType generalized_mishchenko_kernel(
        VectorOptFunctionWrapper G,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        np.ndarray[np.float64_t, ndim=3] H_x0,
        np.ndarray[np.float64_t, ndim=1] d_x,
        double hess_lip=1e-5,
        bint adaptive=False,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        size_t max_iter=MAX_ITER):
    cdef size_t step = 0
    cdef double precision = INFINITY, error = norm_inf(F_x0), lamb, r2
    cdef StopState stop
    init_stop(&stop, error, norm_inf(x0), etol, ertol, ptol, prtol)
    if check_stop(&stop, error, precision, False):
        return NewtonMethodsReturnType(
            x0, F_x0, (J_x0,), step, (G.F.n_f_calls, G.J.n_f_calls),
            precision, error, successful(stop.reason), stop.reason == RESIDUAL_TOLERANCE,
            termination_reason=<int>stop.reason)

    cdef np.ndarray[np.float64_t, ndim=2] a, I = np.eye(x0.shape[0], dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] xminus1, candidate, candidate_F
    cdef np.ndarray[np.float64_t, ndim=2] J_xminus1 = J_x0.copy()
    cdef np.ndarray[np.float64_t, ndim=3] H_xminus1 = H_x0.copy()

    if adaptive:
        r2 = vops.norm(d_x) ** 2
        if r2 > 0:
            xminus1 = x0 - d_x
            if not np.isfinite(xminus1).all():
                stop.reason = NONFINITE
            else:
                J_xminus1 = G.eval_J(xminus1)
                H_xminus1 = G.eval_H(xminus1)
                if not np.isfinite(J_xminus1).all() or not np.isfinite(H_xminus1).all():
                    stop.reason = NONFINITE
                else:
                    hess_lip = vops.norm(J_x0.squeeze(0) - J_xminus1.squeeze(0) - H_xminus1.squeeze(0) @ d_x) / r2

    while not check_stop(&stop, error, precision, step > 0):
        if step >= max_iter > 0:
            stop.reason = MAX_ITERATIONS
            break
        if (J_x0.shape[0] != 1 or J_x0.shape[1] != x0.shape[0] or H_x0.shape[0] != 1 or
                H_x0.shape[1] != x0.shape[0] or H_x0.shape[2] != x0.shape[0]):
            raise ValueError('Objective derivative shape does not match the input dimension.')
        if not np.isfinite(J_x0).all() or not np.isfinite(H_x0).all():
            stop.reason = NONFINITE
            break
        if adaptive:
            hess_lip = max(empirical_hess_lip(d_x, J_x0.squeeze(0), J_xminus1.squeeze(0), H_xminus1.squeeze(0)),
                           hess_lip / 2)
        lamb = sops.sqrt(hess_lip * vops.norm(J_x0.squeeze(0)))
        a = H_x0.squeeze(0) + lamb * I
        if not np.isfinite(a).all():
            stop.reason = NONFINITE
            break
        try:
            d_x = mops.inverse(a, -J_x0.squeeze(0))
        except np.linalg.LinAlgError:
            stop.reason = BREAKDOWN
            break
        candidate = x0 + d_x
        if not np.isfinite(d_x).all() or not np.isfinite(candidate).all():
            stop.reason = NONFINITE
            break
        candidate_F = G.eval_G(candidate)
        step += 1
        if candidate_F.shape[0] != F_x0.shape[0]:
            raise ValueError('Residual output dimension changed during iteration.')
        if not np.isfinite(candidate).all() or not np.isfinite(candidate_F).all():
            stop.reason = NONFINITE
            break
        precision = norm_inf(candidate - x0)
        x0, F_x0 = candidate, candidate_F
        error = norm_inf(F_x0)
        advance_x(&stop, norm_inf(x0))
        if check_stop(&stop, error, precision, True):
            J_x0 = np.full((1, x0.shape[0]), np.nan)
            break
        if step >= max_iter > 0:
            stop.reason = MAX_ITERATIONS
            J_x0 = np.full((1, x0.shape[0]), np.nan)
            break
        if adaptive:
            J_xminus1, H_xminus1 = J_x0, H_x0
        J_x0 = G.eval_J(x0)
        H_x0 = G.eval_H(x0)

    finish_stop(&stop, error, precision, step > 0)
    return NewtonMethodsReturnType(
        x0, F_x0, (J_x0,), step, (G.F.n_f_calls, G.J.n_f_calls),
        precision, error, successful(stop.reason), stop.reason == RESIDUAL_TOLERANCE,
        termination_reason=<int>stop.reason)

# noinspection DuplicatedCode
@register('cyroot.vector.newton')
@cython.binding(True)
def generalized_mishchenko(F: Callable[[VectorLike], VectorLike],
                           J: Optional[Callable[[VectorLike], Array2DLike]],
                           H: Optional[Callable[[VectorLike], Array3DLike]],
                           x0: VectorLike,
                           F_x0: Optional[VectorLike] = None,
                           J_x0: Optional[Array2DLike] = None,
                           H_x0: Optional[Array3DLike] = None,
                           d_x0: Optional[VectorLike] = None,
                           hess_lip: Optional[float] = None,
                           adaptive: bool = False,
                           h: Optional[Union[float, VectorLike]] = FINITE_DIFF_STEP,
                           etol: float = ETOL,
                           ertol: float = ERTOL,
                           ptol: float = PTOL,
                           prtol: float = PRTOL,
                           max_iter = MAX_ITER) -> NewtonMethodsReturnType:
    r"""
    Generalized Mishchenko's regularized Newton method for vector root-finding as presented in the paper
    "Regularized Newton Method with Global :math:`\mathcal{O}(1/k^2)` Convergence".

    References:
        https://arxiv.org/abs/2112.02089

    Args:
        F (function): Function for which the root is sought.
        J (function, optional): Function returning the Jacobian
          of ``F``.
        H (function, optional): Function returning the Hessian
          of ``F``.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
          point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
        H_x0 (np.ndarray, optional): Hessian at initial point.
        d_x0 (np.ndarray, optional): Initial delta used when ``adaptive=True``.
          Defaults to None.
        hess_lip (double, optional): estimate for the Hessian Lipschitz
          constant. If not provided, it is estimated or a small value
          is used. Defaults to None.
        adaptive (bool): Use the adaptive variant or not. Defaults to False.
        h (float, np.ndarray, optional): Finite difference step size,
          ignored when ``J`` and ``H`` are not None. Defaults to {h}.
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
    if hess_lip is None:
        hess_lip = 1e-5
        if not adaptive:
            warn_value(
                f'No estimate of Hessian-Lipschitzness is given, '
                f'a small value {hess_lip} is used as a heuristic.')

    x0 = np.asarray(x0, dtype=np.float64)
    if x0.ndim != 1 or x0.size == 0:
        raise ValueError('x0 must be a nonempty vector.')
    if not np.isfinite(hess_lip) or hess_lip < 0:
        raise ValueError('hess_lip must be finite and nonnegative.')

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if J is None:
        J_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=1)
    else:
        J_wrapper = PyNdArrayFPtr.from_f(J)
    # if H is None:
    #     H_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=2)
    # else:
    #     H_wrapper = PyNdArrayFPtr.from_f(H)

    G_wrapper = VectorOptFunctionWrapper(F_wrapper, J_wrapper)

    F_x0 = G_wrapper.eval_G(x0)
    if F_x0.ndim != 1 or F_x0.size == 0:
        raise ValueError('F must return a nonempty vector.')
    initial_done = (not np.isfinite(x0).all() or not np.isfinite(F_x0).all() or
                    np.max(np.abs(F_x0)) <= etol + ertol * np.max(np.abs(F_x0)))
    J_x0 = np.full((1, x0.size), np.nan) if initial_done else G_wrapper.eval_J(x0)
    H_x0 = np.full((1, x0.size, x0.size), np.nan) if initial_done else G_wrapper.eval_H(x0)
    # if H_x0 is None:
    #     H_x0 = H_wrapper.eval(x0)
    # else:
    #     H_x0 = np.asarray(H_x0, dtype=np.float64)

    if d_x0 is None:
        d_x0 = np.full_like(x0, 1e-3) if adaptive else np.zeros_like(x0)
    else:
        d_x0 = np.asarray(d_x0, dtype=np.float64)
        if d_x0.shape != x0.shape or not np.isfinite(d_x0).all():
            raise ValueError('d_x0 must be a finite vector with the same shape as x0.')

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got d_in={x0.shape[0]}, d_out={F_x0.shape[0]}.')

    return generalized_mishchenko_kernel(
        G_wrapper, x0, F_x0, J_x0, H_x0, d_x0, hess_lip, adaptive,
        etol, ertol, ptol, prtol, max_iter)
