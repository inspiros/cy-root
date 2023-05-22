# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

import numpy as np
cimport numpy as np
import cython
from dynamic_default_args import dynamic_default_args, named_default

from ._check_args import _check_stop_cond_args
from ._check_args cimport _check_stop_cond_vector_initial_guess
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER, FINITE_DIFF_STEP
from ._types import VectorLike, Array2DLike, Array3DLike
from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .ops cimport scalar_ops as sops, vector_ops as vops, matrix_ops as mops
from .return_types cimport NewtonMethodsReturnType
from .utils._function_registering import register
from .utils._warnings import warn_value
from .vector_derivative_approximation import GeneralizedFiniteDifference, VectorDerivativeApproximation

__all__ = [
    'generalized_newton',
    'generalized_halley',
    'generalized_super_halley',
    'generalized_chebyshev',
    'generalized_tangent_hyperbolas',
]

################################################################################
# Generalized Newton
################################################################################
# noinspection DuplicatedCode
cdef NewtonMethodsReturnType generalized_newton_kernel(
        NdArrayFPtr F,
        NdArrayFPtr J,
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
        return NewtonMethodsReturnType(
            x0, F_x0, J_x0, step, (F.n_f_calls, J.n_f_calls), precision, error, converged, optimal)

    cdef bint use_derivative_approximation = isinstance(J, VectorDerivativeApproximation)
    cdef np.ndarray[np.float64_t, ndim=1] d_x
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        d_x = mops.inv(J_x0, -F_x0)
        x0 = x0 + d_x
        F_x0 = F.eval(x0)
        if use_derivative_approximation:
            J_x0 = J.eval_with_f_val(x0, F_x0)
        else:
            J_x0 = J.eval(x0)
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, F_x0, J_x0, step, (F.n_f_calls, J.n_f_calls), precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_newton(F: Callable[[VectorLike], VectorLike],
                       J: Optional[Callable[[VectorLike], Array2DLike]],
                       x0: VectorLike,
                       F_x0: Optional[VectorLike] = None,
                       J_x0: Optional[Array2DLike] = None,
                       h: Optional[Union[float, VectorLike]] = named_default(
                           FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                       etol: float = named_default(ETOL=ETOL),
                       ertol: float = named_default(ERTOL=ERTOL),
                       ptol: float = named_default(PTOL=PTOL),
                       prtol: float = named_default(PRTOL=PRTOL),
                       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Generalized Newton's method for vector root-finding.

    Args:
        F (function): Function for which the root is sought.
        J (function, optional): Function returning the Jacobian
         of ``F``.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
         point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
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

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if J is None:
        J_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=1)
    else:
        J_wrapper = PyNdArrayFPtr.from_f(J)

    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = J_wrapper.eval(x0)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got d_in={x0.shape[0]}, d_out={F_x0.shape[0]}.')

    res = generalized_newton_kernel(
        F_wrapper, J_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    return res

################################################################################
# Generalized Halley
################################################################################
# noinspection DuplicatedCode
cdef NewtonMethodsReturnType generalized_halley_kernel(
        NdArrayFPtr F,
        NdArrayFPtr J,
        NdArrayFPtr H,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        np.ndarray[np.float64_t, ndim=3] H_x0,
        double etol=ETOL,
        double ertol=ERTOL,
        double ptol=PTOL,
        double prtol=PRTOL,
        unsigned long max_iter=MAX_ITER):
    cdef unsigned long d_in = x0.shape[0], d_out = F_x0.shape[0]
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_cond_vector_initial_guess(x0, F_x0, etol, ertol, ptol, prtol,
                                             &precision, &error, &converged, &optimal):
        return NewtonMethodsReturnType(
            x0, F_x0, J_x0, H_x0, step, (F.n_f_calls, J.n_f_calls, H.n_f_calls),
            precision, error, converged, optimal)

    cdef unsigned long i
    cdef bint[2] use_derivative_approximation = [isinstance(J, VectorDerivativeApproximation),
                                                 isinstance(H, VectorDerivativeApproximation)]
    cdef np.ndarray[np.float64_t, ndim=1] a, b, denom = np.zeros(d_in, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] d_x = np.zeros(d_in, dtype=np.float64)
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        a = mops.inv(J_x0, -F_x0)  # -J^-1.F
        b = mops.inv(J_x0, H_x0.dot(a).dot(a))  # J^-1.H.a^2
        denom[:d_out] = a[:d_out] + .5 * b[:d_out]
        for i in range(min(d_in, d_out)):
            if denom[i] == 0:
                converged = False
        if not converged:
            break

        d_x[:d_out] = np.power(a[:d_out], 2) / denom[:d_out]  # a^2 / (a + .5 * b)
        x0 = x0 + d_x
        F_x0 = F.eval(x0)
        if use_derivative_approximation[0]:
            J_x0 = J.eval_with_f_val(x0, F_x0)
        else:
            J_x0 = J.eval(x0)
        if use_derivative_approximation[1]:
            H_x0 = H.eval_with_f_val(x0, F_x0)
        else:
            H_x0 = H.eval(x0)
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, F_x0, (J_x0, H_x0), step, (F.n_f_calls, J.n_f_calls, H.n_f_calls),
        precision, error, converged, optimal)

# noinspection DuplicatedCode
cdef NewtonMethodsReturnType generalized_modified_halley_kernel(
        NdArrayFPtr F,
        NdArrayFPtr J,
        NdArrayFPtr H,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        np.ndarray[np.float64_t, ndim=3] H_x0,
        double alpha=0.5,
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
        return NewtonMethodsReturnType(
            x0, F_x0, J_x0, H_x0, step, (F.n_f_calls, J.n_f_calls, H.n_f_calls),
            precision, error, converged, optimal)

    cdef bint[2] use_derivative_approximation = [isinstance(J, VectorDerivativeApproximation),
                                                 isinstance(H, VectorDerivativeApproximation)]
    cdef np.ndarray[np.float64_t, ndim=1] d_x, a
    cdef np.ndarray[np.float64_t, ndim=2] L_F, I = np.eye(x0.shape[0], dtype=np.float64)
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        a = mops.inv(J_x0, F_x0)  # J^-1.F
        L_F = mops.inv(J_x0, H_x0.dot(a))  # J^-1.H.J^-1.F
        d_x = -(I + .5 * L_F.dot(mops.inv(I - alpha * L_F))).dot(a)  # (I - alpha * L_f)^-1
        x0 = x0 + d_x
        F_x0 = F.eval(x0)
        if use_derivative_approximation[0]:
            J_x0 = J.eval_with_f_val(x0, F_x0)
        else:
            J_x0 = J.eval(x0)
        if use_derivative_approximation[1]:
            H_x0 = H.eval_with_f_val(x0, F_x0)
        else:
            H_x0 = H.eval(x0)
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, F_x0, (J_x0, H_x0), step, (F.n_f_calls, J.n_f_calls, H.n_f_calls),
        precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_halley(F: Callable[[VectorLike], VectorLike],
                       J: Optional[Callable[[VectorLike], Array2DLike]],
                       H: Optional[Callable[[VectorLike], Array3DLike]],
                       x0: VectorLike,
                       F_x0: Optional[VectorLike] = None,
                       J_x0: Optional[Array2DLike] = None,
                       H_x0: Optional[Array3DLike] = None,
                       alpha: Optional[float] = None,
                       h: Optional[Union[float, VectorLike]] = named_default(
                           FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                       etol: float = named_default(ETOL=ETOL),
                       ertol: float = named_default(ERTOL=ERTOL),
                       ptol: float = named_default(PTOL=PTOL),
                       prtol: float = named_default(PRTOL=PRTOL),
                       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Generalized Halley's method for vector root-finding as presented in the paper
    "Abstract Padé-approximants for the solution of a system of nonlinear equations".

    References:
        https://www.sciencedirect.com/science/article/pii/0898122183901190

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
        alpha (float, optional): If set, the modified halley
         formula which has parameter alpha will be used.
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

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if J is None:
        J_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=1)
    else:
        J_wrapper = PyNdArrayFPtr.from_f(J)
    if H is None:
        H_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=2)
    else:
        H_wrapper = PyNdArrayFPtr.from_f(H)

    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = J_wrapper.eval(x0)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)
    if H_x0 is None:
        H_x0 = H_wrapper.eval(x0)
    else:
        H_x0 = np.asarray(H_x0, dtype=np.float64)

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got d_in={x0.shape[0]}, d_out={F_x0.shape[0]}.')

    if alpha is None:
        res = generalized_halley_kernel(
            F_wrapper, J_wrapper, H_wrapper, x0, F_x0, J_x0, H_x0,
            etol, ertol, ptol, prtol, max_iter)
    else:
        res = generalized_modified_halley_kernel(
            F_wrapper, J_wrapper, H_wrapper, x0, F_x0, J_x0, H_x0,
            alpha, etol, ertol, ptol, prtol, max_iter)
    return res

#------------------------
# Super-Halley
#------------------------
# noinspection DuplicatedCode
@register('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_super_halley(F: Callable[[VectorLike], VectorLike],
                             J: Optional[Callable[[VectorLike], Array2DLike]],
                             H: Optional[Callable[[VectorLike], Array3DLike]],
                             x0: VectorLike,
                             F_x0: Optional[VectorLike] = None,
                             J_x0: Optional[Array2DLike] = None,
                             H_x0: Optional[Array3DLike] = None,
                             h: Optional[Union[float, VectorLike]] = named_default(
                                 FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                             etol: float = named_default(ETOL=ETOL),
                             ertol: float = named_default(ERTOL=ERTOL),
                             ptol: float = named_default(PTOL=PTOL),
                             prtol: float = named_default(PRTOL=PRTOL),
                             max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Generalized Super-Halley's method for vector root-finding.
    This is equivalent to calling ``generalized_halley`` with ``alpha=1``.

    References:
        https://www.sciencedirect.com/science/article/abs/pii/S0096300399001757

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
    return generalized_halley(F, J, H, x0, F_x0, J_x0, H_x0, 1, h, etol, ertol, ptol, prtol, max_iter)

#------------------------
# Chebyshev
#------------------------
# noinspection DuplicatedCode
@register('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_chebyshev(F: Callable[[VectorLike], VectorLike],
                          J: Optional[Callable[[VectorLike], Array2DLike]],
                          H: Optional[Callable[[VectorLike], Array3DLike]],
                          x0: VectorLike,
                          F_x0: Optional[VectorLike] = None,
                          J_x0: Optional[Array2DLike] = None,
                          H_x0: Optional[Array3DLike] = None,
                          h: Optional[Union[float, VectorLike]] = named_default(
                              FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                          etol: float = named_default(ETOL=ETOL),
                          ertol: float = named_default(ERTOL=ERTOL),
                          ptol: float = named_default(PTOL=PTOL),
                          prtol: float = named_default(PRTOL=PRTOL),
                          max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Generalized Chebyshev's method for vector root-finding.
    This is equivalent to calling ``generalized_halley`` with ``alpha=0``.

    References:
        https://www.sciencedirect.com/science/article/abs/pii/S0096300399001757

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
    return generalized_halley(F, J, H, x0, F_x0, J_x0, H_x0, 0, h, etol, ertol, ptol, prtol, max_iter)

#------------------------
# Tangent Hyperbolas
#------------------------
# noinspection DuplicatedCode
cdef NewtonMethodsReturnType generalized_tangent_hyperbolas_kernel(
        NdArrayFPtr F,
        NdArrayFPtr J,
        NdArrayFPtr H,
        np.ndarray[np.float64_t, ndim=1] x0,
        np.ndarray[np.float64_t, ndim=1] F_x0,
        np.ndarray[np.float64_t, ndim=2] J_x0,
        np.ndarray[np.float64_t, ndim=3] H_x0,
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
        return NewtonMethodsReturnType(
            x0, F_x0, J_x0, H_x0, step, (F.n_f_calls, J.n_f_calls, H.n_f_calls),
            precision, error, converged, optimal)

    cdef bint[2] use_derivative_approximation = [isinstance(J, VectorDerivativeApproximation),
                                                 isinstance(H, VectorDerivativeApproximation)]
    cdef np.ndarray[np.float64_t, ndim=1] d_x, a
    cdef np.ndarray[np.float64_t, ndim=2] I, B
    if formula == 1:
        I = np.eye(F_x0.shape[0], dtype=np.float64)
    converged = True
    while not (sops.isclose(0, error, ertol, etol) or
               sops.isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        a = mops.inv(J_x0, -F_x0)  # -J^-1.F
        if formula == 2:  # more likely
            d_x = mops.inv(J_x0 + .5 * H_x0.dot(a), -F_x0)  # (J + .5 * H.a)^-1.F
        else:
            B = mops.inv(J_x0, H_x0.dot(a))  # J^-1.H.a
            d_x = mops.inv(I + .5 * B, a)  # (I + .5 * J^-1.H.a)^-1.a
        x0 = x0 + d_x
        F_x0 = F.eval(x0)
        if use_derivative_approximation[0]:
            J_x0 = J.eval_with_f_val(x0, F_x0)
        else:
            J_x0 = J.eval(x0)
        if use_derivative_approximation[1]:
            H_x0 = H.eval_with_f_val(x0, F_x0)
        else:
            H_x0 = H.eval(x0)
        precision = vops.max(vops.fabs(d_x))
        error = vops.max(vops.fabs(F_x0))

    optimal = sops.isclose(0, error, ertol, etol)
    return NewtonMethodsReturnType(
        x0, F_x0, (J_x0, H_x0), step, (F.n_f_calls, J.n_f_calls, H.n_f_calls),
        precision, error, converged, optimal)

# noinspection DuplicatedCode
@register('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_tangent_hyperbolas(F: Callable[[VectorLike], VectorLike],
                                   J: Optional[Callable[[VectorLike], Array2DLike]],
                                   H: Optional[Callable[[VectorLike], Array3DLike]],
                                   x0: VectorLike,
                                   F_x0: Optional[VectorLike] = None,
                                   J_x0: Optional[Array2DLike] = None,
                                   H_x0: Optional[Array3DLike] = None,
                                   formula: int = 2,
                                   h: Optional[Union[float, VectorLike]] = named_default(
                                       FINITE_DIFF_STEP=FINITE_DIFF_STEP),
                                   etol: float = named_default(ETOL=ETOL),
                                   ertol: float = named_default(ERTOL=ERTOL),
                                   ptol: float = named_default(PTOL=PTOL),
                                   prtol: float = named_default(PRTOL=PRTOL),
                                   max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodsReturnType:
    """
    Generalized Tangent Hyperbolas method for vector root-finding.

    This method is the same as Generalized Halley's method.
    There are 2 formulas ``formula=1`` and ``formula=2``, the later
    (default) requires 1 less matrix inversion.

    This is essentially similar to generalized Halley method.

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
        formula (int): Formula 1 or 2. Defaults to 2.
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
    if formula not in [1, 2]:
        raise ValueError(f'Unknown formula {formula}.')

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if J is None:
        J_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=1)
    else:
        J_wrapper = PyNdArrayFPtr.from_f(J)
    if H is None:
        H_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=2)
    else:
        H_wrapper = PyNdArrayFPtr.from_f(H)

    if F_x0 is None:
        F_x0 = F_wrapper.eval(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = J_wrapper.eval(x0)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)
    if H_x0 is None:
        H_x0 = H_wrapper.eval(x0)
    else:
        H_x0 = np.asarray(H_x0, dtype=np.float64)

    if x0.shape[0] < F_x0.shape[0]:
        warn_value('Input dimension is smaller than output dimension. '
                   f'Got d_in={x0.shape[0]}, d_out={F_x0.shape[0]}.')

    res = generalized_tangent_hyperbolas_kernel(
        F_wrapper, J_wrapper, H_wrapper, x0, F_x0, J_x0, H_x0, formula,
        etol, ertol, ptol, prtol, max_iter)
    return res
