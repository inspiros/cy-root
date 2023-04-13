# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Callable, Optional, Union

cimport numpy as np
import numpy as np
import cython
from dynamic_default_args import dynamic_default_args, named_default

from ._check_args import _check_stop_condition_args
from ._check_args cimport _check_stop_condition_initial_guess_vector
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER, FINITE_DIFF_STEP
from ._return_types import NewtonMethodReturnType
from .fptr cimport NdArrayFPtr, PyNdArrayFPtr
from .ops.scalar_ops cimport isclose
from .ops.vector_ops cimport fabs, max
from .typing import *
from .utils.function_tagging import tag
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
@cython.returns((np.ndarray, np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef generalized_newton_kernel(
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
    if _check_stop_condition_initial_guess_vector(x0, F_x0, etol, ertol, ptol, prtol,
                                                  &precision, &error, &converged, &optimal):
        return x0, F_x0, J_x0, step, precision, error, converged, optimal

    cdef bint use_derivative_approximation = isinstance(J, VectorDerivativeApproximation)
    cdef np.ndarray[np.float64_t, ndim=1] d_x
    converged = True
    while not (isclose(0, error, ertol, etol) or isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        d_x = np.linalg.solve(J_x0, -F_x0)
        x0 = x0 + d_x
        F_x0 = F.eval(x0)
        if use_derivative_approximation:
            J_x0 = J.eval_with_f_val(x0, F_x0)
        else:
            J_x0 = J.eval(x0)
        precision = max(fabs(d_x))
        error = max(fabs(F_x0))

    optimal = isclose(0, error, ertol, etol)
    return x0, F_x0, J_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
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
                       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)

    x0 = np.asarray(x0, dtype=np.float64)

    F_wrapper = PyNdArrayFPtr.from_f(F)
    if J is None:
        J_wrapper = GeneralizedFiniteDifference(F_wrapper, h=h, order=1)
    else:
        J_wrapper = PyNdArrayFPtr.from_f(J)

    if F_x0 is None:
        F_x0 = F_wrapper(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = J_wrapper(x0)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)

    res = generalized_newton_kernel(
        <NdArrayFPtr> F_wrapper, <NdArrayFPtr> J_wrapper,
        x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    return NewtonMethodReturnType.from_results(res, (F_wrapper.n_f_calls, J_wrapper.n_f_calls))

################################################################################
# Generalized Halley
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, tuple[np.ndarray], cython.unsignedlong, double, double, bint, bint))
cdef generalized_halley_kernel(
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
    cdef unsigned long step = 0
    cdef double precision, error
    cdef bint converged, optimal
    if _check_stop_condition_initial_guess_vector(x0, F_x0, etol, ertol, ptol, prtol,
                                                  &precision, &error, &converged, &optimal):
        return x0, F_x0, J_x0, H_x0, step, precision, error, converged, optimal

    cdef bint[2] use_derivative_approximation = [isinstance(J, VectorDerivativeApproximation),
                                                 isinstance(H, VectorDerivativeApproximation)]
    cdef np.ndarray[np.float64_t, ndim=1] d_x, a, b
    converged = True
    while not (isclose(0, error, ertol, etol) or isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        a = np.linalg.solve(J_x0, -F_x0)  # -J^-1.F
        b = np.linalg.solve(J_x0, H_x0.dot(a).dot(a))  # J^-1.H.a^2
        d_x = np.power(a, 2) / (a + .5 * b)  # a^2 / (a + .5 * b)
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
        precision = max(fabs(d_x))
        error = max(fabs(F_x0))

    optimal = isclose(0, error, ertol, etol)
    return x0, F_x0, (J_x0, H_x0), step, precision, error, converged, optimal

# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, tuple[np.ndarray], cython.unsignedlong, double, double, bint, bint))
cdef generalized_modified_halley_kernel(
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
    if _check_stop_condition_initial_guess_vector(x0, F_x0, etol, ertol, ptol, prtol,
                                                  &precision, &error, &converged, &optimal):
        return x0, F_x0, J_x0, H_x0, step, precision, error, converged, optimal

    cdef bint[2] use_derivative_approximation = [isinstance(J, VectorDerivativeApproximation),
                                                 isinstance(H, VectorDerivativeApproximation)]
    cdef np.ndarray[np.float64_t, ndim=1] d_x, a
    cdef np.ndarray[np.float64_t, ndim=2] L_F, I_sub_L_F_inv, I = np.eye(x0.shape[0], dtype=np.float64)
    converged = True
    while not (isclose(0, error, ertol, etol) or isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        a = np.linalg.solve(J_x0, F_x0)  # J^-1.F
        L_F = np.linalg.solve(J_x0, H_x0.dot(a))  # J^-1.H.J^-1.F
        I_sub_L_F_inv = np.linalg.solve(I - alpha * L_F, I)  # (I - alpha * L_f)^-1
        d_x = -(I + .5 * L_F.dot(I_sub_L_F_inv)).dot(a)
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
        precision = max(fabs(d_x))
        error = max(fabs(F_x0))

    optimal = isclose(0, error, ertol, etol)
    return x0, F_x0, (J_x0, H_x0), step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
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
                       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)

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
        F_x0 = F_wrapper(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = J_wrapper(x0)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)
    if H_x0 is None:
        H_x0 = H_wrapper(x0)
    else:
        H_x0 = np.asarray(H_x0, dtype=np.float64)

    if alpha is None:
        res = generalized_halley_kernel(
            <NdArrayFPtr> F_wrapper, <NdArrayFPtr> J_wrapper, <NdArrayFPtr> H_wrapper,
            x0, F_x0, J_x0, H_x0, etol, ertol, ptol, prtol, max_iter)
    else:
        res = generalized_modified_halley_kernel(
            <NdArrayFPtr> F_wrapper, <NdArrayFPtr> J_wrapper, <NdArrayFPtr> H_wrapper,
            x0, F_x0, J_x0, H_x0, alpha, etol, ertol, ptol, prtol, max_iter)
    return NewtonMethodReturnType.from_results(res, (F_wrapper.n_f_calls,
                                                     J_wrapper.n_f_calls,
                                                     H_wrapper.n_f_calls))

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
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
                             max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
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

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
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
                          max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
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
@cython.returns((np.ndarray, np.ndarray, tuple[np.ndarray], cython.unsignedlong, double, double, bint, bint))
cdef generalized_tangent_hyperbolas_kernel(
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
    if _check_stop_condition_initial_guess_vector(x0, F_x0, etol, ertol, ptol, prtol,
                                                  &precision, &error, &converged, &optimal):
        return x0, F_x0, J_x0, H_x0, step, precision, error, converged, optimal

    cdef bint[2] use_derivative_approximation = [isinstance(J, VectorDerivativeApproximation),
                                                 isinstance(H, VectorDerivativeApproximation)]
    cdef np.ndarray[np.float64_t, ndim=1] d_x, a
    cdef np.ndarray[np.float64_t, ndim=2] I, B
    if formula == 1:
        I = np.eye(F_x0.shape[0], dtype=np.float64)
    converged = True
    while not (isclose(0, error, ertol, etol) or isclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        a = np.linalg.solve(J_x0, -F_x0)  # -J^-1.F
        if formula == 2:  # more likely
            d_x = np.linalg.solve(J_x0 + .5 * H_x0.dot(a), -F_x0)  # (J + .5 * H.a)^-1.F
        else:
            B = np.linalg.solve(J_x0, H_x0.dot(a))  # J^-1.H.a
            d_x = np.linalg.solve(I + .5 * B, a)  # (I + .5 * J^-1.H.a)^-1.a
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
        precision = max(fabs(d_x))
        error = max(fabs(F_x0))

    optimal = isclose(0, error, ertol, etol)
    return x0, F_x0, (J_x0, H_x0), step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
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
                                   max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
    """
    Generalized Tangent Hyperbolas method for vector root-finding.
    There are 2 formulas implemented ``formula=1`` and ``formula=2``.
    The later (default) requires 1 less matrix inversion.

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
        formula (int): Formula 1 or 2. Defaults to {formula}.
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
    etol, ertol, ptol, prtol, max_iter = _check_stop_condition_args(etol, ertol, ptol, prtol, max_iter)
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
        F_x0 = F_wrapper(x0)
    else:
        F_x0 = np.asarray(F_x0, dtype=np.float64)
    if J_x0 is None:
        J_x0 = J_wrapper(x0)
    else:
        J_x0 = np.asarray(J_x0, dtype=np.float64)
    if H_x0 is None:
        H_x0 = H_wrapper(x0)
    else:
        H_x0 = np.asarray(H_x0, dtype=np.float64)

    res = generalized_tangent_hyperbolas_kernel(
        <NdArrayFPtr> F_wrapper, <NdArrayFPtr> J_wrapper, <NdArrayFPtr> H_wrapper,
        x0, F_x0, J_x0, H_x0, formula, etol, ertol, ptol, prtol, max_iter)
    return NewtonMethodReturnType.from_results(res, (F_wrapper.n_f_calls,
                                                     J_wrapper.n_f_calls,
                                                     H_wrapper.n_f_calls))
