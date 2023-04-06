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
)
from ._check_args import (
    _check_stop_condition_args,
)
from ._defaults import ETOL, ERTOL, PTOL, PRTOL, MAX_ITER
from ._return_types import NewtonMethodReturnType
from .fptr cimport (
double_np_ndarray_func_type, DoubleNpNdArrayFPtr, PyDoubleNpNdArrayFPtr,
)
from .utils.scalar_ops cimport fisclose
from .utils.vector_ops cimport fabs, fmax
from .utils.function_tagging import tag

__all__ = [
    'generalized_newton',
    'generalized_halley',
    'generalized_tangent_hyperbolas',
]

################################################################################
# Generalized Newton
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, np.ndarray, cython.unsignedlong, double, double, bint, bint))
cdef generalized_newton_kernel(
        double_np_ndarray_func_type F,
        double_np_ndarray_func_type J,
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

    cdef np.ndarray[np.float64_t, ndim=1] d_x
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        # d_x = -np.linalg.inv(J_x0).dot(F_x0)
        d_x = np.linalg.solve(J_x0, -F_x0)
        x0 = x0 + d_x
        F_x0 = F(x0)
        J_x0 = J(x0)
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x0))

    optimal = fisclose(0, error, ertol, etol)
    return x0, F_x0, J_x0, step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_newton(F: Callable[[np.ndarray], np.ndarray],
                       J: Callable[[np.ndarray], np.ndarray],
                       x0: np.ndarray,
                       F_x0: Optional[np.ndarray] = None,
                       J_x0: Optional[np.ndarray] = None,
                       etol: float = named_default(ETOL=ETOL),
                       ertol: float = named_default(ERTOL=ERTOL),
                       ptol: float = named_default(PTOL=PTOL),
                       prtol: float = named_default(PRTOL=PRTOL),
                       max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
    """
    Generalized Newton's method for vector root-finding.

    Args:
        F (function): Function for which the root is sought.
        J (function): Function returning the Jacobian of F.
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

    F_wrapper = PyDoubleNpNdArrayFPtr(F)
    J_wrapper = PyDoubleNpNdArrayFPtr(J)
    if F_x0 is None:
        F_x0 = F_wrapper(x0)
    if J_x0 is None:
        J_x0 = J_wrapper(x0)

    res = generalized_newton_kernel[DoubleNpNdArrayFPtr](
        F_wrapper, J_wrapper, x0, F_x0, J_x0, etol, ertol, ptol, prtol, max_iter)
    return NewtonMethodReturnType.from_results(res, (F_wrapper.n_f_calls, J_wrapper.n_f_calls))

################################################################################
# Generalized Halley
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, tuple[np.ndarray], cython.unsignedlong, double, double, bint, bint))
cdef generalized_halley_kernel(
        double_np_ndarray_func_type F,
        double_np_ndarray_func_type J,
        double_np_ndarray_func_type H,
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

    cdef np.ndarray[np.float64_t, ndim=1] d_x, a, b
    cdef np.ndarray[np.float64_t, ndim=2] denom
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
        if step >= max_iter > 0:
            converged = False
            break
        step += 1

        # denom = 2 * J_x0.dot(J_x0) - F_x0.dot(H_x0)
        # s = -2 * np.linalg.inv(denom).dot(F_x0.dot(J_x0))
        a = np.linalg.solve(J_x0, -F_x0)  # -J^-1.F
        b = np.linalg.solve(J_x0, H_x0.dot(a).dot(a))  # J^-1.H.a^2
        d_x = np.power(a, 2) / (a + .5 * b)  # a^2 / (a + .5 * b)
        x0 = x0 + d_x
        F_x0 = F(x0)
        J_x0 = J(x0)
        H_x0 = H(x0)
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x0))

    optimal = fisclose(0, error, ertol, etol)
    return x0, F_x0, (J_x0, H_x0), step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_halley(F: Callable[[np.ndarray], np.ndarray],
                       J: Callable[[np.ndarray], np.ndarray],
                       H: Callable[[np.ndarray], np.ndarray],
                       x0: np.ndarray,
                       F_x0: Optional[np.ndarray] = None,
                       J_x0: Optional[np.ndarray] = None,
                       H_x0: Optional[np.ndarray] = None,
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
        J (function): Function returning the Jacobian of F.
        H (function): Function returning the Hessian of F.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
         point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
        H_x0 (np.ndarray, optional): Hessian at initial point.
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

    F_wrapper = PyDoubleNpNdArrayFPtr(F)
    J_wrapper = PyDoubleNpNdArrayFPtr(J)
    H_wrapper = PyDoubleNpNdArrayFPtr(H)
    if F_x0 is None:
        F_x0 = F_wrapper(x0)
    if J_x0 is None:
        J_x0 = J_wrapper(x0)
    if H_x0 is None:
        H_x0 = H_wrapper(x0)

    res = generalized_halley_kernel[DoubleNpNdArrayFPtr](
        F_wrapper, J_wrapper, H_wrapper, x0, F_x0, J_x0, H_x0, etol, ertol, ptol, prtol, max_iter)
    return NewtonMethodReturnType.from_results(res, (F_wrapper.n_f_calls,
                                                     J_wrapper.n_f_calls,
                                                     H_wrapper.n_f_calls))

################################################################################
# Generalized Tangent Hyperbolas
################################################################################
# noinspection DuplicatedCode
@cython.returns((np.ndarray, np.ndarray, tuple[np.ndarray], cython.unsignedlong, double, double, bint, bint))
cdef generalized_tangent_hyperbolas_kernel(
        double_np_ndarray_func_type F,
        double_np_ndarray_func_type J,
        double_np_ndarray_func_type H,
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

    cdef np.ndarray[np.float64_t, ndim=1] d_x, a
    cdef np.ndarray[np.float64_t, ndim=2] I, B
    if formula == 1:
        I = np.eye(F_x0.shape[0], dtype=np.float64)
    converged = True
    while not (fisclose(0, error, ertol, etol) or fisclose(0, precision, prtol, ptol)):
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
        F_x0 = F(x0)
        J_x0 = J(x0)
        H_x0 = H(x0)
        precision = fmax(fabs(d_x))
        error = fmax(fabs(F_x0))

    optimal = fisclose(0, error, ertol, etol)
    return x0, F_x0, (J_x0, H_x0), step, precision, error, converged, optimal

# noinspection DuplicatedCode
@tag('cyroot.vector.newton')
@dynamic_default_args()
@cython.binding(True)
def generalized_tangent_hyperbolas(F: Callable[[np.ndarray], np.ndarray],
                                   J: Callable[[np.ndarray], np.ndarray],
                                   H: Callable[[np.ndarray], np.ndarray],
                                   x0: np.ndarray,
                                   F_x0: Optional[np.ndarray] = None,
                                   J_x0: Optional[np.ndarray] = None,
                                   H_x0: Optional[np.ndarray] = None,
                                   formula: int = 2,
                                   etol: float = named_default(ETOL=ETOL),
                                   ertol: float = named_default(ERTOL=ERTOL),
                                   ptol: float = named_default(PTOL=PTOL),
                                   prtol: float = named_default(PRTOL=PRTOL),
                                   max_iter: int = named_default(MAX_ITER=MAX_ITER)) -> NewtonMethodReturnType:
    """
    Generalized Tangent Hyperbolas method for vector root-finding.
    There are 2 formulas implemented (which can be set through
    parameter ``formula``), the later (default) one requires 1 less
    matrix inversion.

    Args:
        F (function): Function for which the root is sought.
        J (function): Function returning the Jacobian of F.
        H (function): Function returning the Hessian of F.
        x0 (np.ndarray): First initial point.
        F_x0 (np.ndarray, optional): Value evaluated at initial
         point.
        J_x0 (np.ndarray, optional): Jacobian at initial point.
        H_x0 (np.ndarray, optional): Hessian at initial point.
        formula (int): Formula 1 or 2. Defaults to {formula}.
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

    x0 = np.asarray(x0).astype(np.float64)

    F_wrapper = PyDoubleNpNdArrayFPtr(F)
    J_wrapper = PyDoubleNpNdArrayFPtr(J)
    H_wrapper = PyDoubleNpNdArrayFPtr(H)
    if F_x0 is None:
        F_x0 = F_wrapper(x0)
    if J_x0 is None:
        J_x0 = J_wrapper(x0)
    if H_x0 is None:
        H_x0 = H_wrapper(x0)

    res = generalized_tangent_hyperbolas_kernel[DoubleNpNdArrayFPtr](
        F_wrapper, J_wrapper, H_wrapper, x0, F_x0, J_x0, H_x0, formula, etol, ertol, ptol, prtol, max_iter)
    return NewtonMethodReturnType.from_results(res, (F_wrapper.n_f_calls,
                                                     J_wrapper.n_f_calls,
                                                     H_wrapper.n_f_calls))
