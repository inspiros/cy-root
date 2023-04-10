# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Union

import math
import numpy as np
from libc cimport math

from .ops cimport scalar_ops
from .ops cimport vector_ops

__all__ = [
    '_check_stop_condition_args',
    '_check_bracket',
    '_check_bracket_val',
    '_check_unique_initial_guesses',
    '_check_unique_initial_vals',
]

################################################################################
# Python
################################################################################
# noinspection DuplicatedCode
def _check_stop_condition_arg(tol: float, arg_name='tol'):
    if tol is None:
        return 0
    elif math.isnan(tol) or not math.isfinite(tol) or tol < 0:
        raise ValueError(f'{arg_name} must be non-negative finite number. Got {tol}.')
    return tol

# noinspection DuplicatedCode
def _check_stop_condition_args(etol: float,
                               ertol: float,
                               ptol: float,
                               prtol: float,
                               max_iter: int):
    """Check tolerances and max_iter."""
    etol = _check_stop_condition_arg(etol, 'etol')
    ertol = _check_stop_condition_arg(ertol, 'ertol')
    ptol = _check_stop_condition_arg(ptol, 'ptol')
    prtol = _check_stop_condition_arg(prtol, 'prtol')

    if max_iter is None or max_iter < 0 or math.isinf(max_iter) or math.isnan(max_iter):
        max_iter = 0
    elif not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if etol == ertol == ptol == prtol == max_iter == 0:
        raise ValueError(f'Disabling both tolerances and max_iter will '
                         f'likely cause the algorithm to run indefinitely.')
    return etol, ertol, ptol, prtol, max_iter

# noinspection DuplicatedCode
def _check_bracket(a: float, b: float, check_nan=True):
    if check_nan and (math.isnan(a) or math.isnan(b)):
        raise ValueError(f'nan value encountered a={a}, b={b}.')
    if a > b:
        raise ValueError(f'Expect a<b. Got a={a}, b={b}.')

# noinspection DuplicatedCode
def _check_bracket_val(f_a: float, f_b: float, check_nan=True):
    if check_nan and (math.isnan(f_a) or math.isnan(f_b)):
        raise ValueError(f'nan value encountered a={f_a}, b={f_b}.')
    if math.copysign(1, f_a) == math.copysign(1, f_b):
        raise ValueError('f_a and f_b must have opposite sign. '
                         f'Got f_a={f_a} and f_b={f_b}.')

# noinspection DuplicatedCode
def _check_unique_initial_guesses(*xs: Union[float, complex, np.ndarray]):
    if not len(xs):
        pass
    elif isinstance(xs[0], np.ndarray):
        if np.unique(np.stack(xs), axis=0).shape[0] < len(xs):
            raise ValueError(f'Initial guesses must be different. Got:\n' +
                             '\n'.join(repr(_) for _ in xs))
    elif len(set(xs)) < len(xs):
        raise ValueError(f'Initial guesses must be different. Got {xs}.')

# noinspection DuplicatedCode
def _check_unique_initial_vals(*f_xs: Union[float, complex, np.ndarray]):
    if not len(f_xs):
        pass
    elif isinstance(f_xs[0], np.ndarray):
        if np.unique(np.stack(f_xs), axis=0).shape[0] < len(f_xs):
            raise ValueError(f'Initial guesses must be different. Got:\n' +
                             '\n'.join(repr(_) for _ in f_xs))
    elif len(set(f_xs)) < len(f_xs):
        raise ValueError(f'Values evaluated at initial guesses must be '
                         f'different. Got {f_xs}.')

################################################################################
# Bracketing methods
################################################################################
# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_bracket_scalar(
        double a,
        double b,
        double f_a,
        double f_b,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* r,
        double* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    precision[0] = math.fabs(b - a)
    cdef double error_a = math.fabs(f_a), error_b = math.fabs(f_b)
    error[0] = math.fmin(error_a, error_b)
    r[0], f_r[0] = (a, f_a) if error_a < error_b else (b, f_b)
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = scalar_ops.fisclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or scalar_ops.fisclose(0, precision[0], prtol, ptol)

# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_bracket_vector(
        np.ndarray[np.float64_t, ndim=2] bs,
        np.ndarray[np.float64_t, ndim=2] F_bs,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        np.ndarray[np.float64_t, ndim=1] r,
        np.ndarray[np.float64_t, ndim=1] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    if bs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    precision[0] = np.max(bs.max(0) - bs.min(0))
    cdef double[:] errors = np.abs(F_bs).max(1)
    cdef unsigned long best_i = vector_ops.fargmin(errors)
    error[0] = errors[best_i]
    r[:], F_r[:] = bs[best_i], F_bs[best_i]
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = scalar_ops.fisclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or scalar_ops.fisclose(0, precision[0], prtol, ptol)

################################################################################
# Quasi-Newton methods with multiple guesses
################################################################################
# --------------------------------
# Double
# --------------------------------
# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_initial_guess_scalar(
        double x0,
        double f_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    precision[0] = math.INFINITY
    error[0] = math.fabs(f_x0)
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_initial_guess_vector(
        double[:] x0,
        double[:] F_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    precision[0] = math.INFINITY
    error[0] = vector_ops.fmax(vector_ops.fabs(F_x0))
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_initial_guesses_scalar(
        double[:] xs,
        double[:] f_xs,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* r,
        double* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if xs.shape[0] == 1:
        r[0], f_r[0] = xs[0], f_xs[0]
        precision[0] = math.INFINITY
        error[0] = math.fabs(f_xs[0])
        optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = vector_ops.fabs(f_xs)
    cdef unsigned long best_i = vector_ops.fargmin(errors)
    r[0], f_r[0] = xs[best_i], f_xs[best_i]
    error[0] = errors[best_i]
    precision[0] = vector_ops.fmax(xs) - vector_ops.fmin(xs)
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = scalar_ops.fisclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or scalar_ops.fisclose(0, precision[0], prtol, ptol)

# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_initial_guesses_vector(
        np.ndarray[np.float64_t, ndim=2] xs,
        np.ndarray[np.float64_t, ndim=2] F_xs,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        np.ndarray[np.float64_t, ndim=1] r,
        np.ndarray[np.float64_t, ndim=1] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if xs.shape[0] == 1:
        r[0], F_r[0] = xs[0], F_xs[0]
        precision[0] = math.INFINITY
        error[0] = np.max(np.abs(F_xs[0]))
        optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
        converged[0] = optimal[0]
        return optimal[0]
    cdef np.ndarray[np.float64_t, ndim=1] errors = np.abs(F_xs).max(1)
    cdef unsigned long best_i = np.argmin(errors)
    r[:], F_r[:] = xs[best_i], F_xs[best_i]
    error[0] = errors[best_i]
    precision[0] = np.max(xs.max(0) - xs.min(0))
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = scalar_ops.fisclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or scalar_ops.fisclose(0, precision[0], prtol, ptol)

# --------------------------------
# Double Complex
# --------------------------------
# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_initial_guess_complex_scalar(
        double complex x0,
        double complex f_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    precision[0] = math.INFINITY
    error[0] = scalar_ops.cabs(f_x0)
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_initial_guess_complex_vector(
        double complex[:] x0,
        double complex[:] F_x0,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    precision[0] = math.INFINITY
    error[0] = vector_ops.fmax(vector_ops.cabs(F_x0))
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_condition_initial_guesses_complex_scalar(
        double complex[:] xs,
        double complex[:] f_xs,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double complex* r,
        double complex* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if xs.shape[0] == 1:
        r[0], f_r[0] = xs[0], f_xs[0]
        precision[0] = math.INFINITY
        error[0] = scalar_ops.cabs(f_xs[0])
        optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = vector_ops.cabs(f_xs)
    cdef unsigned long best_i = vector_ops.fargmin(errors)
    r[0], f_r[0] = xs[best_i], f_xs[best_i]
    error[0] = errors[best_i]
    cdef double[:] xs_abs = vector_ops.cabs(xs)
    precision[0] = vector_ops.fmax(xs_abs) - vector_ops.fmin(xs_abs)
    optimal[0] = scalar_ops.fisclose(0, error[0], ertol, etol)
    converged[0] = scalar_ops.fisclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or scalar_ops.fisclose(0, precision[0], prtol, ptol)
