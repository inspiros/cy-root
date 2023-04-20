# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from typing import Sequence, Union

import math
import numpy as np
from libc cimport math

from .ops cimport scalar_ops as sops, vector_ops as vops

__all__ = [
    '_check_stop_cond_args',
    '_check_initial_guesses_uniqueness',
    '_check_initial_vals_uniqueness',
]

################################################################################
# Python-side checks
################################################################################
# noinspection DuplicatedCode
def _check_stop_cond_arg(tol: float, arg_name='tol'):
    if tol is None:
        return 0
    elif math.isnan(tol) or not math.isfinite(tol) or tol < 0:
        raise ValueError(f'{arg_name} must be non-negative finite number. Got {tol}.')
    return tol

# noinspection DuplicatedCode
def _check_stop_cond_args(etol: float,
                          ertol: float,
                          ptol: float,
                          prtol: float,
                          max_iter: int):
    """Check tolerances and max_iter."""
    etol = _check_stop_cond_arg(etol, 'etol')
    ertol = _check_stop_cond_arg(ertol, 'ertol')
    ptol = _check_stop_cond_arg(ptol, 'ptol')
    prtol = _check_stop_cond_arg(prtol, 'prtol')

    if max_iter is None or max_iter < 0 or math.isinf(max_iter) or math.isnan(max_iter):
        max_iter = 0
    elif not isinstance(max_iter, int):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')

    if etol == ertol == ptol == prtol == max_iter == 0:
        raise ValueError(f'Disabling both tolerances and max_iter will '
                         f'likely cause the algorithm to run indefinitely.')
    return etol, ertol, ptol, prtol, max_iter

# noinspection DuplicatedCode
def _check_initial_guesses_uniqueness(x0s: Union[Sequence[Union[float, complex, np.ndarray]], np.ndarray]):
    if not len(x0s):
        raise ValueError('Empty.')
    elif isinstance(x0s[0], np.ndarray):
        if np.unique(x0s if isinstance(x0s, np.ndarray) else
                     np.stack(x0s), axis=0).shape[0] < len(x0s):
            raise ValueError(f'Initial guesses must be unique. Got:\n' +
                             '\n'.join(repr(_) for _ in x0s))
    elif len(set(x0s)) < len(x0s):
        raise ValueError(f'Initial guesses must be unique. Got {x0s}.')

# noinspection DuplicatedCode
def _check_initial_vals_uniqueness(f_x0s: Union[Sequence[Union[float, complex, np.ndarray]], np.ndarray]):
    if not len(f_x0s):
        raise ValueError('Empty.')
    elif isinstance(f_x0s[0], np.ndarray):
        if np.unique(f_x0s if isinstance(f_x0s, np.ndarray) else
                     np.stack(f_x0s), axis=0).shape[0] < len(f_x0s):
            raise ValueError('Initial guesses\' values must be unique. '
                             'Got:\n' + '\n'.join(repr(_) for _ in f_x0s))
    elif len(set(f_x0s)) < len(f_x0s):
        raise ValueError('Initial guesses\' values must be unique. '
                         f'Got {f_x0s}.')

################################################################################
# Bracketing methods
################################################################################
# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_scalar_bracket(
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
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = sops.isclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or sops.isclose(0, precision[0], prtol, ptol)

# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_vector_bracket(
        double[:, :] bs,
        double[:, :] F_bs,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double[:] r,
        double[:] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    if bs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    precision[0] = vops.max(np.max(bs, 0) - np.min(bs, 0))
    cdef double[:] errors = np.abs(F_bs).max(1)
    cdef unsigned long best_i = vops.argmin(errors)
    error[0] = errors[best_i]
    r[:], F_r[:] = bs[best_i], F_bs[best_i]
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = sops.isclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or sops.isclose(0, precision[0], prtol, ptol)

################################################################################
# Quasi-Newton methods with multiple guesses
################################################################################
# --------------------------------
# Double
# --------------------------------
# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_scalar_initial_guess(
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
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_vector_initial_guess(
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
    error[0] = vops.max(vops.fabs(F_x0))
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_scalar_initial_guesses(
        double[:] x0s,
        double[:] f_x0s,
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
    if x0s.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if x0s.shape[0] == 1:
        r[0], f_r[0] = x0s[0], f_x0s[0]
        precision[0] = math.INFINITY
        error[0] = math.fabs(f_x0s[0])
        optimal[0] = sops.isclose(0, error[0], ertol, etol)
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = vops.fabs(f_x0s)
    cdef unsigned long best_i = vops.argmin(errors)
    r[0], f_r[0] = x0s[best_i], f_x0s[best_i]
    error[0] = errors[best_i]
    precision[0] = vops.max(x0s) - vops.min(x0s)
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = sops.isclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or sops.isclose(0, precision[0], prtol, ptol)

# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_vector_initial_guesses(
        double[:, :] x0s,
        double[:, :] F_x0s,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double[:] r,
        double[:] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    if x0s.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if x0s.shape[0] == 1:
        r[:], F_r[:] = x0s[0], F_x0s[0]
        precision[0] = math.INFINITY
        error[0] = vops.max(vops.fabs(F_x0s[0]))
        optimal[0] = sops.isclose(0, error[0], ertol, etol)
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = np.abs(F_x0s).max(1)
    cdef unsigned long best_i = vops.argmin(errors)
    r[:], F_r[:] = x0s[best_i], F_x0s[best_i]
    error[0] = errors[best_i]
    precision[0] = vops.max(np.max(x0s, 0) - np.min(x0s, 0))
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = sops.isclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or sops.isclose(0, precision[0], prtol, ptol)

# --------------------------------
# Double Complex
# --------------------------------
# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_complex_scalar_initial_guess(
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
    error[0] = sops.cabs(f_x0)
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_complex_vector_initial_guess(
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
    error[0] = vops.max(vops.cabs(F_x0))
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_complex_scalar_initial_guesses(
        double complex[:] x0s,
        double complex[:] f_x0s,
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
    if x0s.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if x0s.shape[0] == 1:
        r[0], f_r[0] = x0s[0], f_x0s[0]
        precision[0] = math.INFINITY
        error[0] = sops.cabs(f_x0s[0])
        optimal[0] = sops.isclose(0, error[0], ertol, etol)
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = vops.cabs(f_x0s)
    cdef unsigned long best_i = vops.argmin(errors)
    r[0], f_r[0] = x0s[best_i], f_x0s[best_i]
    error[0] = errors[best_i]
    cdef double[:] xs_abs = vops.cabs(x0s)
    precision[0] = vops.max(xs_abs) - vops.min(xs_abs)
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = sops.isclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or sops.isclose(0, precision[0], prtol, ptol)

# noinspection DuplicatedCode
cdef inline bint _check_stop_cond_complex_vector_initial_guesses(
        double complex[:, :] x0s,
        double complex[:, :] F_x0s,
        double etol,
        double ertol,
        double ptol,
        double prtol,
        double complex[:] r,
        double complex[:] F_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    if x0s.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if x0s.shape[0] == 1:
        r[:], F_r[:] = x0s[0], F_x0s[0]
        precision[0] = math.INFINITY
        error[0] = vops.max(vops.cabs(F_x0s[0]))
        optimal[0] = sops.isclose(0, error[0], ertol, etol)
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = np.abs(F_x0s).max(1)
    cdef unsigned long best_i = vops.argmin(errors)
    r[:], F_r[:] = x0s[best_i], F_x0s[best_i]
    error[0] = errors[best_i]
    cdef double[:, :] xs_abs = np.abs(x0s)
    precision[0] = vops.max(np.max(xs_abs, 0) - np.min(xs_abs, 0))
    optimal[0] = sops.isclose(0, error[0], ertol, etol)
    converged[0] = sops.isclose(0, precision[0], prtol, ptol) or optimal[0]
    return optimal[0] or sops.isclose(0, precision[0], prtol, ptol)
