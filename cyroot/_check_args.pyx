# distutils: language=c++
# cython: cdivision = True
# cython: initializedcheck = False
# cython: boundscheck = False
# cython: profile = False

from libc cimport math

from .utils.array_ops cimport fabs, cabs, argmin

cdef extern from '<complex>':
    double abs(double complex) nogil

__all__ = [
    '_check_stopping_condition_args',
    '_check_bracket',
    '_check_bracket_val',
]

################################################################################
# Python
################################################################################
def _check_stopping_condition_args(etol: float, ptol: float, max_iter: int):
    if etol < 0:
        raise ValueError(f'etol must be positive. Got {etol}.')
    if ptol < 0:
        raise ValueError(f'ptol must be positive. Got {ptol}.')
    if not isinstance(max_iter, int) or max_iter == float('inf'):
        raise ValueError(f'max_iter must be an integer. Got {max_iter}.')
    if etol == ptol == 0 and max_iter <= 0:
        raise ValueError(f'Disabling both etol, ptol, and max_iter will '
                         f'likely cause the algorithm to run indefinitely.')

def _check_bracket(a: float, b: float, check_nan=True):
    if check_nan and (math.isnan(a) or math.isnan(b)):
        raise ValueError(f'nan value encountered a={a}, b={b}.')
    if a > b:
        raise ValueError(f'Expect a<b. Got a={a}, b={b}.')

def _check_bracket_val(f_a: float, f_b: float, check_nan=True):
    if check_nan and (math.isnan(f_a) or math.isnan(f_b)):
        raise ValueError(f'nan value encountered a={f_a}, b={f_b}.')
    if math.copysign(1, f_a) == math.copysign(1, f_b):
        raise ValueError('f_a and f_b must have opposite sign. '
                         f'Got f_a={f_a} and f_b={f_b}.')

################################################################################
# Bracketing methods
################################################################################
# noinspection DuplicatedCode
cdef inline bint _check_initial_bracket(
        double a,
        double b,
        double f_a,
        double f_b,
        double etol,
        double ptol,
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
    optimal[0] = error[0] <= etol
    converged[0] = precision[0] <= ptol or optimal[0]
    return (math.copysign(1, f_a) * math.copysign(1, f_b) > 0
            and not optimal[0]) or precision[0] <= ptol

################################################################################
# Quasi-Newton methods with multiple guesses
################################################################################
# --------------------------------
# Double
# --------------------------------
# noinspection DuplicatedCode
cdef inline bint _check_initial_guess(
        double x0,
        double f_x0,
        double etol,
        double ptol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    precision[0] = math.INFINITY
    error[0] = math.fabs(f_x0)
    optimal[0] = error[0] <= etol
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_initial_guesses(
        double[:] xs,
        double[:] f_xs,
        double etol,
        double ptol,
        double* r,
        double* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal,
        precision_func_type precision_func=NULL):
    """Check if stop condition is already met."""
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if xs.shape[0] == 1:
        r[0], f_r[0] = xs[0], f_xs[0]
        precision[0] = math.INFINITY
        error[0] = math.fabs(f_xs[0])
        optimal[0] = error[0] <= etol
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = fabs(f_xs)
    cdef long best_i = argmin(errors)
    r[0], f_r[0] = xs[best_i], f_xs[best_i]
    error[0] = errors[best_i]
    precision[0] = precision_func(xs) if precision_func is not NULL else math.INFINITY
    optimal[0] = error[0] <= etol
    converged[0] = precision[0] <= ptol or optimal[0]
    return optimal[0] or precision[0] <= ptol

# --------------------------------
# Double Complex
# --------------------------------
# noinspection DuplicatedCode
cdef inline bint _check_initial_guess_complex(
        double complex x0,
        double complex f_x0,
        double etol,
        double ptol,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal):
    """Check if stop condition is already met."""
    precision[0] = math.INFINITY
    error[0] = abs(f_x0)
    optimal[0] = error[0] <= etol
    converged[0] = optimal[0]
    return optimal[0]

# noinspection DuplicatedCode
cdef inline bint _check_initial_guesses_complex(
        double complex[:] xs,
        double complex[:] f_xs,
        double etol,
        double ptol,
        double complex* r,
        double complex* f_r,
        double* precision,
        double* error,
        bint* converged,
        bint* optimal,
        precision_func_type_complex precision_func=NULL):
    """Check if stop condition is already met."""
    if xs.shape[0] == 0:
        raise ValueError('Empty sequence.')
    if xs.shape[0] == 1:
        r[0], f_r[0] = xs[0], f_xs[0]
        precision[0] = math.INFINITY
        error[0] = abs(f_xs[0])
        optimal[0] = error[0] <= etol
        converged[0] = optimal[0]
        return optimal[0]
    cdef double[:] errors = cabs(f_xs)
    cdef long best_i = argmin(errors)
    r[0], f_r[0] = xs[best_i], f_xs[best_i]
    error[0] = errors[best_i]
    precision[0] = precision_func(xs) if precision_func is not NULL else math.INFINITY
    optimal[0] = error[0] <= etol
    converged[0] = precision[0] <= ptol or optimal[0]
    return optimal[0] or precision[0] <= ptol
