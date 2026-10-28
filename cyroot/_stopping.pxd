"""Native stopping primitives shared by scalar and vector solver kernels."""
cimport cython
from libc.math cimport fabs, fmax, hypot, isfinite, isnan, INFINITY


cdef enum StopReason:
    RUNNING = 0
    RESIDUAL_TOLERANCE = 1
    BRACKET_TOLERANCE = 2
    STAGNATION = 3
    MAX_ITERATIONS = 4
    ZERO_DERIVATIVE = 5
    BREAKDOWN = 6
    NONFINITE = 7
    MAX_SPLIT_ITERATIONS = 8
    COMPLETED = 9
    UNSPECIFIED = 10
    CALLBACK_FAILURE = 11


cdef struct StopState:
    double f_scale
    double x_scale
    double previous_x_norm
    double etol
    double ertol
    double ptol
    double prtol
    StopReason reason


cdef inline void init_stop(StopState* state, double f_scale, double x_norm,
                           double etol, double ertol, double ptol, double prtol) noexcept nogil:
    state.f_scale = f_scale
    state.x_scale = x_norm
    state.previous_x_norm = x_norm
    state.etol = etol
    state.ertol = ertol
    state.ptol = ptol
    state.prtol = prtol
    state.reason = RUNNING


cdef inline void advance_x(StopState* state, double x_norm) noexcept nogil:
    if not isfinite(x_norm) or not isfinite(state.previous_x_norm):
        state.x_scale = INFINITY
    else:
        state.x_scale = fmax(x_norm, state.previous_x_norm)
    state.previous_x_norm = x_norm


cdef inline bint within_tolerance(double metric, double atol, double rtol,
                                  double reference) noexcept nogil:
    # Compare without overflowing atol + rtol * reference.
    if not isfinite(metric) or metric < 0 or not isfinite(reference) or reference < 0:
        return False
    return metric <= atol or metric - atol <= rtol * reference


cdef inline bint residual_met(StopState* state, double error) noexcept nogil:
    return within_tolerance(error, state.etol, state.ertol, state.f_scale)


cdef inline bint check_stop(StopState* state, double error, double precision,
                            bint has_step, bint bracket=False) noexcept nogil:
    if state.reason != RUNNING:
        return True
    if (not isfinite(error) or error < 0 or not isfinite(state.f_scale) or
            not isfinite(state.x_scale) or isnan(precision) or precision < 0 or
            (has_step and not isfinite(precision))):
        state.reason = NONFINITE
    elif residual_met(state, error):
        state.reason = RESIDUAL_TOLERANCE
    elif isfinite(precision) and within_tolerance(precision, state.ptol, state.prtol, state.x_scale):
        state.reason = BRACKET_TOLERANCE if bracket else STAGNATION
    return state.reason != RUNNING


cdef inline void finish_stop(StopState* state, double error, double precision,
                             bint has_step, bint bracket=False) noexcept nogil:
    if not check_stop(state, error, precision, has_step, bracket):
        state.reason = BREAKDOWN


cdef inline bint successful(StopReason reason) noexcept nogil:
    return reason == RESIDUAL_TOLERANCE or reason == BRACKET_TOLERANCE


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm_inf(const double[:] values) noexcept nogil:
    cdef Py_ssize_t i
    cdef double result = 0., value
    if values.shape[0] == 0:
        return INFINITY
    for i in range(values.shape[0]):
        value = values[i]
        if not isfinite(value):
            return INFINITY
        result = fmax(result, fabs(value))
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double norm_inf2(const double[:, :] values) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double result = 0., value
    if values.shape[0] == 0 or values.shape[1] == 0:
        return INFINITY
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not isfinite(value):
                return INFINITY
            result = fmax(result, fabs(value))
    return result


cdef inline double complex_abs(double complex value) noexcept nogil:
    if not isfinite(value.real) or not isfinite(value.imag):
        return INFINITY
    return hypot(value.real, value.imag)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double complex_norm_inf(const double complex[:] values) noexcept nogil:
    cdef Py_ssize_t i
    cdef double result = 0., value
    if values.shape[0] == 0:
        return INFINITY
    for i in range(values.shape[0]):
        value = complex_abs(values[i])
        if not isfinite(value):
            return INFINITY
        result = fmax(result, value)
    return result
