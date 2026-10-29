"""Shared scalar Halley-family state and derivative contracts."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState
from .scalar cimport ScalarCallback
from .scalar_derivatives cimport ScalarDerivativePlan, ScalarSecondDerivativePlan


cdef struct HigherOrderRecord:
    double root
    double f_root
    double df_root
    double d2f_root
    double precision
    double error
    size_t iters
    size_t f_calls
    size_t df_calls
    size_t d2f_calls
    int reason
    int callback_code
    int callback_stage


cdef class _ThirdOrderNewton:
    cdef ScalarCallback _function, _derivative, _second_derivative
    cdef ScalarDerivativePlan _plan
    cdef ScalarSecondDerivativePlan _second_plan
    cdef StopState _stop
    cdef HigherOrderRecord _state
    cdef vector[HigherOrderRecord] _records
    cdef size_t _max_iter, _record_count, _history_every
    cdef int _method
    cdef double _alpha
    cdef bint _initialized, _constructing, _busy, _history_truncated
    cdef void _setup(self, object function, object derivative,
                      object second_derivative, object x0, object f_x0,
                      object df_x0, object d2f_x0, object etol, object ertol,
                      object ptol, object prtol, object max_iter, object history,
                      object history_every, object history_bytes, int method,
                      object alpha) except *
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, int stage, double x, double* out) noexcept nogil
    cdef void _initialize(self, bint cached_f, bint cached_df,
                           bint cached_d2f) noexcept nogil
    cdef void _derivatives(self, bint cached_df, bint cached_d2f) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, HigherOrderRecord record)
    cdef object _copy_result(self)


cdef class Halley(_ThirdOrderNewton):
    pass


cdef class Chebyshev(_ThirdOrderNewton):
    pass


cdef class ModifiedHalley(_ThirdOrderNewton):
    pass


cdef class SuperHalley(_ThirdOrderNewton):
    pass


cdef class TangentHyperbolas(Halley):
    pass
