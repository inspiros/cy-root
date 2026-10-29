"""Native scalar interpolation state, workspace and bounded copied history."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState
from .scalar cimport ScalarCallback
from .complex_callbacks cimport ComplexScalarCallback

cdef struct InterpolationRecord:
    double complex root
    double complex f_root
    double precision
    double error
    size_t iters
    size_t f_calls
    int reason
    int callback_code

cdef class _Interpolation:
    cdef ScalarCallback _callback
    cdef ComplexScalarCallback _complex_callback
    cdef StopState _stop
    cdef InterpolationRecord _state
    cdef vector[double complex] _xs, _fs, _work
    cdef vector[unsigned char] _cached
    cdef vector[InterpolationRecord] _records
    cdef size_t _max_iter, _record_count, _history_every
    cdef int _method
    cdef bint _aitken, _initialized, _constructing, _busy, _history_truncated
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, double complex x, double complex* out) noexcept nogil
    cdef void _initialize(self) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef bint _candidate(self, double complex* out) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, InterpolationRecord state)
    cdef object _copy_result(self)
