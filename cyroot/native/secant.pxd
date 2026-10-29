"""Draft native resumable secant solver; scalar callback ABI is shared."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState
from .scalar cimport ScalarCallback


cdef struct SecantRecord:
    double root
    double f_root
    double previous_root
    double previous_f_root
    double precision
    double error
    size_t iters
    size_t f_calls
    int reason
    int callback_code


cdef class Secant:
    cdef ScalarCallback _callback
    cdef StopState _stop
    cdef SecantRecord _state
    cdef size_t _max_iter, _record_count, _history_every
    cdef vector[SecantRecord] _records
    cdef bint _initialized, _constructing, _busy, _history_truncated
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, double x, double* out) noexcept nogil
    cdef void _initialize(self, double x1, double f_x1,
                          bint cached_f0, bint cached_f1) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, SecantRecord record)
    cdef object _copy_result(self)
