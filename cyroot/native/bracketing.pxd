"""Shared native bracket state; callbacks use the scalar descriptor ABI."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState
from .scalar cimport ScalarCallback, ScalarRecord

cdef class BracketSolver:
    cdef ScalarCallback _callback
    cdef StopState _stop
    cdef ScalarRecord _state
    cdef size_t _max_iter, _record_count, _history_every, _k, _n0
    cdef vector[ScalarRecord] _records
    cdef bint _initialized, _constructing, _busy, _history_truncated, _default_k1
    cdef int _method, _interp, _last_method
    cdef double _sigma, _mu, _k1, _k2, _rho, _weight, _t, _initial_width
    cdef double _p, _q, _r, _fp, _fq, _fr, _d, _fd, _e, _fe
    cdef double _itp_tolerance, _itp_exponent
    cdef double _last_x, _last_f
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _leave(self) noexcept
    cdef int _evaluate(self, double x, double* out) noexcept nogil
    cdef void _initialize(self, bint cached_a, bint cached_b) noexcept nogil
    cdef void _best(self) noexcept nogil
    cdef void _swap(self) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef bint _trial(self, double x) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, ScalarRecord record)
    cdef object _copy_result(self)
