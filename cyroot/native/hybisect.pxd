"""Native interval enclosure callback and bounded splitting bisection."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState, StopReason
from .scalar cimport ScalarCallback, ScalarRecord

ctypedef int (*IntervalFunction)(double a, double b, double* lower,
    double* upper, void* context) noexcept nogil
cdef extern from "interval_callbacks.h":
    ctypedef struct cyroot_interval_descriptor:
        size_t struct_size
        unsigned int abi_version
        IntervalFunction function
        void* context

cdef class IntervalCallback:
    cdef IntervalFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef bint _native, _active

cdef IntervalCallback make_interval_callback(IntervalFunction function,
    void* context, object keepalive=*)

cdef struct Interval:
    double a
    double b

cdef struct SplitRecord:
    ScalarRecord state
    size_t splits
    size_t total_iters
    size_t interval_calls
    size_t roots
    size_t pending
    int event
    int search_reason

cdef class Hybisect:
    cdef ScalarCallback _function
    cdef IntervalCallback _interval_function
    cdef StopState _stop
    cdef ScalarRecord _state
    cdef vector[Interval] _pending
    cdef vector[ScalarRecord] _roots
    cdef vector[SplitRecord] _records
    cdef size_t _pending_count, _root_count, _splits, _total_iters, _interval_calls
    cdef size_t _max_iter, _max_splits, _record_count, _history_every, _events
    cdef StopReason _search_reason
    cdef int _event, _callback_stage
    cdef bint _busy, _constructing, _initialized, _bracketing
    cdef bint _history_truncated, _workspace_exhausted
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, double point, double* out) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _finish_branch(self) noexcept nogil
    cdef void _process(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef SplitRecord _current(self) noexcept nogil
    cdef object _snapshot(self, SplitRecord record)
    cdef object _copy_result(self)
    cdef void _raise_callback_error(self) except *
