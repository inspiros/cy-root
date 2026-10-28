"""Native generalized bisection with bounded copied polygon state."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from .buffers cimport OwnedDoubleBuffer
from .vector cimport VectorCallback
from .rectangular cimport RectangularCallback
from cyroot._stopping cimport StopState

cdef struct VrahatisRecord:
    double precision
    double error
    size_t iters
    size_t f_calls
    size_t count
    int reason
    int callback_code

cdef class Vrahatis:
    cdef VectorCallback _function
    cdef RectangularCallback _rect_function
    cdef OwnedDoubleBuffer *_storage, *_history_data
    cdef double *_x, *_f, *_points, *_values, *_source_points, *_source_values
    cdef double *_trial, *_trial_f, *_h, *_polygon_work
    cdef vector[VrahatisRecord] _records
    cdef vector[unsigned char] _seen
    cdef StopState _stop
    cdef VrahatisRecord _state
    cdef size_t _n, _m, _k, _capacity, _count, _source_count, _stride
    cdef size_t _record_count, _max_iter, _history_every, _max_evaluations, _max_visits, _max_bisections
    cdef double _polygon_eps
    cdef bint _busy, _constructing, _initialized, _single, _history_truncated, _invalid_polygon, _polygon_root
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _leave(self) noexcept
    cdef int _evaluate(self, const double* point, double* out) noexcept nogil
    cdef int _polygon_evaluate(self,const double* x,double* out) noexcept nogil
    cdef void _initialize(self, bint cached, bint discovered) noexcept nogil
    cdef void _canonicalize(self) noexcept nogil
    cdef void _best(self) noexcept nogil
    cdef double _geometry(self) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, VrahatisRecord record, const double* data)
    cdef object _copy_result(self)
