from libc.stddef cimport size_t
from libcpp.vector cimport vector
from .buffers cimport OwnedDoubleBuffer
from .vector cimport VectorCallback
from .derivatives cimport JacobianPlan
from cyroot._stopping cimport StopState

cdef struct BroydenRecord:
    double precision
    double error
    size_t iters
    size_t f_calls
    size_t approximation_iteration
    size_t initialization_calls
    bint approximation_valid
    int callback_stage
    int reason
    int callback_code
    int phase
    int linear_status

cdef class BroydenGood:
    cdef VectorCallback _function
    cdef JacobianPlan _plan
    cdef StopState _stop
    cdef BroydenRecord _state
    cdef size_t _n, _square, _stride, _max_iter, _history_every, _record_count
    cdef size_t _direction_count
    cdef OwnedDoubleBuffer* _storage
    cdef OwnedDoubleBuffer* _history_data
    cdef vector[int] _pivots
    cdef vector[BroydenRecord] _records
    cdef double* _x
    cdef double* _f
    cdef double* _approx
    cdef double* _trial
    cdef double* _trial_f
    cdef double* _delta
    cdef double* _change
    cdef double* _lu
    cdef double* _next
    cdef double* _diff_work
    cdef double* _directions
    cdef double* _basis
    cdef double* _projected
    cdef bint _inverse, _klement, _barnes, _from_jacobian, _busy, _constructing, _initialized, _history_truncated
    cdef void _setup(self, object function, object approximation, object x0,
                     object f_x0, object etol, object ertol, object ptol,
                     object prtol, object max_iter, object history,
                     object history_every, object history_bytes,
                     object workspace_bytes, bint from_jacobian,
                     object algo, bint dispatch) except *
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, const double* point, double* out) noexcept nogil
    cdef void _initialize(self, bint cached) noexcept nogil
    cdef void _initial_approximation(self) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, BroydenRecord record, const double* data)

cdef class BroydenBad(BroydenGood):
    pass

cdef class Klement(BroydenGood):
    pass

cdef class Barnes(BroydenGood):
    pass
