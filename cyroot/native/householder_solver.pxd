"""Resumable runtime-order Householder and its preallocated native workspace."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState
from .scalar cimport ScalarFunction
from .higher_derivatives cimport ScalarDifferenceConfig
from .householder cimport ScaledCoefficient


cdef struct HouseholderRole:
    ScalarFunction function
    void* context
    ScalarDifferenceConfig difference
    bint numerical


cdef struct HouseholderRecord:
    double root
    double f_root
    double precision
    double error
    size_t iters
    int reason
    int callback_code
    int callback_stage
    size_t callback_order


cdef class Householder:
    # Owners are touched only under the GIL. Numerical methods use the POD
    # roles and never index an adapter/provider tuple or create Python refs.
    cdef tuple _adapters, _providers
    cdef StopState _stop
    cdef HouseholderRecord _state
    cdef size_t _order, _max_iter, _history_every, _record_count
    cdef bint _initialized, _constructing, _busy, _history_truncated
    # Allocation probes for external consumers: all vector storage is sized
    # once during construction, including flat history values and counters.
    cdef vector[double] _values, _difference_workspace, _history_derivatives
    cdef vector[size_t] _calls, _history_calls
    cdef vector[HouseholderRole] _roles
    cdef vector[unsigned char] _cached
    cdef vector[ScaledCoefficient] _recurrence
    cdef vector[HouseholderRecord] _records
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, size_t role, double x, double* out) noexcept nogil
    cdef void _clear_derivatives(self, size_t start) noexcept nogil
    cdef void _derivatives(self) noexcept nogil
    cdef void _initialize(self) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, HouseholderRecord record,
                          const double* derivatives, const size_t* calls)
    cdef object _copy_result(self)
