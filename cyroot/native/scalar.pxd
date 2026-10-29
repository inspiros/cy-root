"""Version 1 native scalar callback ABI for compiled Cython consumers."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState
from .scalar_derivatives cimport ScalarDerivativePlan

ctypedef int (*ScalarFunction)(double x, double* out, void* context) noexcept nogil

cdef extern from "callbacks.h":
    ctypedef struct cyroot_scalar_descriptor:
        size_t struct_size
        unsigned int abi_version
        ScalarFunction function
        void* context

cdef struct ScalarRecord:
    double root
    double f_root
    double a
    double b
    double f_a
    double f_b
    double precision
    double error
    size_t iters
    size_t f_calls
    int reason
    int callback_code

cdef class ScalarCallback:
    cdef ScalarFunction _function
    cdef void* _context
    cdef object _keepalive
    cdef object _python_function
    cdef object _error
    cdef bint _active
    cdef bint _native

cdef ScalarCallback make_scalar_callback(ScalarFunction function, void* context, object keepalive=*)

cdef class Bisection:
    cdef ScalarCallback _callback
    cdef StopState _stop
    cdef ScalarRecord _state
    cdef size_t _max_iter
    cdef vector[ScalarRecord] _records
    cdef size_t _record_count
    cdef size_t _history_every
    cdef bint _history_truncated
    cdef bint _busy
    cdef bint _initialized
    cdef bint _constructing
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, double x, double* out) noexcept nogil
    cdef void _initialize(self, bint cached_a, bint cached_b) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef object _snapshot(self, ScalarRecord record)
    cdef void _raise_callback_error(self) except *
    cdef object _copy_result(self)

cdef struct NewtonRecord:
    double root
    double f_root
    double df_root
    double precision
    double error
    size_t iters
    size_t f_calls
    size_t df_calls
    int reason
    int callback_code
    int callback_stage

cdef class Newton:
    cdef ScalarCallback _function
    cdef ScalarCallback _derivative
    cdef ScalarDerivativePlan _plan
    cdef StopState _stop
    cdef NewtonRecord _state
    cdef size_t _max_iter
    cdef vector[NewtonRecord] _records
    cdef size_t _record_count
    cdef size_t _history_every
    cdef bint _history_truncated
    cdef bint _busy
    cdef bint _initialized
    cdef bint _constructing
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, bint derivative, double x, double* out) noexcept nogil
    cdef void _initialize(self, bint cached_f, bint cached_df) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef object _snapshot(self, NewtonRecord record)
    cdef void _raise_callback_error(self) except *
    cdef object _copy_result(self)
