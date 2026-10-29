"""Draft native vector Newton and preserved square callback API."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from .buffers cimport OwnedDoubleBuffer
from .derivatives cimport JacobianPlan, RectangularJacobianPlan
from .rectangular cimport RectangularCallback
from .least_squares cimport LeastSquaresPlan
from cyroot._stopping cimport StopState

ctypedef int (*VectorFunction)(const double* x, size_t n, double* out, void* context) noexcept nogil

cdef extern from "callbacks.h":
    ctypedef struct cyroot_vector_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t dimension
        unsigned int kind
        VectorFunction function
        void* context

cdef class VectorCallback:
    cdef VectorFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef size_t _dimension
    cdef bint _jacobian, _active, _native, _constructing

cdef VectorCallback make_vector_callback(VectorFunction function, size_t dimension,
                                         void* context, object keepalive=*, bint jacobian=*)

cdef struct VectorRecord:
    double precision
    double error
    size_t iters
    size_t f_calls
    size_t jac_calls
    int reason
    int callback_code
    int callback_stage
    int linear_status
    int linear_rank

cdef class VectorNewton:
    cdef VectorCallback _function, _jacobian
    cdef JacobianPlan _plan
    cdef RectangularCallback _rect_function, _rect_jacobian
    cdef RectangularJacobianPlan _rect_plan
    cdef LeastSquaresPlan _linear_plan
    cdef StopState _stop
    cdef VectorRecord _state
    cdef size_t _n, _square, _record_stride, _max_iter, _record_count, _history_every
    cdef size_t _m
    cdef OwnedDoubleBuffer* _storage
    cdef OwnedDoubleBuffer* _history_data
    cdef double *_x, *_f, *_jac, *_trial, *_trial_f, *_lu, *_delta
    cdef double* _diff_work
    cdef double *_linear_work, *_rhs
    cdef double *_rob_previous, *_rob_previous_f, *_rob_nodes, *_rob_orient, *_rob_row
    cdef vector[int] _pivots
    cdef vector[VectorRecord] _records
    cdef bint _busy, _initialized, _constructing, _history_truncated, _steffensen, _robinson, _rob_identity
    cdef void _setup(self, object function, object jacobian, object x0,
        object f_x0, object jac_x0, object etol, object ertol, object ptol,
        object prtol, object max_iter, object history, object history_every,
        object history_bytes, object workspace_bytes, object previous,
        object previous_f, object orth) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef void _ready(self) except *
    cdef int _evaluate(self, bint jacobian, const double* x, double* out) noexcept nogil
    cdef int _residual(self, const double* x, double* out) noexcept nogil
    cdef void _initialize(self, bint cached_f, bint cached_jac) noexcept nogil
    cdef void _initialize_robinson(self, bint cached_previous, bint cached_f) noexcept nogil
    cdef int _robinson_jacobian(self, const double* x, double* out) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, VectorRecord record, const double* data)
    cdef object _copy_result(self)

cdef class TraubSteffensen(VectorNewton):
    pass

cdef class Robinson(VectorNewton):
    pass

cdef class WolfeBittner:
    cdef VectorCallback _function
    cdef RectangularCallback _rect_function
    cdef LeastSquaresPlan _linear_plan
    cdef StopState _stop
    cdef VectorRecord _state
    cdef size_t _n, _m, _k, _rows, _stride, _max_iter, _record_count, _history_every
    cdef OwnedDoubleBuffer* _storage
    cdef OwnedDoubleBuffer* _history_data
    cdef double *_points, *_values, *_x, *_f, *_active_x, *_active_f
    cdef double *_trial, *_trial_f, *_affine, *_rhs, *_weights, *_work
    cdef double _affine_error
    cdef vector[int] _pivots
    cdef vector[VectorRecord] _records
    cdef vector[double] _affine_records
    cdef bint _busy, _initialized, _constructing, _history_truncated
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, const double* point, double* out) noexcept nogil
    cdef double _geometry(self) noexcept nogil
    cdef void _initialize(self, bint cached, bint discovered) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, VectorRecord record, const double* data, double affine_error)
    cdef object _copy_result(self)
