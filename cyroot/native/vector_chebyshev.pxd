"""Shared square-vector higher-order state and preallocated work/history."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot._stopping cimport StopState
from .buffers cimport OwnedDoubleBuffer
from .derivatives cimport JacobianPlan, RectangularJacobianPlan, ResidualFunction
from .hessians cimport HessianPlan, HessianConfig, RectangularHessianPlan, RectangularHessianConfig
from .vector cimport VectorCallback, VectorFunction
from .tensor_callbacks cimport HessianCallback, HessianFunction, RectangularHessianCallback, RectangularHessianFunction
from .rectangular cimport RectangularCallback, RectangularFunction
from .least_squares cimport LeastSquaresPlan


cdef struct VectorChebyshevRecord:
    double precision
    double error
    size_t iters
    size_t f_calls
    size_t jac_calls
    size_t hess_calls
    int reason
    int callback_code
    int callback_stage
    int linear_status
    int linear_stage
    int linear_rank


cdef class VectorChebyshev:
    cdef VectorCallback _function, _jacobian
    cdef RectangularCallback _rect_function, _rect_jacobian
    cdef RectangularHessianCallback _rect_hessian
    cdef RectangularJacobianPlan _rect_jplan
    cdef RectangularHessianPlan _rect_hplan
    cdef LeastSquaresPlan _linear_plan
    cdef RectangularHessianConfig _rhconfig
    cdef RectangularFunction _rect_function_ptr, _rect_jacobian_ptr
    cdef RectangularHessianFunction _rect_hessian_ptr
    cdef double* _linear_work
    cdef HessianCallback _hessian
    cdef JacobianPlan _jplan
    cdef HessianPlan _hplan
    cdef HessianConfig _hconfig
    # Plain callback fields are copied once; numerical methods do not touch
    # Python owner tuples or construct temporary extension references.
    cdef VectorFunction _function_ptr, _jacobian_ptr
    cdef HessianFunction _hessian_ptr
    cdef void* _function_context
    cdef void* _jacobian_context
    cdef void* _hessian_context
    cdef StopState _stop
    cdef VectorChebyshevRecord _state
    cdef size_t _n, _m, _square, _cube, _diff_size, _record_stride
    cdef size_t _max_iter, _record_count, _history_every
    cdef int _method, _path
    cdef double _alpha
    cdef OwnedDoubleBuffer* _storage
    cdef OwnedDoubleBuffer* _history_data
    cdef double* _x
    cdef double* _f
    cdef double* _jac
    cdef double* _hess
    cdef double* _trial
    cdef double* _trial_f
    cdef double* _lu
    cdef double* _s
    cdef double* _q
    cdef double* _t
    cdef double* _delta
    cdef double* _diff_work
    # NULL for Chebyshev. Matrix methods place n*n doubles immediately after
    # _diff_work[_diff_size], leaving every existing segment unchanged.
    cdef double* _secondary
    # General modified-Halley alone retains A after the secondary matrix.
    cdef double* _directional
    cdef vector[int] _pivots
    cdef vector[int] _secondary_pivots
    cdef vector[VectorChebyshevRecord] _records
    cdef bint _initialized, _constructing, _busy, _history_truncated
    cdef void _setup(self, object function, object jacobian, object hessian,
                      object x0, object f_x0, object jac_x0, object hess_x0,
                      object etol, object ertol, object ptol, object prtol,
                      object max_iter, object history, object history_every,
                      object history_bytes, object workspace_bytes,
                      object alpha_option) except *
    cdef void _ready(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef int _evaluate(self, int stage, const double* x, double* out) noexcept nogil
    cdef void _derivatives(self, bint cached_jac, bint cached_hess) noexcept nogil
    cdef void _initialize(self, bint cached_f, bint cached_jac,
                           bint cached_hess) noexcept nogil
    cdef void _check(self) noexcept nogil
    cdef void _record(self) noexcept nogil
    cdef void _step(self) noexcept nogil
    cdef void _raise_callback_error(self) except *
    cdef object _snapshot(self, VectorChebyshevRecord record, const double* data)
    cdef object _copy_result(self)


cdef class VectorHalley(VectorChebyshev):
    pass


cdef class VectorSuperHalley(VectorChebyshev):
    pass


cdef class VectorModifiedHalley(VectorChebyshev):
    pass
