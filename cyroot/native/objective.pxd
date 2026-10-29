"""Native exact least-squares objective and low-order derivative adapter."""
from libc.stddef cimport size_t
from .buffers cimport OwnedDoubleBuffer
from .scalar cimport ScalarCallback
from .vector cimport VectorCallback
from .rectangular cimport RectangularCallback
from .tensor_callbacks cimport HessianCallback, RectangularHessianCallback

cdef extern from "objective.hpp" namespace "cyroot::native" nogil:
    cdef cppclass ObjectiveAccumulator:
        ObjectiveAccumulator() noexcept
        void add_scaled(double mantissa, int exponent) noexcept
        void add_product(double a, double b) noexcept
        void add_triple(double a, double b, double c) noexcept
        int value(double* out) noexcept
    int objective_value(const double* residual, size_t m, double* out) noexcept
    int objective_gradient(const double* residual, const double* jacobian,
                           size_t n, size_t m, double* out) noexcept
    int objective_hessian(const double* residual, const double* jacobian,
                          const double* hessian, size_t n, size_t m,
                          double* out) noexcept

cdef class LeastSquaresObjective:
    cdef ScalarCallback _scalar_function, _scalar_jacobian, _scalar_hessian
    cdef bint _scalar
    cdef VectorCallback _function, _jacobian
    cdef RectangularCallback _rect_function, _rect_jacobian
    cdef HessianCallback _hessian
    cdef RectangularHessianCallback _rect_hessian
    cdef OwnedDoubleBuffer* _storage
    cdef double *_x, *_f, *_j, *_h, *_result
    cdef size_t _n, _m, _matrix, _tensor, _capacity, _memory_bytes
    cdef int _callback_code, _callback_stage
    cdef bint _initialized, _constructing, _busy
    cdef void _ready(self) except *
    # Reserve all attached callbacks and this state before any input coercion;
    # hold reservation through raw evaluation, exception and result exports.
    cdef void _reserve(self) except *
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    cdef void _raise_callback_error(self, object result) except *
    cdef int _sample(self, int stage, const double* x, double* out) noexcept nogil
    cdef object _evaluate(self, object point, int order)
    # Caller pins self/context lifetimes, holds _enter reservation, and passes
    # valid disjoint x[n],out[capacity] spans outside this state's storage.
    # order=-1 residual(m),0 value(1),1 gradient(n),2 Hessian(n*n, column-major).
    # Status0 ok,1 nonfinite,2 invalid/uninitialized/not reserved/missing
    # derivative provider,3 callback failure. No output writes on failure.
    # capacity is checked before sampling. No native heap allocation; Python
    # providers acquire GIL. _release must run with GIL even after failure.
    cdef int evaluate_raw(self, const double* x, int order, double* out,
                          size_t capacity) noexcept nogil
