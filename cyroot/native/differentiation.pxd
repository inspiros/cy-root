"""Native derivative providers using the ordinary scalar/Jacobian callback ABI.

Providers own scratch and private adapters; their standard callback reservation
guards concurrent/reentrant use. Native producer contexts must be thread-safe
if shared through different adapters, as with other native callbacks.
"""
from libc.stddef cimport size_t
from .buffers cimport OwnedDoubleBuffer
from .scalar cimport ScalarCallback
from .rectangular cimport RectangularCallback
from .complex_callbacks cimport ComplexScalarCallback
from .tensor_callbacks cimport RectangularHessianCallback

ctypedef int (*ComplexVectorFunction)(const double* real, const double* imag,
    size_t n, size_t m, double* out_real, double* out_imag, void* context) noexcept nogil

cdef extern from "complex_callbacks.h":
    ctypedef struct cyroot_complex_vector_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t input_dimension
        size_t output_dimension
        ComplexVectorFunction function
        void* context

cdef class ComplexVectorCallback:
    cdef ComplexVectorFunction _function
    cdef void* _context
    cdef size_t _n, _m
    cdef object _keepalive, _python_function, _error
    cdef bint _active, _native, _constructing
    cdef void _enter(self) except *
    cdef void _release(self) noexcept

cdef ComplexVectorCallback make_complex_vector_callback(ComplexVectorFunction function,
    size_t n, size_t m, void* context, object keepalive=*)

cdef extern from "differentiation.hpp" namespace "cyroot" nogil:
    ctypedef int (*DiffRealFunction)(const double*, size_t, size_t, double*, void*) noexcept
    ctypedef int (*DiffComplexFunction)(const double*, const double*, size_t,
                                       size_t, double*, double*, void*) noexcept
    cdef enum EstimateStatus:
        ESTIMATE_OK
        ESTIMATE_NONFINITE
        ESTIMATE_BREAKDOWN
        ESTIMATE_CALLBACK_FAILURE
        ESTIMATE_MAX_STEPS
        ESTIMATE_ROUNDOFF
    cdef cppclass EstimateReport:
        size_t calls
        int status
        int callback_code
    int complex_step_jacobian(const double*, size_t, size_t, const double*, double*,
        double*, DiffComplexFunction, void*, EstimateReport&) noexcept
    int adaptive_jacobian(const double*, size_t, size_t, const double*, double, double,
        size_t, double*, double*, double*, DiffRealFunction, void*, EstimateReport&) noexcept
    int adaptive_second_derivative(double, const double*, double, double, size_t,
        double*, double*, double*, DiffRealFunction, void*, EstimateReport&) noexcept
    int richardson_update(const double*, size_t, size_t, double*, double*, double*,
        double*, double*, int*) noexcept
    int diff_failure(EstimateReport&, int, double*, double*, size_t) noexcept

cdef class _ScalarDifferentiation(ScalarCallback):
    cdef ScalarCallback _real
    cdef ComplexScalarCallback _complex
    cdef OwnedDoubleBuffer* _storage
    cdef double _step, _atol, _rtol
    cdef size_t _levels
    cdef int _order
    cdef bint _adaptive, _building
    cdef EstimateReport _report
    cdef void _configure(self, object function, bint adaptive, object step,
                          object atol, object rtol, object max_steps, int order) except *
    cdef int _compute(self, double x) noexcept nogil
    cdef object _estimate(self)
    cdef void _check_ready(self) except *

cdef class ComplexStepDerivative(_ScalarDifferentiation):
    pass
cdef class AdaptiveDerivative(_ScalarDifferentiation):
    pass
cdef class AdaptiveSecondDerivative(_ScalarDifferentiation):
    pass

cdef class _JacobianDifferentiation(RectangularCallback):
    cdef RectangularCallback _real
    cdef ComplexVectorCallback _complex
    cdef OwnedDoubleBuffer* _storage
    cdef OwnedDoubleBuffer* _steps
    cdef double _atol, _rtol
    cdef size_t _levels, _memory_bytes
    cdef bint _adaptive
    cdef EstimateReport _report
    cdef void _configure(self, object function, object n, object m, bint adaptive,
        object step, object atol, object rtol, object max_steps, object memory_bytes) except *
    cdef int _compute(self, const double* x) noexcept nogil
    cdef object _estimate(self)

cdef class ComplexStepJacobian(_JacobianDifferentiation):
    pass
cdef class AdaptiveJacobian(_JacobianDifferentiation):
    pass

cdef class AdaptiveHessian(RectangularHessianCallback):
    cdef RectangularCallback _real
    cdef OwnedDoubleBuffer* _storage
    cdef OwnedDoubleBuffer* _steps
    cdef double _atol, _rtol
    cdef size_t _levels, _fixed_work, _memory_bytes
    cdef EstimateReport _report
    cdef int _compute(self, const double* x) noexcept nogil
    cdef object _estimate(self)
