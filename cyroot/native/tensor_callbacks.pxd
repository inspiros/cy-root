"""Draft square Hessian callback API, separate from vector residual/Jacobian ABI."""
from libc.stddef cimport size_t

ctypedef int (*HessianFunction)(const double* x, size_t n, double* out,
                               void* context) noexcept nogil

cdef extern from "callbacks.h":
    ctypedef struct cyroot_hessian_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t dimension
        HessianFunction function
        void* context

cdef class HessianCallback:
    cdef HessianFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef size_t _dimension
    cdef bint _active, _native, _constructing

cdef HessianCallback make_hessian_callback(HessianFunction function,
                                           size_t dimension, void* context,
                                           object keepalive=*)

# Shared GIL-held copied conversion, also used for cached solver Hessians.
cdef void _pack_tensor(object value, double* out, size_t n) except *


ctypedef int (*RectangularHessianFunction)(const double* x, size_t n, size_t m,
                                          double* out, void* context) noexcept nogil

cdef extern from "callbacks.h":
    ctypedef struct cyroot_rectangular_hessian_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t input_dimension
        size_t output_dimension
        RectangularHessianFunction function
        void* context

cdef class RectangularHessianCallback:
    cdef RectangularHessianFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef size_t _n, _m, _tensor_size
    cdef bint _active, _native, _constructing
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    # Trusted disjoint x[n],out[m*n*n]; caller reserves and pins lifetime.
    # Output starts NaN; nonzero status clears entire output; caller validates
    # numerical finiteness on code0. Python bridge stores original _error.
    cdef int evaluate_raw(self, const double* x, double* out) noexcept nogil

cdef RectangularHessianCallback make_rectangular_hessian_callback(
    RectangularHessianFunction function, size_t n, size_t m, void* context,
    object keepalive=*)

cdef void _pack_rectangular_tensor(object value, double* out, size_t n, size_t m) except *
