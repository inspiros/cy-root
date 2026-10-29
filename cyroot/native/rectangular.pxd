"""Separate rectangular residual/Jacobian callback interface (draft v1)."""
from libc.stddef cimport size_t

ctypedef int (*RectangularFunction)(const double* x, size_t n, size_t m,
                                  double* out, void* context) noexcept nogil

cdef extern from "callbacks.h":
    ctypedef struct cyroot_rectangular_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t input_dimension
        size_t output_dimension
        unsigned int kind
        RectangularFunction function
        void* context

cdef class RectangularCallback:
    cdef RectangularFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef size_t _n, _m, _output_size
    cdef bint _jacobian, _active, _native, _constructing
    cdef void _enter(self) except *
    cdef void _release(self) noexcept
    # Caller reserves this adapter and pins its lifetime. Trusted disjoint
    # x[n], out[m or m*n] spans; input is borrowed and must not be mutated.
    # Native nonzero status poisons all output. Python bridge saves _error.
    cdef int evaluate_raw(self, const double* x, double* out) noexcept nogil

cdef RectangularCallback make_rectangular_callback(RectangularFunction function,
    size_t n, size_t m, void* context, object keepalive=*, bint jacobian=*)
