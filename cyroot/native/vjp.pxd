"""Native supplied Jacobian-transpose callbacks."""
from libc.stddef cimport size_t

ctypedef int (*JacobianTransposeFunction)(const double* x, const double* direction,
    size_t n, size_t m, double* out, void* context) noexcept nogil

cdef extern from "vjp_callbacks.h":
    ctypedef struct cyroot_vjp_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t input_dimension
        size_t output_dimension
        JacobianTransposeFunction function
        void* context

cdef class JacobianTransposeCallback:
    cdef JacobianTransposeFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef size_t _n, _m
    cdef bint _active, _native, _constructing
    cdef void _enter(self) except *
    cdef void _leave(self) noexcept
    # Caller reserves adapter, pins context, and supplies disjoint x[n],w[m],out[n].
    # Nonzero status poisons the entire output; no allocation occurs here.
    cdef int evaluate_raw(self, const double* x, const double* direction,
                          double* out) noexcept nogil

cdef JacobianTransposeCallback make_jacobian_transpose_callback(
    JacobianTransposeFunction function, size_t n, size_t m, void* context, object keepalive=*)
