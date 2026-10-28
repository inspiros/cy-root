"""Portable status/output-pointer complex scalar callback ABI."""
from libc.stddef cimport size_t

ctypedef int (*ComplexScalarFunction)(double real, double imag, double* out_real,
                                    double* out_imag, void* context) noexcept nogil
cdef extern from "complex_callbacks.h":
    ctypedef struct cyroot_complex_scalar_descriptor:
        size_t struct_size
        unsigned int abi_version
        ComplexScalarFunction function
        void* context

cdef class ComplexScalarCallback:
    cdef ComplexScalarFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef bint _active, _native

cdef ComplexScalarCallback make_complex_scalar_callback(
    ComplexScalarFunction function, void* context, object keepalive=*)
