"""Native Jacobian-vector callbacks and allocation-free directional plans."""
from libc.stddef cimport size_t
from .rectangular cimport RectangularFunction

ctypedef int (*JacobianVectorFunction)(const double* x, const double* direction,
    size_t n, size_t m, double* out, void* context) noexcept nogil

cdef extern from "jvp_callbacks.h":
    ctypedef struct cyroot_jvp_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t input_dimension
        size_t output_dimension
        JacobianVectorFunction function
        void* context

cdef class JacobianVectorCallback:
    cdef JacobianVectorFunction _function
    cdef void* _context
    cdef object _keepalive, _python_function, _error
    cdef size_t _n, _m
    cdef bint _active, _native, _constructing
    cdef void _enter(self) except *
    cdef void _leave(self) noexcept
    # Caller reserves adapter, pins context, and supplies disjoint x[n],v[n],out[m].
    # Nonzero status poisons the entire output; no allocation occurs here.
    cdef int evaluate_raw(self, const double* x, const double* direction,
                          double* out) noexcept nogil

cdef JacobianVectorCallback make_jacobian_vector_callback(
    JacobianVectorFunction function, size_t n, size_t m, void* context, object keepalive=*)

cdef class JacobianVectorPlan:
    cdef size_t _n, _m, _work_size, _evaluation_size, _memory_bytes
    cdef double _step
    cdef int _scheme
    cdef bint _initialized, _constructing
    # Trusted initialized plan; disjoint x[n], direction[n], optional base[m],
    # out[m], work[2*n+3*m]. function/counters are non-NULL. Caller reserves and
    # pins callback/context. Base is NULL to sample F(x), otherwise a trusted cache.
    # No shared mutable scratch: one plan may serve independent concurrent callers.
    # On any failure the whole output is NaN; zero direction needs no samples.
    cdef int compute(self, const double* x, const double* direction,
                     const double* base, double* out, double* work,
                     RectangularFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil
