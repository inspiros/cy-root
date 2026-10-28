"""Reusable finite-difference configuration for native square residuals."""
from libc.stddef cimport size_t
from .buffers cimport OwnedDoubleBuffer
from .rectangular cimport RectangularFunction


# Deliberately independent of vector.pxd: derivative planning must not import
# the solver, and the identical native callback signature is interchangeable.
ctypedef int (*ResidualFunction)(const double* x, size_t n, double* out,
                                void* context) noexcept nogil

cdef enum DiffStatus:
    DIFF_OK = 0
    DIFF_NONFINITE = 1
    DIFF_BREAKDOWN = 2
    DIFF_CALLBACK_FAILURE = 3


# Square Traub-Steffensen divided differences: signed steps f0[j], same
# disjoint x[n], f0[n], out[n*n], work[5*n] contract as JacobianPlan.compute.
# n/products/spans must already be checked; callback/counters must be valid.
# Zero/collapsed steps fail before sampling; no automatic-step fallback.
cdef int _steffensen_jacobian(size_t n, const double* x, const double* f0,
    double* out, double* work, ResidualFunction function, void* context,
    size_t* calls, int* callback_code) noexcept nogil


cdef class JacobianPlan:
    cdef size_t _dimension, _square, _step_count
    cdef int _scheme
    cdef OwnedDoubleBuffer* _steps
    cdef bint _initialized, _constructing

    # Trusted native boundary: the initialized plan's dimension is n. x and f0
    # address n readable doubles; out addresses n*n writable column-major
    # doubles; work addresses 5*n writable doubles. These four spans must be
    # disjoint. function, calls and callback_code must be non-NULL; context is
    # supplied unchanged to function. The callback may only read its x input,
    # must write all n output values on success, and must obey noexcept.
    # Scratch/output and callback context belong to the caller, so one immutable
    # plan may serve independent concurrent solves. No allocation occurs here.
    cdef int compute(self, const double* x, const double* f0, double* out,
                     double* work, ResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil

    cdef void _configure(self, object dimension, object output_dimension,
                         object scheme, object step, object workspace_bytes) except *


cdef class RectangularJacobianPlan:
    cdef JacobianPlan _axes
    cdef size_t _m, _work, _evaluation_size, _memory_bytes
    cdef bint _initialized, _constructing
    # Trusted disjoint x[n], base[m], out[m*n], work[3*n+2*m] spans.
    # Caller reserves callback, pins its context and provides non-NULL counters.
    # No allocation; any failure clears the entire known output to NaN.
    cdef int compute(self, const double* x, const double* f0, double* out,
                     double* work, RectangularFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil
