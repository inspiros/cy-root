"""Immutable finite-difference scalar derivative plans, independent of solvers."""
from libc.stddef cimport size_t


ctypedef int (*ScalarResidualFunction)(double x, double* out,
                                      void* context) noexcept nogil

cdef enum ScalarDiffStatus:
    SCALAR_DIFF_OK = 0
    SCALAR_DIFF_NONFINITE = 1
    SCALAR_DIFF_BREAKDOWN = 2
    SCALAR_DIFF_CALLBACK_FAILURE = 3


cdef class ScalarDerivativePlan:
    cdef int _scheme
    cdef double _step
    cdef bint _automatic, _initialized, _constructing

    # The plan must be initialized. out, function, calls and callback_code are
    # non-NULL trusted native pointers; context is passed unchanged. A callback
    # writes one double and returns zero on success; it must obey noexcept.
    # calls is incremented for each callback attempt, including failed ones.
    # The plan owns no mutable scratch and can serve independent threads.
    cdef int compute(self, double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil


cdef class ScalarSecondDerivativePlan:
    cdef int _scheme
    cdef double _step
    cdef bint _automatic, _initialized, _constructing

    # Same trusted pointer/callback/lifetime contract as ScalarDerivativePlan.
    # f0 is the cached residual at x. Each scheme samples two other points.
    cdef int compute(self, double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code) noexcept nogil
