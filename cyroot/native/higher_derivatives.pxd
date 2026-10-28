"""Runtime-order scalar finite differences with caller-owned native scratch."""
from libc.stddef cimport size_t
from .scalar_derivatives cimport ScalarResidualFunction


cdef struct ScalarDifferenceConfig:
    size_t order
    size_t nodes
    size_t work
    int scheme
    double step
    bint automatic


# Native value descriptor copied once from an initialized immutable plan.
# Pointer/spans/counting preconditions match plan.compute below. The free
# kernel checks shape/product/capacity before scratch writes; no Python owner
# or object reference is needed by callers holding the descriptor.
cdef int compute_scalar_difference(const ScalarDifferenceConfig* config,
                     double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code,
                     double* workspace, size_t capacity) noexcept nogil


cdef class ScalarFiniteDifferencePlan:
    cdef size_t _order, _nodes, _work
    cdef int _scheme
    cdef double _step
    cdef bint _automatic, _initialized, _constructing

    # Caller first checks native _initialized. This copies value fields only.
    cdef void copy_config(self, ScalarDifferenceConfig* config) noexcept nogil

    # Initialized plan; all pointers non-NULL. workspace has capacity doubles,
    # out/calls/callback_code and workspace are disjoint from one another and
    # callback-visible storage. Scratch contents are unspecified on return.
    # function obeys the no-throw ScalarResidualFunction contract. calls counts
    # all stencil attempts; f0 is supplied by the caller and is not counted here.
    cdef int compute(self, double x, double f0, double* out,
                     ScalarResidualFunction function, void* context,
                     size_t* calls, int* callback_code,
                     double* workspace, size_t capacity) noexcept nogil
