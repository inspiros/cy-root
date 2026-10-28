"""Bounded exact derivative transform for a squared-residual objective."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector

cdef extern from "objective_derivatives.hpp" namespace "cyroot::native" nogil:
    int objective_derivative(const double* const* derivatives,
        const size_t* capacities, size_t n, size_t m, size_t order,
        double* out, size_t output_capacity, double* scratch,
        size_t scratch_capacity, size_t max_terms) noexcept

cdef class ObjectiveDerivativePlan:
    cdef size_t _n, _m, _order, _input_size, _output_size, _terms, _memory_bytes
    cdef vector[size_t] _sizes
    cdef bint _initialized, _constructing, _busy
    cdef void _ready(self) except *
    cdef object _transform(self, object tensors, object point=*)
    # Initialized immutable plan, pinned for this call. Trusted disjoint spans
    # with derivative_count pointers/capacities. No Python owner/reservation
    # needed; callers own input/output/scratch and obey configured shape/order.
    # Status/capacity/transactional rules are in objective_derivatives.hpp.
    cdef int compute(self, const double* const* derivatives,
        const size_t* capacities, size_t derivative_count, double* out,
        size_t output_capacity, double* scratch, size_t scratch_capacity) noexcept nogil
