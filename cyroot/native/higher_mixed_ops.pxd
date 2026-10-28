"""Allocation-free tensor finite differences over an immutable planned graph."""
from libc.stddef cimport size_t
from .rectangular cimport RectangularFunction

cdef extern from "higher_mixed.hpp" namespace "cyroot::native" nogil:
    cdef struct TensorDifferenceConfig:
        size_t n
        size_t m
        size_t order
        size_t components
        size_t samples
        size_t terms
        size_t tensor_size
        size_t input_size
        size_t nodes
        size_t work
        int scheme
        const double* steps
        const size_t* indices
        const size_t* starts
        const size_t* term_samples
        const int* offsets
        const size_t* scatter

    # Graph contents/allocated spans are trusted, immutable and pinned by the
    # caller. x[n],f0[m],out[output_capacity],work[capacity],counter/code are
    # disjoint; callback obeys RectangularFunction's no-throw contract.
    # Invalid counts/capacities return2 before scratch/output writes; otherwise
    # failures poison the entire tensor and restore the work trialpoint to x.
    # Return0 ok,1 nonfinite,2 breakdown,3 callback failure. calls excludes f0.
    int compute_tensor_difference(const TensorDifferenceConfig* config,
        const double* x, const double* f0, double* out, size_t output_capacity,
        double* work, size_t capacity, RectangularFunction function,
        void* context, size_t* calls, int* callback_code) noexcept
