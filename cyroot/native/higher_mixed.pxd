"""Bounded dense runtime-order vector finite differences (native preview)."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from .buffers cimport OwnedDoubleBuffer
from .rectangular cimport RectangularFunction
from .higher_mixed_ops cimport TensorDifferenceConfig

cdef class TensorDifferencePlan:
    cdef vector[size_t] _indices, _starts, _term_samples, _scatter
    cdef vector[int] _offsets
    cdef OwnedDoubleBuffer* _steps
    cdef size_t _n, _m, _order, _components, _samples, _terms, _input_size
    cdef size_t _tensor_size, _nodes, _work, _evaluation_size, _step_count
    cdef size_t _metadata_bytes, _memory_bytes
    cdef int _scheme
    cdef bint _initialized, _constructing
    cdef void copy_config(self, TensorDifferenceConfig* config) noexcept nogil
    cdef int compute(self, const double* x, const double* f0,
        double* out, size_t output_capacity, double* workspace, size_t capacity,
        RectangularFunction function, void* context, size_t* calls,
        int* callback_code) noexcept nogil
