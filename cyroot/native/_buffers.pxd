from libc.stddef cimport size_t
from .buffers cimport OwnedDoubleBuffer, VectorView, MatrixView

cdef class DoubleBuffer:
    cdef Py_buffer _buffer
    cdef object _exporter
    cdef bint _has_buffer, _initialized, _initializing
    cdef OwnedDoubleBuffer* _owned
    cdef VectorView _vector
    cdef MatrixView _matrix
    cdef int _ndim
    cdef size_t _count
    cdef Py_ssize_t _shape[2]
    cdef Py_ssize_t _strides[2]
    cdef Py_ssize_t _nbytes
    cdef size_t _flat_index(self, object index) except *
    # The calling Cython code must retain this guard while using these copies.
    cdef VectorView vector_view(self) except *
    cdef MatrixView matrix_view(self) except *
