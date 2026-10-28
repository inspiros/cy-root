"""Header-only storage; use cyroot.native.get_include() for native headers."""
from libc.stddef cimport size_t, ptrdiff_t

cdef extern from "buffers.hpp" namespace "cyroot::native" nogil:
    cdef cppclass VectorView:
        VectorView() noexcept
        VectorView(const void* data, size_t size, ptrdiff_t stride_bytes) except +
        @staticmethod
        VectorView from_span(const void* storage, size_t storage_bytes,
                             size_t origin_bytes, size_t size, ptrdiff_t stride_bytes) except +
        size_t size() noexcept
        ptrdiff_t stride_bytes() noexcept
        const void* data() noexcept
        bint empty() noexcept
        double get(size_t index) except +

    cdef cppclass MatrixView:
        MatrixView() noexcept
        MatrixView(const void* data, size_t rows, size_t columns,
                   ptrdiff_t row_stride_bytes, ptrdiff_t column_stride_bytes) except +
        @staticmethod
        MatrixView from_span(const void* storage, size_t storage_bytes,
                             size_t origin_bytes, size_t rows, size_t columns,
                             ptrdiff_t row_stride_bytes, ptrdiff_t column_stride_bytes) except +
        size_t rows() noexcept
        size_t columns() noexcept
        size_t size() noexcept
        ptrdiff_t row_stride_bytes() noexcept
        ptrdiff_t column_stride_bytes() noexcept
        const void* data() noexcept
        bint empty() noexcept
        double get(size_t row, size_t column) except +

    cdef cppclass OwnedDoubleBuffer:
        OwnedDoubleBuffer() noexcept
        OwnedDoubleBuffer(size_t size) except +
        @staticmethod
        size_t max_size() noexcept
        size_t size() noexcept
        size_t size_bytes() noexcept
        bint empty() noexcept
        double* data() noexcept
        double get(size_t index) except +
        void set(size_t index, double value) except +
        void fill(double value) noexcept
        void reset() noexcept
        VectorView view() except +
        MatrixView matrix_view(size_t rows, size_t columns) except +
