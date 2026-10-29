# distutils: language=c++
# cython: language_level=3
"""Pinned, read-only borrowing and private native copies of double buffers.

No NumPy dependency is used here. Borrowing pins a managed memoryview and never
makes its underlying storage writable through this guard. A mutable exporter
can still be changed by its owner; callers must synchronize such mutations
while native code reads a borrowed view. copy=True releases the exporter after
copying and supplies private writable contiguous storage.
"""
from operator import index as integer_index

from cpython.buffer cimport (
    PyObject_CheckBuffer, PyObject_GetBuffer, PyBuffer_Release, PyBuffer_FillInfo,
    PyBuffer_IsContiguous,
    PyBUF_FORMAT, PyBUF_STRIDES, PyBUF_ND, PyBUF_WRITABLE,
    PyBUF_C_CONTIGUOUS, PyBUF_F_CONTIGUOUS, PyBUF_ANY_CONTIGUOUS,
)
from libc.stddef cimport ptrdiff_t
from libc.string cimport strcmp
from libcpp.limits cimport numeric_limits


cdef bint _native_double_format(const char* format) noexcept:
    return (format != NULL and (strcmp(format, 'd') == 0 or strcmp(format, '@d') == 0 or
                                strcmp(format, '=d') == 0))


cdef bint is_native_double_buffer(object source) except -1:
    """True for a buffer exporter whose elements DoubleBuffer can borrow.

    Other exporters, such as integer or float32 arrays, are left to the
    callers' element-wise sequence conversion instead of being rejected here.
    """
    cdef Py_buffer view
    cdef bint result
    if not PyObject_CheckBuffer(source):
        return False
    try:
        # Request strides as DoubleBuffer does, so strided exporters qualify.
        PyObject_GetBuffer(source, &view, PyBUF_FORMAT | PyBUF_STRIDES)
    except BufferError:
        return False
    try:
        result = view.itemsize == sizeof(double) and _native_double_format(view.format)
        # Element-wise conversion of NumPy complex scalars would silently
        # discard imaginary parts; booleans are not numeric coordinates.
        if not result and view.format != NULL and (
                b'Z' in <bytes>view.format or b'?' in <bytes>view.format):
            raise TypeError('Expected real numeric values, not complex or bool.')
    finally:
        PyBuffer_Release(&view)
    return result


cdef class DoubleBuffer:
    """Borrow a native-format double vector/matrix, or copy to private storage.

    Only direct 1D/2D buffers with native ``d`` format are supported. Negative,
    zero, and unaligned byte strides are valid; indirect buffers are rejected.
    The guard's Python buffer export is read-only for borrowed data and writable
    for private copies. Shape/stride metadata is copied at construction.
    """

    def __cinit__(self):
        self._has_buffer = False
        self._exporter = None
        self._initialized = False
        self._owned = NULL
        self._ndim = 0
        self._count = 0
        self._nbytes = 0

    def __init__(self, source, *, copy=False):
        cdef Py_ssize_t axis
        cdef size_t row, column, offset
        cdef double value
        cdef bint make_copy
        if self._initialized or self._initializing:
            raise RuntimeError('DoubleBuffer cannot be reinitialized.')
        self._initializing = True
        try:
            # Convert once: arbitrary __bool__ implementations can change their
            # answer or reenter construction. Pinning decisions must agree.
            make_copy = bool(copy)
            # The managed memoryview exposes its exporter reference to GC.
            self._exporter = memoryview(source)
            PyObject_GetBuffer(self._exporter, &self._buffer, PyBUF_FORMAT | PyBUF_STRIDES)
            self._has_buffer = True
            if self._buffer.ndim not in (1, 2):
                raise ValueError('DoubleBuffer requires a one- or two-dimensional buffer.')
            if (self._buffer.itemsize != sizeof(double) or
                    not _native_double_format(self._buffer.format)):
                raise ValueError('DoubleBuffer requires native-format double elements.')
            if self._buffer.shape == NULL or self._buffer.strides == NULL:
                raise ValueError('Buffer exporter omitted shape or stride metadata.')
            self._ndim = self._buffer.ndim
            for axis in range(self._ndim):
                if self._buffer.shape[axis] < 0:
                    raise ValueError('Buffer shapes must be nonnegative.')
                if (self._buffer.strides[axis] > numeric_limits[ptrdiff_t].max() or
                        self._buffer.strides[axis] < numeric_limits[ptrdiff_t].min()):
                    raise OverflowError('Buffer stride exceeds ptrdiff_t.')
                if self._buffer.suboffsets != NULL and self._buffer.suboffsets[axis] >= 0:
                    raise ValueError('Indirect buffers are not supported.')
                self._shape[axis] = self._buffer.shape[axis]
                self._strides[axis] = self._buffer.strides[axis]
            if self._ndim == 1:
                self._vector = VectorView(self._buffer.buf, <size_t>self._shape[0],
                                          <ptrdiff_t>self._strides[0])
                self._count = self._vector.size()
            else:
                self._matrix = MatrixView(self._buffer.buf,
                                          <size_t>self._shape[0], <size_t>self._shape[1],
                                          <ptrdiff_t>self._strides[0], <ptrdiff_t>self._strides[1])
                self._count = self._matrix.size()
            self._nbytes = <Py_ssize_t>(self._count * sizeof(double))
            if self._buffer.len != self._nbytes:
                raise ValueError('Buffer byte length disagrees with its shape.')
            if make_copy:
                self._owned = new OwnedDoubleBuffer(self._count)
                offset = 0
                if self._ndim == 1:
                    for column in range(self._count):
                        self._owned.set(column, self._vector.get(column))
                    self._vector = self._owned.view()
                    self._strides[0] = sizeof(double)
                else:
                    for row in range(<size_t>self._shape[0]):
                        for column in range(<size_t>self._shape[1]):
                            value = self._matrix.get(row, column)
                            self._owned.set(offset, value)
                            offset += 1
                    self._matrix = self._owned.matrix_view(
                        <size_t>self._shape[0], <size_t>self._shape[1])
                    self._strides[0] = self._matrix.row_stride_bytes()
                    self._strides[1] = sizeof(double)
            PyBuffer_Release(&self._buffer)
            self._has_buffer = False
            if make_copy:
                self._exporter = None
            self._initialized = True
        except:
            if self._has_buffer:
                PyBuffer_Release(&self._buffer)
                self._has_buffer = False
            if self._owned != NULL:
                del self._owned
                self._owned = NULL
            self._exporter = None
            self._ndim = 0
            raise
        finally:
            self._initializing = False

    def __dealloc__(self):
        if self._has_buffer:
            PyBuffer_Release(&self._buffer)
        if self._owned != NULL:
            del self._owned

    def __getbuffer__(self, Py_buffer* buffer, int flags):
        cdef Py_buffer metadata
        cdef bint c_contiguous, f_contiguous
        cdef void* data
        if not self._initialized:
            raise BufferError('DoubleBuffer is not initialized.')
        if self._ndim == 1:
            data = <void*>self._vector.data()
        else:
            data = <void*>self._matrix.data()
        metadata.buf = data
        metadata.len = self._nbytes
        metadata.ndim = self._ndim
        metadata.itemsize = sizeof(double)
        metadata.shape = self._shape
        metadata.strides = self._strides
        metadata.suboffsets = NULL
        c_contiguous = PyBuffer_IsContiguous(&metadata, 'C')
        f_contiguous = PyBuffer_IsContiguous(&metadata, 'F')
        if (((flags & PyBUF_C_CONTIGUOUS) == PyBUF_C_CONTIGUOUS and not c_contiguous) or
                ((flags & PyBUF_F_CONTIGUOUS) == PyBUF_F_CONTIGUOUS and not f_contiguous) or
                ((flags & PyBUF_ANY_CONTIGUOUS) == PyBUF_ANY_CONTIGUOUS and
                 not (c_contiguous or f_contiguous)) or
                ((flags & PyBUF_STRIDES) != PyBUF_STRIDES and not c_contiguous)):
            raise BufferError('Requested buffer layout is not available.')
        PyBuffer_FillInfo(buffer, self, data, self._nbytes, self._owned == NULL, flags)
        buffer.itemsize = sizeof(double) if flags & (PyBUF_ND | PyBUF_FORMAT) else 1
        buffer.ndim = self._ndim if flags & PyBUF_ND else 1
        if flags & PyBUF_FORMAT:
            buffer.format = <char*>'d'
        else:
            buffer.format = NULL
        if flags & PyBUF_ND:
            buffer.shape = self._shape
        else:
            buffer.shape = NULL
        if (flags & PyBUF_STRIDES) == PyBUF_STRIDES:
            buffer.strides = self._strides
        else:
            buffer.strides = NULL
        buffer.suboffsets = NULL

    def __releasebuffer__(self, Py_buffer* buffer):
        pass

    @property
    def ndim(self):
        return self._ndim

    @property
    def shape(self):
        return tuple(self._shape[axis] for axis in range(self._ndim))

    @property
    def strides(self):
        return tuple(self._strides[axis] for axis in range(self._ndim))

    @property
    def size(self):
        return self._count

    @property
    def readonly(self):
        return self._owned == NULL

    @property
    def owns_data(self):
        return self._owned != NULL

    def __len__(self):
        return self._shape[0] if self._initialized else 0

    cdef size_t _flat_index(self, object index) except *:
        cdef object row, column
        if not self._initialized:
            raise RuntimeError('DoubleBuffer is not initialized.')
        if self._ndim == 1:
            if isinstance(index, tuple):
                if len(index) != 1:
                    raise IndexError('A vector requires one index.')
                index = index[0]
            column = integer_index(index)
            if column < 0:
                column += self._shape[0]
            if column < 0 or column >= self._shape[0]:
                raise IndexError('Vector index is outside its shape.')
            return <size_t>column
        if not isinstance(index, tuple) or len(index) != 2:
            raise IndexError('A matrix requires a (row, column) index.')
        row, column = integer_index(index[0]), integer_index(index[1])
        if row < 0:
            row += self._shape[0]
        if column < 0:
            column += self._shape[1]
        if row < 0 or row >= self._shape[0] or column < 0 or column >= self._shape[1]:
            raise IndexError('Matrix index is outside its shape.')
        return <size_t>row * <size_t>self._shape[1] + <size_t>column

    def __getitem__(self, index):
        cdef size_t flat = self._flat_index(index)
        if self._ndim == 1:
            return self._vector.get(flat)
        return self._matrix.get(flat // <size_t>self._shape[1], flat % <size_t>self._shape[1])

    def __setitem__(self, index, double value):
        if self._owned == NULL:
            raise TypeError('Borrowed DoubleBuffer storage is read-only.')
        self._owned.set(self._flat_index(index), value)

    def fill(self, double value):
        if self._owned == NULL:
            raise TypeError('Borrowed DoubleBuffer storage is read-only.')
        self._owned.fill(value)

    def copy(self):
        return DoubleBuffer(self, copy=True)

    def tolist(self):
        if not self._initialized:
            raise RuntimeError('DoubleBuffer is not initialized.')
        if self._ndim == 1:
            return [self._vector.get(column) for column in range(self._count)]
        return [[self._matrix.get(row, column) for column in range(<size_t>self._shape[1])]
                for row in range(<size_t>self._shape[0])]

    cdef VectorView vector_view(self) except *:
        if not self._initialized or self._ndim != 1:
            raise ValueError('A vector view requires an initialized one-dimensional guard.')
        return self._vector

    cdef MatrixView matrix_view(self) except *:
        if not self._initialized or self._ndim != 2:
            raise ValueError('A matrix view requires an initialized two-dimensional guard.')
        return self._matrix
