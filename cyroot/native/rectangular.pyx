# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Python/native callback adapters with separate input and output dimensions."""
from libc.limits cimport INT_MAX
from libc.math cimport NAN
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from .buffers cimport OwnedDoubleBuffer, VectorView, MatrixView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .higher_derivatives import _checked_integer
from .scalar import NativeCallbackError


cdef bint _valid_shape(size_t n, size_t m, bint jacobian) noexcept nogil:
    cdef size_t maximum = OwnedDoubleBuffer.max_size()
    if n == 0 or m == 0 or n > INT_MAX or m > INT_MAX:return False
    if n > maximum or m > maximum:return False
    if jacobian and m > maximum // n:return False
    return n <= maximum - (m*n if jacobian else m)


cdef void _pack_vector(object source, double* out, size_t count) except *:
    cdef DoubleBuffer guard
    cdef VectorView view
    cdef size_t i
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric vector.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        view = guard.vector_view()
        if view.size() != count:raise ValueError('Vector length must match the callback dimension.')
        for i in range(count):out[i] = view.get(i)
    else:
        if len(source) != <Py_ssize_t>count:
            raise ValueError('Vector length must match the callback dimension.')
        for i in range(count):out[i] = source[i]


cdef void _pack_matrix(object source, double* out, size_t m, size_t n) except *:
    cdef DoubleBuffer guard
    cdef MatrixView view
    cdef size_t i, j
    if isinstance(source, (str, bytes, bytearray)):
        raise TypeError('Expected a numeric Jacobian matrix.')
    if is_native_double_buffer(source):
        guard = DoubleBuffer(source)
        view = guard.matrix_view()
        if view.rows() != m or view.columns() != n:
            raise ValueError('Jacobian shape must be (output_dimension, input_dimension).')
        for j in range(n):
            for i in range(m):out[i+j*m] = view.get(i, j)
    else:
        if len(source) != <Py_ssize_t>m:
            raise ValueError('Jacobian shape must be (output_dimension, input_dimension).')
        for i in range(m):
            row = source[i]
            if isinstance(row, (str, bytes, bytearray)) or len(row) != <Py_ssize_t>n:
                raise ValueError('Jacobian shape must be (output_dimension, input_dimension).')
            for j in range(n):out[i+j*m] = row[j]


cdef void _poison(double* out, size_t count) noexcept nogil:
    cdef size_t i
    for i in range(count):out[i] = NAN


cdef int _python_callback(const double* x, size_t n, size_t m,
                          double* out, void* context) noexcept nogil:
    cdef size_t i
    with gil:
        callback = <RectangularCallback><object>context
        try:
            value = callback._python_function(tuple([x[i] for i in range(n)]))
            if callback._jacobian:_pack_matrix(value, out, m, n)
            else:_pack_vector(value, out, m)
        except BaseException as error:
            callback._error = error
            return 1
    return 0


cdef class RectangularCallback:
    """A residual of length m or column-major Jacobian of shape (m, n)."""
    def __init__(self, function, input_dimension, output_dimension, *, jacobian=False):
        if self._function != NULL or self._constructing:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        self._constructing = True
        try:
            n = _checked_integer(input_dimension, 'input_dimension')
            m = _checked_integer(output_dimension, 'output_dimension')
            kind = bool(jacobian)
            if not _valid_shape(n, m, kind):
                raise ValueError('Positive callback dimensions must fit native capacities and C int.')
            if not callable(function):raise TypeError('function must be callable.')
            self._n, self._m = n, m
            self._output_size = m*n if kind else m
            self._jacobian = kind
            self._python_function = function
            self._context = <void*>self
            self._function = _python_callback
        finally:
            self._constructing = False

    @classmethod
    def from_capsule(cls, capsule, keepalive=None):
        cdef cyroot_rectangular_descriptor* descriptor
        if not PyCapsule_IsValid(capsule, b'cyroot.rectangular_callback.v1'):
            raise ValueError('Expected a cyroot.rectangular_callback.v1 capsule.')
        descriptor = <cyroot_rectangular_descriptor*>PyCapsule_GetPointer(
            capsule, b'cyroot.rectangular_callback.v1')
        if (descriptor.struct_size < sizeof(cyroot_rectangular_descriptor) or
                descriptor.abi_version != 1 or descriptor.kind not in (1, 2)):
            raise ValueError('Unsupported rectangular callback descriptor size, version or kind.')
        return make_rectangular_callback(descriptor.function, descriptor.input_dimension,
            descriptor.output_dimension, descriptor.context, (capsule, keepalive), descriptor.kind == 2)

    cdef void _enter(self) except *:
        if self._constructing or self._function == NULL:
            raise RuntimeError('Callback adapter is not initialized.')
        if self._active:raise RuntimeError('This callback adapter is already active.')
        self._active = True
        self._error = None

    cdef void _release(self) noexcept:
        self._active = False

    cdef int evaluate_raw(self, const double* x, double* out) noexcept nogil:
        cdef int code
        _poison(out, self._output_size)
        code = self._function(x, self._n, self._m, out, self._context)
        if code != 0:_poison(out, self._output_size)
        return code

    def evaluate(self, point, *, workspace_bytes=64*1024*1024):
        """Copy/evaluate one point; native loops use a reserved evaluate_raw call."""
        cdef OwnedDoubleBuffer* storage = NULL
        cdef double* output
        cdef size_t i, j
        cdef int code
        self._enter()
        try:
            limit = _checked_integer(workspace_bytes, 'workspace_bytes')
            count = int(self._n)+int(self._output_size)
            if count*sizeof(double) > limit:
                raise ValueError('Callback input/output storage exceeds workspace_bytes.')
            storage = new OwnedDoubleBuffer(count)
            output = storage.data()+self._n
            _pack_vector(point, storage.data(), self._n)
            with nogil:code = self.evaluate_raw(storage.data(), output)
            if self._error is not None:
                error = self._error
                self._error = None
                raise error
            if self._jacobian:
                result = tuple([tuple([output[i+j*self._m] for j in range(self._n)])
                                for i in range(self._m)])
            else:result = tuple([output[i] for i in range(self._m)])
            if code != 0:
                raise NativeCallbackError(code, result, 'jacobian' if self._jacobian else 'function')
            return result
        finally:
            if storage != NULL:del storage
            self._release()

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def jacobian(self):return bool(self._jacobian)
    @property
    def native(self):return bool(self._native)


cdef RectangularCallback make_rectangular_callback(RectangularFunction function,
    size_t n, size_t m, void* context, object keepalive=None, bint jacobian=False):
    if function == NULL:raise ValueError('Native callback function must not be NULL.')
    if not _valid_shape(n, m, jacobian):
        raise ValueError('Positive callback dimensions must fit native capacities and C int.')
    cdef RectangularCallback callback = RectangularCallback.__new__(RectangularCallback)
    callback._n, callback._m = n, m
    callback._output_size = m*n if jacobian else m
    callback._jacobian = jacobian
    callback._function = function
    callback._context = context
    callback._keepalive = keepalive
    callback._native = True
    return callback
