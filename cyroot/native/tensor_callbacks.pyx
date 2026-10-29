# distutils: language=c++
# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Python/native Hessian adapters with pinned contexts and copied boundaries."""
from numbers import Complex, Real
from libc.limits cimport INT_MAX
from libc.math cimport NAN
from .buffers cimport VectorView
from ._buffers cimport DoubleBuffer, is_native_double_buffer
from .scalar import NativeCallbackError
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from .buffers cimport OwnedDoubleBuffer
from .higher_derivatives import _checked_integer


cdef bint _valid_dimension(size_t n) noexcept nogil:
    cdef size_t maximum = OwnedDoubleBuffer.max_size()
    if n == 0 or n > INT_MAX or n > maximum // n:
        return False
    return n <= maximum // (n * n)


cdef void _pack_rectangular_tensor(object value, double* out, size_t n, size_t m) except *:
    cdef size_t i, j, k
    if isinstance(value, (str, bytes, bytearray)) or len(value) != m:
        raise ValueError('Hessian shape must be (output_dimension, input_dimension, input_dimension).')
    for i in range(m):
        plane = value[i]
        if isinstance(plane, (str, bytes, bytearray)) or len(plane) != n:
            raise ValueError('Hessian shape must be (output_dimension, input_dimension, input_dimension).')
        for j in range(n):
            row = plane[j]
            if isinstance(row, (str, bytes, bytearray)) or len(row) != n:
                raise ValueError('Hessian shape must be (output_dimension, input_dimension, input_dimension).')
            for k in range(n):
                entry = row[k]
                if ((isinstance(entry, Complex) and not isinstance(entry, Real)) or
                        getattr(getattr(entry, 'dtype', None), 'kind', None) == 'c'):
                    raise TypeError('Hessian entries must be real scalars.')
                out[i+m*(j+n*k)] = entry


cdef void _pack_tensor(object value, double* out, size_t n) except *:
    _pack_rectangular_tensor(value, out, n, n)


cdef int _python_hessian(const double* x, size_t n, double* out,
                          void* context) noexcept nogil:
    cdef size_t i
    with gil:
        callback = <HessianCallback><object>context
        try:
            point = tuple([x[i] for i in range(n)])
            value = callback._python_function(point)
            _pack_tensor(value, out, n)
        except BaseException as error:
            callback._error = error
            return 1
    return 0


cdef class HessianCallback:
    """A square residual Hessian callback; the adapter pins native context owners.

    Python callbacks receive a copied point tuple and return an indexable tensor
    with shape (residual, input, input). Native output is residual-first packed
    column-major. All entries must be written on success; no symmetry is imposed.
    """
    def __init__(self, function, dimension):
        cdef size_t n
        if self._function != NULL or self._constructing:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        self._constructing = True
        try:
            degree = _checked_integer(dimension, 'dimension')
            n = degree
            if not _valid_dimension(n):
                raise ValueError('dimension must fit the square Hessian storage and C-int ABI.')
            if not callable(function):
                raise TypeError('function must be callable.')
            self._dimension = n
            self._python_function = function
            self._context = <void*>self
            self._function = _python_hessian
        finally:
            self._constructing = False

    @classmethod
    def from_capsule(cls, capsule, keepalive=None):
        cdef cyroot_hessian_descriptor* descriptor
        if not PyCapsule_IsValid(capsule, b'cyroot.hessian_callback.v1'):
            raise ValueError('Expected a cyroot.hessian_callback.v1 capsule.')
        descriptor = <cyroot_hessian_descriptor*>PyCapsule_GetPointer(
            capsule, b'cyroot.hessian_callback.v1')
        if (descriptor.struct_size < sizeof(cyroot_hessian_descriptor) or
                descriptor.abi_version != 1):
            raise ValueError('Unsupported Hessian callback descriptor size or version.')
        return make_hessian_callback(descriptor.function, descriptor.dimension,
                                      descriptor.context, (capsule, keepalive))

    @property
    def native(self):
        return bool(self._native)

    @property
    def dimension(self):
        return self._dimension


cdef HessianCallback make_hessian_callback(HessianFunction function,
                                           size_t dimension, void* context,
                                           object keepalive=None):
    if function == NULL:
        raise ValueError('Native Hessian callback function must not be NULL.')
    if not _valid_dimension(dimension):
        raise ValueError('dimension must fit the square Hessian storage and C-int ABI.')
    cdef HessianCallback callback = HessianCallback.__new__(HessianCallback)
    callback._dimension = dimension
    callback._function = function
    callback._context = context
    callback._keepalive = keepalive
    callback._native = True
    return callback


cdef bint _valid_rectangular_dimension(size_t n, size_t m) noexcept nogil:
    cdef size_t maximum = OwnedDoubleBuffer.max_size(), matrix, tensor
    if n == 0 or m == 0 or n > INT_MAX or m > INT_MAX or n > maximum//m:return False
    matrix = n*m
    if n > maximum//matrix:return False
    tensor = matrix*n
    return n <= maximum-tensor


cdef void _poison(double* out, size_t count) noexcept nogil:
    cdef size_t i
    for i in range(count):out[i] = NAN


cdef void _pack_point(object value, double* out, size_t n) except *:
    cdef size_t i
    cdef DoubleBuffer guard
    cdef VectorView view
    if isinstance(value, (str, bytes, bytearray)):raise TypeError('Expected a numeric input vector.')
    if is_native_double_buffer(value):
        guard = DoubleBuffer(value)
        view = guard.vector_view()
        if view.size() != n:raise ValueError('Input length must match input_dimension.')
        for i in range(n):out[i] = view.get(i)
    else:
        if len(value) != n:raise ValueError('Input length must match input_dimension.')
        for i in range(n):out[i] = value[i]


cdef object _rectangular_tensor_tuple(const double* out, size_t n, size_t m):
    return tuple([tuple([tuple([out[i+m*(j+n*k)] for k in range(n)])
                        for j in range(n)]) for i in range(m)])


cdef int _python_rectangular_hessian(const double* x, size_t n, size_t m,
                                     double* out, void* context) noexcept nogil:
    cdef size_t i
    with gil:
        callback = <RectangularHessianCallback><object>context
        try:
            value = callback._python_function(tuple([x[i] for i in range(n)]))
            _pack_rectangular_tensor(value, out, n, m)
        except BaseException as error:
            callback._error = error
            return 1
    return 0


cdef class RectangularHessianCallback:
    """Analytic F:R^n -> R^m Hessian, preserving shape(m,n,n) without symmetry."""
    def __init__(self, function, input_dimension, output_dimension):
        cdef size_t n, m
        if self._function != NULL or self._constructing:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        self._constructing = True
        try:
            n_value = _checked_integer(input_dimension, 'input_dimension')
            m_value = _checked_integer(output_dimension, 'output_dimension')
            if n_value == 0 or m_value == 0 or n_value > INT_MAX or m_value > INT_MAX:
                raise ValueError('Positive Hessian dimensions must fit C int.')
            n, m = n_value, m_value
            if not _valid_rectangular_dimension(n, m):
                raise ValueError('Hessian shape and input storage exceed native capacities.')
            if not callable(function):raise TypeError('function must be callable.')
            self._n, self._m = n, m
            self._tensor_size = n*m*n
            self._python_function = function
            self._context = <void*>self
            self._function = _python_rectangular_hessian
        finally:self._constructing = False

    @classmethod
    def from_capsule(cls, capsule, keepalive=None):
        cdef cyroot_rectangular_hessian_descriptor* descriptor
        if not PyCapsule_IsValid(capsule, b'cyroot.rectangular_hessian_callback.v1'):
            raise ValueError('Expected a cyroot.rectangular_hessian_callback.v1 capsule.')
        descriptor = <cyroot_rectangular_hessian_descriptor*>PyCapsule_GetPointer(
            capsule, b'cyroot.rectangular_hessian_callback.v1')
        if (descriptor.struct_size < sizeof(cyroot_rectangular_hessian_descriptor) or
                descriptor.abi_version != 1):
            raise ValueError('Unsupported rectangular Hessian descriptor size or version.')
        return make_rectangular_hessian_callback(descriptor.function, descriptor.input_dimension,
            descriptor.output_dimension, descriptor.context, (capsule, keepalive))

    cdef void _enter(self) except *:
        if self._constructing or self._function == NULL:
            raise RuntimeError('Callback adapter is not initialized.')
        if self._active:raise RuntimeError('This callback adapter is already active.')
        self._active, self._error = True, None

    cdef void _leave(self) noexcept:self._active = False

    cdef int evaluate_raw(self, const double* x, double* out) noexcept nogil:
        cdef int code
        _poison(out, self._tensor_size)
        code = self._function(x, self._n, self._m, out, self._context)
        if code != 0:_poison(out, self._tensor_size)
        return code

    def evaluate(self, point, *, workspace_bytes=64*1024*1024):
        """Copy one tensor. Native consumers reserve this adapter and reuse spans."""
        cdef OwnedDoubleBuffer* storage = NULL
        cdef double* out
        cdef int code
        self._enter()
        try:
            limit = _checked_integer(workspace_bytes, 'workspace_bytes')
            count = int(self._n)+int(self._tensor_size)
            if count*sizeof(double) > limit:
                raise ValueError('Hessian input/output storage exceeds workspace_bytes.')
            storage = new OwnedDoubleBuffer(count)
            out = storage.data()+self._n
            _pack_point(point, storage.data(), self._n)
            with nogil:code = self.evaluate_raw(storage.data(), out)
            if self._error is not None:
                error = self._error
                self._error = None
                raise error
            result = _rectangular_tensor_tuple(out, self._n, self._m)
            if code != 0:raise NativeCallbackError(code, result, 'hessian')
            return result
        finally:
            if storage != NULL:del storage
            self._leave()

    @property
    def input_dimension(self):return self._n
    @property
    def output_dimension(self):return self._m
    @property
    def shape(self):return self._m, self._n, self._n
    @property
    def native(self):return bool(self._native)


cdef RectangularHessianCallback make_rectangular_hessian_callback(
        RectangularHessianFunction function, size_t n, size_t m, void* context,
        object keepalive=None):
    if function == NULL:raise ValueError('Native Hessian callback function must not be NULL.')
    if not _valid_rectangular_dimension(n, m):
        raise ValueError('Hessian dimensions/storage must fit native capacities and C int.')
    cdef RectangularHessianCallback callback = RectangularHessianCallback.__new__(RectangularHessianCallback)
    callback._n, callback._m, callback._tensor_size = n, m, n*m*n
    callback._function, callback._context = function, context
    callback._keepalive = keepalive
    callback._native = True
    return callback
