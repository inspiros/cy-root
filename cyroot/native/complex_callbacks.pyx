# distutils: language=c++
# cython: language_level=3
"""Python and native bridges for complex scalar residuals."""
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid

cdef int _python_complex(double real, double imag, double* out_real,
                         double* out_imag, void* context) noexcept nogil:
    cdef double complex value
    with gil:
        callback = <ComplexScalarCallback><object>context
        try:
            value = callback._python_function(real + imag * 1j)
            out_real[0], out_imag[0] = value.real, value.imag
        except BaseException as error:
            callback._error = error
            return 1
    return 0

cdef class ComplexScalarCallback:
    """A Python callable or native complex residual with a pinned context.

    Native callbacks receive real/imaginary components, write both output
    components and return zero. Nonzero status terminates without consuming
    output. Concurrent/reentrant use of one adapter is rejected.
    """
    def __init__(self, function):
        if self._function != NULL:
            raise RuntimeError('Callback adapters cannot be reinitialized.')
        if not callable(function):
            raise TypeError('function must be callable.')
        self._python_function = function
        self._function = _python_complex
        self._context = <void*>self
        self._native = False

    @classmethod
    def from_capsule(cls, capsule, keepalive=None):
        cdef cyroot_complex_scalar_descriptor* descriptor
        if not PyCapsule_IsValid(capsule, b'cyroot.complex_scalar_callback.v1'):
            raise ValueError('Expected a cyroot.complex_scalar_callback.v1 capsule.')
        descriptor = <cyroot_complex_scalar_descriptor*>PyCapsule_GetPointer(
            capsule, b'cyroot.complex_scalar_callback.v1')
        if descriptor.struct_size < sizeof(cyroot_complex_scalar_descriptor) or descriptor.abi_version != 1:
            raise ValueError('Unsupported complex scalar callback descriptor size or ABI version.')
        return make_complex_scalar_callback(descriptor.function, descriptor.context, (capsule, keepalive))

    @property
    def native(self):
        return bool(self._native)

cdef ComplexScalarCallback make_complex_scalar_callback(
        ComplexScalarFunction function, void* context, object keepalive=None):
    if function == NULL:
        raise ValueError('Native complex scalar callback function must not be NULL.')
    cdef ComplexScalarCallback callback = ComplexScalarCallback.__new__(ComplexScalarCallback)
    callback._function = function
    callback._context = context
    callback._keepalive = keepalive
    callback._native = True
    return callback
