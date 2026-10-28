"""Reuse real callback fixtures; add a portable complex callback producer."""
include "native_secant_consumer.pyx"

from cyroot.native.complex_callbacks cimport make_complex_scalar_callback

cdef int complex_quadratic(double real, double imag, double* out_real,
                            double* out_imag, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    parameters.calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after >= 0 and parameters.calls > <size_t>parameters.fail_after:
        return parameters.failure_code
    out_real[0] = real * real - imag * imag + parameters.target
    if parameters.partial_after < 0 or parameters.calls <= <size_t>parameters.partial_after:
        out_imag[0] = 2 * real * imag
    return 0

cdef class ComplexOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, target=1., fail_after=-1, failure_code=-23, partial_after=-1):
        self.parameters.target = target
        self.parameters.calls = self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.failure_code = failure_code
        self.parameters.partial_after = partial_after

    def callback(self):
        return make_complex_scalar_callback(complex_quadratic, <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held
