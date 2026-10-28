"""Standalone installed-header consumer, with no NumPy or solver dependency."""
from libc.stddef cimport size_t
from libc.stdint cimport int64_t
from libcpp.vector cimport vector
from cyroot.native.householder cimport ScaledCoefficient, householder_step


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


def compute_direct(values, order=None, capacity=None, size_t repeat=1):
    """Guarded native spans; deliberately short capacity is safe to test."""
    cdef size_t length = len(values)
    cdef size_t actual_capacity = 2 * length
    cdef size_t claimed_order = length - 1 if order is None else order
    cdef size_t claimed_capacity = actual_capacity if capacity is None else capacity
    if length == 0:
        raise ValueError('consumer needs one input for its guarded span')
    if claimed_capacity > actual_capacity:
        raise ValueError('claimed capacity exceeds consumer allocation')
    cdef vector[double] derivatives
    cdef vector[ScaledCoefficient] scratch
    derivatives.resize(length + 2)
    scratch.resize(actual_capacity + 2)
    cdef size_t i
    derivatives[0] = 1234567.
    derivatives[length+1] = 7654321.
    for i in range(length):
        derivatives[i+1] = values[i]
    for i in range(actual_capacity+2):
        scratch[i].mantissa = 918273.
        scratch[i].exponent = -314159
    scratch[0].mantissa = 1234567.
    scratch[0].exponent = 3456
    scratch[actual_capacity+1].mantissa = 7654321.
    scratch[actual_capacity+1].exponent = -6543
    cdef double output = 42.
    cdef int status = -1
    cdef int gil_before = -1
    cdef int gil_after = -1
    with nogil:
        gil_before = PyGILState_Check()
        for i in range(repeat):
            status = householder_step(&derivatives[1], claimed_order,
                                      &scratch[1], claimed_capacity, &output)
        gil_after = PyGILState_Check()
    return {
        'status': status,
        'step': output,
        'gil_held': (gil_before, gil_after),
        'input': tuple(derivatives[i+1] for i in range(length)),
        'input_guards': (derivatives[0], derivatives[length+1]),
        'workspace_guards': ((scratch[0].mantissa, scratch[0].exponent),
                             (scratch[actual_capacity+1].mantissa,
                              scratch[actual_capacity+1].exponent)),
        'workspace': tuple((scratch[i+1].mantissa, scratch[i+1].exponent)
                           for i in range(actual_capacity)),
    }
