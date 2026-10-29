"""Small external consumer of the rectangular callback ABI."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cpython.pycapsule cimport PyCapsule_New
from cyroot.native.rectangular cimport (
    RectangularCallback, make_rectangular_callback, cyroot_rectangular_descriptor)

cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil

cdef struct State:
    size_t calls
    size_t gil_held
    bint jacobian
    bint partial
    int code

cdef int evaluate(const double* x, size_t n, size_t m,
                  double* out, void* context) noexcept nogil:
    cdef State* state = <State*>context
    cdef size_t i, j
    state.calls += 1
    if PyGILState_Check(): state.gil_held += 1
    for i in range(m):
        if state.jacobian:
            for j in range(n):
                if not state.partial or i + j*m == 0:
                    out[i + j*m] = 10*(i+1) + j+1
        elif not state.partial or i == 0:
            out[i] = 0.
            for j in range(n): out[i] += (10*(i+1) + j+1)*x[j]
    return state.code

cdef class Owner:
    cdef State state
    cdef cyroot_rectangular_descriptor descriptor
    cdef object __weakref__

    def __init__(self, jacobian=False, partial=False, code=0):
        self.state.jacobian = jacobian
        self.state.partial = partial
        self.state.code = code

    def callback(self, n, m):
        return make_rectangular_callback(evaluate, n, m, <void*>&self.state,
                                         self, jacobian=self.state.jacobian)

    def capsule(self, n=2, m=3, kind=None, version=1, struct_size=None,
                null_function=False, wrong_name=False):
        self.descriptor.struct_size = sizeof(cyroot_rectangular_descriptor) if struct_size is None else struct_size
        self.descriptor.abi_version = version
        self.descriptor.input_dimension = n
        self.descriptor.output_dimension = m
        self.descriptor.kind = (2 if self.state.jacobian else 1) if kind is None else kind
        if null_function:self.descriptor.function = NULL
        else:self.descriptor.function = evaluate
        self.descriptor.context = <void*>&self.state
        return PyCapsule_New(<void*>&self.descriptor,
                             b'wrong.rectangular' if wrong_name else b'cyroot.rectangular_callback.v1', NULL)

    @property
    def counters(self): return self.state.calls, self.state.gil_held


def raw(RectangularCallback callback, point):
    """Follow the documented reservation/lifetime/error-transfer protocol."""
    cdef size_t n = callback.input_dimension, m = callback.output_dimension
    cdef size_t count = n*m if callback.jacobian else m
    cdef vector[double] x, out
    cdef int code
    callback._enter()
    try:
        x.resize(n); out.resize(count)
        for i in range(n): x[i] = point[i]
        with nogil:
            code = callback.evaluate_raw(x.data(), out.data())
        if callback._error is not None:
            error = callback._error
            callback._error = None
            raise error
        return code, tuple(out[i] for i in range(count))
    finally:
        callback._release()


def reserved(RectangularCallback callback, action):
    callback._enter()
    try: return action()
    finally: callback._release()


def null_factory():
    return make_rectangular_callback(NULL, 2, 3, NULL)
