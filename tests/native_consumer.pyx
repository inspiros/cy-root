"""External installed-package consumer of the public scalar callback API."""
from cpython.pycapsule cimport PyCapsule_New
from cyroot.native.scalar cimport (
    ScalarCallback, make_scalar_callback, cyroot_scalar_descriptor,
)


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct CallbackContext:
    double target
    double scale
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    int failure_code


cdef int quadratic(double x, double* out, void* context) noexcept nogil:
    cdef CallbackContext* values = <CallbackContext*>context
    values.calls += 1
    if PyGILState_Check():
        values.gil_held += 1
    if values.fail_after >= 0 and values.calls > <size_t>values.fail_after:
        return values.failure_code
    out[0] = values.scale * (x * x - values.target)
    return 0


cdef class ContextOwner:
    """Python-owned context whose address remains valid while kept alive."""
    cdef CallbackContext values
    cdef cyroot_scalar_descriptor descriptor
    cdef object __weakref__

    def __init__(self, target=2., scale=1., fail_after=-1, failure_code=17):
        self.values.target = target
        self.values.scale = scale
        self.values.calls = 0
        self.values.gil_held = 0
        self.values.fail_after = fail_after
        self.values.failure_code = failure_code

    def callback(self):
        return make_scalar_callback(quadratic, <void*>&self.values, self)

    def capsule(self, version=1, struct_size=None, null_function=False, wrong_name=False):
        self.descriptor.struct_size = sizeof(cyroot_scalar_descriptor) if struct_size is None else struct_size
        self.descriptor.abi_version = version
        if null_function:
            self.descriptor.function = NULL
        else:
            self.descriptor.function = quadratic
        self.descriptor.context = <void*>&self.values
        if wrong_name:
            return PyCapsule_New(<void*>&self.descriptor, b'wrong.callback.name', NULL)
        return PyCapsule_New(<void*>&self.descriptor, b'cyroot.scalar_callback.v1', NULL)

    @property
    def calls(self):
        return self.values.calls

    @property
    def gil_held(self):
        return self.values.gil_held


def make_callback(target=2., scale=1., fail_after=-1, failure_code=17):
    owner = ContextOwner(target, scale, fail_after, failure_code)
    return owner.callback(), owner


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(double x, double* out) noexcept {
            entered_.store(true);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!released_.load()) {
                if (std::chrono::steady_clock::now() >= deadline) return 77;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            *out = x - 1.0;
            return 0;
        }
    };
    """
    cdef cppclass CyrootTestGate:
        CyrootTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(double x, double* out) noexcept nogil


cdef int waiting_residual(double x, double* out, void* context) noexcept nogil:
    return (<CyrootTestGate*>context).evaluate(x, out)


cdef class WaitingOwner:
    cdef CyrootTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_scalar_callback(waiting_residual, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()


from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_FORMAT
from cyroot.native._buffers cimport DoubleBuffer
from cyroot.native.buffers cimport VectorView


def format_only_metadata(source):
    cdef Py_buffer view
    PyObject_GetBuffer(source, &view, PyBUF_FORMAT)
    try:
        return view.itemsize, view.len, view.format.decode('ascii')
    finally:
        PyBuffer_Release(&view)


def sum_borrowed_vector(DoubleBuffer guard):
    cdef VectorView view = guard.vector_view()
    cdef size_t i
    cdef double total = 0.
    with nogil:
        for i in range(view.size()):
            total += view.get(i)
    return total


cdef int quadratic_derivative(double x, double* out, void* context) noexcept nogil:
    cdef CallbackContext* values = <CallbackContext*>context
    values.calls += 1
    if PyGILState_Check():
        values.gil_held += 1
    if values.fail_after >= 0 and values.calls > <size_t>values.fail_after:
        return values.failure_code
    out[0] = values.scale * 2 * x
    return 0


cdef class DerivativeOwner:
    """Independent derivative counters and lifetime for mixed callback tests."""
    cdef CallbackContext values
    cdef object __weakref__

    def __init__(self, scale=1., fail_after=-1, failure_code=29):
        self.values.scale = scale
        self.values.calls = 0
        self.values.gil_held = 0
        self.values.fail_after = fail_after
        self.values.failure_code = failure_code

    def callback(self):
        return make_scalar_callback(quadratic_derivative, <void*>&self.values, self)

    @property
    def calls(self):
        return self.values.calls

    @property
    def gil_held(self):
        return self.values.gil_held


def make_derivative(scale=1., fail_after=-1, failure_code=29):
    owner = DerivativeOwner(scale, fail_after, failure_code)
    return owner.callback(), owner


cdef int unwritten_scalar(double x, double* out, void* context) noexcept nogil:
    # Deliberately violates the success-output contract to test defensive NaN.
    return 0


def make_unwritten_callback():
    return make_scalar_callback(unwritten_scalar, NULL, None)
