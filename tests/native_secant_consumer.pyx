"""External secant callbacks with native contexts, statuses and GIL checks."""
from libc.stddef cimport size_t
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.secant cimport Secant


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    double target
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t partial_after
    int failure_code


cdef int quadratic(double x, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    parameters.calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after >= 0 and parameters.calls > <size_t>parameters.fail_after:
        return parameters.failure_code
    if parameters.partial_after < 0 or parameters.calls <= <size_t>parameters.partial_after:
        out[0] = x * x - parameters.target
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, target=2., fail_after=-1, failure_code=-17, partial_after=-1):
        self.parameters.target = target
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.failure_code = failure_code
        self.parameters.partial_after = partial_after

    def callback(self):
        return make_scalar_callback(quadratic, <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def history_allocation(Secant solver):
    if not solver._initialized or solver._busy:
        raise RuntimeError('An idle initialized solver is required.')
    if not solver._records.size():
        return 0, 0
    return <size_t>&solver._records[0], solver._records.size()


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootSecantTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(double x, double* out) noexcept {
            entered_.store(true);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!released_.load()) {
                if (std::chrono::steady_clock::now() >= deadline) return 83;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            *out = x - 2.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootSecantTestGate:
        CyrootSecantTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(double x, double* out) noexcept nogil


cdef int waiting_residual(double x, double* out, void* context) noexcept nogil:
    return (<CyrootSecantTestGate*>context).evaluate(x, out)


cdef class WaitingOwner:
    cdef CyrootSecantTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootSecantTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_scalar_callback(waiting_residual, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()
