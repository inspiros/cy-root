"""Installed analytic scalar higher-order callbacks and allocation probes."""
from libc.stddef cimport size_t
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.higher_order cimport _ThirdOrderNewton


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    double target
    double scale
    size_t calls[3]
    size_t gil_held
    Py_ssize_t fail_after[3]
    Py_ssize_t partial_after[3]
    int failure_code


cdef int sample(double x, double* out, Parameters* parameters, size_t role) noexcept nogil:
    parameters.calls[role] += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after[role] >= 0 and parameters.calls[role] > <size_t>parameters.fail_after[role]:
        return parameters.failure_code
    if parameters.partial_after[role] < 0 or parameters.calls[role] <= <size_t>parameters.partial_after[role]:
        if role == 0:
            out[0] = parameters.scale * (x*x - parameters.target)
        elif role == 1:
            out[0] = parameters.scale * 2*x
        else:
            out[0] = parameters.scale * 2
    return 0


cdef int residual(double x, double* out, void* context) noexcept nogil:
    return sample(x, out, <Parameters*>context, 0)


cdef int derivative(double x, double* out, void* context) noexcept nogil:
    return sample(x, out, <Parameters*>context, 1)


cdef int second_derivative(double x, double* out, void* context) noexcept nogil:
    return sample(x, out, <Parameters*>context, 2)


cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, target=2., scale=1., fail_after=(-1,-1,-1), failure_code=-17,
                 partial_after=(-1,-1,-1)):
        cdef size_t role
        self.parameters.target = target
        self.parameters.scale = scale
        self.parameters.gil_held = 0
        self.parameters.failure_code = failure_code
        for role in range(3):
            self.parameters.calls[role] = 0
            self.parameters.fail_after[role] = fail_after[role]
            self.parameters.partial_after[role] = partial_after[role]

    def function(self):
        return make_scalar_callback(residual, <void*>&self.parameters, self)

    def derivative(self):
        return make_scalar_callback(derivative, <void*>&self.parameters, self)

    def second_derivative(self):
        return make_scalar_callback(second_derivative, <void*>&self.parameters, self)

    @property
    def calls(self):
        return tuple(self.parameters.calls[role] for role in range(3))

    @property
    def gil_held(self):
        return self.parameters.gil_held


def history_allocation(_ThirdOrderNewton solver):
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
    class CyrootHigherOrderTestGate {
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
    cdef cppclass CyrootHigherOrderTestGate:
        CyrootHigherOrderTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(double x, double* out) noexcept nogil


cdef int waiting_residual(double x, double* out, void* context) noexcept nogil:
    return (<CyrootHigherOrderTestGate*>context).evaluate(x, out)


cdef class WaitingOwner:
    cdef CyrootHigherOrderTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootHigherOrderTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_scalar_callback(waiting_residual, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()
