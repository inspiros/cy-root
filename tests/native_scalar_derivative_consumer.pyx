"""Installed-package scalar finite-difference consumer and native probes."""
from libc.stddef cimport size_t
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.scalar_derivatives cimport ScalarDerivativePlan


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    double target
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t unwritten_after
    int failure_code


cdef int quadratic(double x, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    parameters.calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after >= 0 and parameters.calls > <size_t>parameters.fail_after:
        return parameters.failure_code
    if parameters.unwritten_after < 0 or parameters.calls <= <size_t>parameters.unwritten_after:
        out[0] = x * x - parameters.target
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, target=2., fail_after=-1, failure_code=-17, unwritten_after=-1):
        self.parameters.target = target
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.failure_code = failure_code
        self.parameters.unwritten_after = unwritten_after

    def callback(self):
        return make_scalar_callback(quadratic, <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def compute_direct(ScalarDerivativePlan plan, point=1.5, base=None,
                   fail_after=-1, failure_code=-19, unwritten_after=-1,
                   initial_calls=0):
    """Call the raw engine with output canaries and a caller-owned counter."""
    if not plan._initialized:
        raise ValueError('An initialized scalar derivative plan is required.')
    cdef double x = point
    cdef double f0 = x * x - 2. if base is None else base
    cdef double guarded_output[3]
    cdef size_t calls = initial_calls
    cdef int callback_code = 123
    cdef int status
    cdef ContextOwner owner = ContextOwner(2., fail_after, failure_code, unwritten_after)
    guarded_output[0] = 1234567.
    guarded_output[1] = 9876543.
    guarded_output[2] = 7654321.
    with nogil:
        status = plan.compute(x, f0, &guarded_output[1], quadratic,
                              <void*>&owner.parameters, &calls, &callback_code)
    return dict(status=status, calls=calls, callback_code=callback_code,
                derivative=guarded_output[1], point=x, base=f0,
                guards=(guarded_output[0], guarded_output[2]),
                gil_held=owner.parameters.gil_held)


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootScalarDerivativeTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
        unsigned int calls_ = 0;
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(double x, double* out) noexcept {
            ++calls_;
            // A cached initial derivative makes call 1 the finite trial;
            // call 2 is then the first finite-difference stencil sample.
            if (calls_ >= 2) {
                entered_.store(true);
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                while (!released_.load()) {
                    if (std::chrono::steady_clock::now() >= deadline) return 83;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            *out = x * x - 2.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootScalarDerivativeTestGate:
        CyrootScalarDerivativeTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(double x, double* out) noexcept nogil


cdef int waiting_residual(double x, double* out, void* context) noexcept nogil:
    return (<CyrootScalarDerivativeTestGate*>context).evaluate(x, out)


cdef class WaitingOwner:
    cdef CyrootScalarDerivativeTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootScalarDerivativeTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_scalar_callback(waiting_residual, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()
