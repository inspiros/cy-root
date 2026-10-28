"""External second-derivative engine calls and higher-order native probes."""
from libc.stddef cimport size_t
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.scalar_derivatives cimport ScalarSecondDerivativePlan
from cyroot.native.higher_order cimport _ThirdOrderNewton


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    double target
    double scale
    int kind
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t unwritten_after
    int failure_code


cdef int residual(double x, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    cdef double delta
    parameters.calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after >= 0 and parameters.calls > <size_t>parameters.fail_after:
        return parameters.failure_code
    if parameters.unwritten_after < 0 or parameters.calls <= <size_t>parameters.unwritten_after:
        if parameters.kind == 2:
            out[0] = 1. + (x/1.5)*x*1e308
        elif parameters.kind == 1:
            delta = x-1.
            out[0] = 1e-16 + delta + 4503599627370496. * delta * delta
        else:
            out[0] = parameters.scale * (x*x-parameters.target)
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, target=2., scale=1., fail_after=-1, failure_code=-17,
                 unwritten_after=-1, shifted=False, overflow_curve=False):
        self.parameters.target = target
        self.parameters.scale = scale
        self.parameters.kind = 2 if overflow_curve else 1 if shifted else 0
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.failure_code = failure_code
        self.parameters.unwritten_after = unwritten_after

    def callback(self):
        return make_scalar_callback(residual, <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def compute_direct(ScalarSecondDerivativePlan plan, point=1.5, base=None,
                   fail_after=-1, failure_code=-19, unwritten_after=-1,
                   initial_calls=0, shifted=False, overflow_curve=False):
    if not plan._initialized:
        raise ValueError('An initialized second derivative plan is required.')
    cdef double x = point
    cdef double delta = x-1.
    cdef double f0
    cdef double output[3]
    cdef size_t calls = initial_calls
    cdef int code = 123, status
    cdef ContextOwner owner = ContextOwner(2., 1., fail_after, failure_code,
                                           unwritten_after, shifted, overflow_curve)
    if base is None and overflow_curve:
        f0 = 1. + (x/1.5)*x*1e308
    elif base is None:
        f0 = 1e-16 + delta + 4503599627370496. * delta * delta if shifted else x*x-2.
    else:
        f0 = base
    output[0], output[1], output[2] = 1234567., 9876543., 7654321.
    with nogil:
        status = plan.compute(x, f0, &output[1], residual, <void*>&owner.parameters,
                              &calls, &code)
    return dict(status=status, calls=calls, callback_code=code,
                derivative=output[1], point=x, base=f0,
                guards=(output[0], output[2]), gil_held=owner.parameters.gil_held)


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
    class CyrootSecondDerivativeTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
        unsigned int calls_ = 0;
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(double x, double* out) noexcept {
            ++calls_;
            // Cached starting derivatives make call 1 the finite trial and
            // call 2 the first numerical second-derivative sample.
            if (calls_ >= 2) {
                entered_.store(true);
                auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
                while (!released_.load()) {
                    if (std::chrono::steady_clock::now() >= deadline) return 83;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }
            *out = x*x-2.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootSecondDerivativeTestGate:
        CyrootSecondDerivativeTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(double x, double* out) noexcept nogil


cdef int waiting_residual(double x, double* out, void* context) noexcept nogil:
    return (<CyrootSecondDerivativeTestGate*>context).evaluate(x, out)


cdef class WaitingOwner:
    cdef CyrootSecondDerivativeTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootSecondDerivativeTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_scalar_callback(waiting_residual, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()
