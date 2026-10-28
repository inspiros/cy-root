"""Installed runtime-order scalar difference engine and callback probes."""
from libc.stddef cimport size_t
from libc.math cimport pow
from libcpp.vector cimport vector
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.higher_derivatives cimport (
    ScalarFiniteDifferencePlan, ScalarDifferenceConfig, compute_scalar_difference,
)


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    size_t order
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t unwritten_after
    int failure_code
    bint shifted


cdef int polynomial(double x, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    cdef double point = (x-1.)/2.220446049250313e-16 if parameters.shifted else x
    parameters.calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after >= 0 and parameters.calls > <size_t>parameters.fail_after:
        return parameters.failure_code
    if parameters.unwritten_after < 0 or parameters.calls <= <size_t>parameters.unwritten_after:
        out[0] = 1. + pow(point, <double>parameters.order)
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, order=3, fail_after=-1, failure_code=-17,
                 unwritten_after=-1, shifted=False):
        self.parameters.order = order
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.failure_code = failure_code
        self.parameters.unwritten_after = unwritten_after
        self.parameters.shifted = shifted

    def callback(self):
        return make_scalar_callback(polynomial, <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def compute_direct(ScalarFiniteDifferencePlan plan, point=0., base=None,
                   fail_after=-1, failure_code=-19, unwritten_after=-1,
                   initial_calls=0, capacity=None, shifted=False,
                   use_config=False, config_overrides=None):
    """Raw trusted engine with capacity checks and output/scratch canaries."""
    if not plan._initialized:
        raise ValueError('An initialized plan is required.')
    cdef double x = point
    cdef double value = (x-1.)/2.220446049250313e-16 if shifted else x
    cdef double f0 = 1.+pow(value, <double>plan._order) if base is None else base
    cdef double guarded_output[3]
    cdef size_t calls = initial_calls
    cdef size_t count = plan.workspace_size if capacity is None else capacity
    cdef size_t needed = plan.workspace_size
    cdef vector[double] work
    cdef int status, code = 123
    cdef size_t index
    cdef ScalarDifferenceConfig config
    plan.copy_config(&config)
    if config_overrides is not None:
        for key, override_value in config_overrides.items():
            if key == 'order':
                config.order = override_value
            elif key == 'nodes':
                config.nodes = override_value
            elif key == 'work':
                config.work = override_value
            elif key == 'scheme':
                config.scheme = override_value
            else:
                raise ValueError('Unknown native configuration field.')
    cdef ContextOwner owner = ContextOwner(plan._order, fail_after, failure_code,
                                           unwritten_after, shifted)
    work.resize(needed+2)
    for index in range(needed+2):
        work[index] = 9876543.
    guarded_output[0], guarded_output[1], guarded_output[2] = 1234567., 5555555., 7654321.
    if use_config:
        with nogil:
            status = compute_scalar_difference(
                &config, x, f0, &guarded_output[1], polynomial,
                <void*>&owner.parameters, &calls, &code, &work[1], count)
    else:
        with nogil:
            status = plan.compute(x, f0, &guarded_output[1], polynomial,
                                  <void*>&owner.parameters, &calls, &code, &work[1], count)
    return dict(status=status, calls=calls, callback_code=code,
                derivative=guarded_output[1], point=x, base=f0,
                guards=(guarded_output[0], guarded_output[2], work[0], work[needed+1]),
                untouched=all(work[index] == 9876543. for index in range(needed+2)),
                gil_held=owner.parameters.gil_held)


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootHigherDerivativeTestGate {
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
            *out = 1.+x*x*x;
            return 0;
        }
    };
    """
    cdef cppclass CyrootHigherDerivativeTestGate:
        CyrootHigherDerivativeTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(double x, double* out) noexcept nogil


cdef int waiting_residual(double x, double* out, void* context) noexcept nogil:
    return (<CyrootHigherDerivativeTestGate*>context).evaluate(x, out)


cdef class WaitingOwner:
    cdef CyrootHigherDerivativeTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootHigherDerivativeTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_scalar_callback(waiting_residual, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()
