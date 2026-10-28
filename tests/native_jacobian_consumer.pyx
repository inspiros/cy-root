"""Independent residual contexts for installed finite-difference consumers."""
from libc.stddef cimport size_t
from cyroot.native.vector cimport make_vector_callback, VectorNewton


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t partial_after
    int failure_code


cdef int coupled(const double* x, size_t n, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    if n != 2:
        return -101
    parameters.calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after >= 0 and parameters.calls > <size_t>parameters.fail_after:
        return parameters.failure_code
    out[0] = x[0] * x[0] + x[1] - 3
    if parameters.partial_after < 0 or parameters.calls <= <size_t>parameters.partial_after:
        out[1] = 2 * x[0] + x[1] * x[1] - 6
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, fail_after=-1, failure_code=-17, partial_after=-1):
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.failure_code = failure_code
        self.parameters.partial_after = partial_after

    def callback(self):
        return make_vector_callback(coupled, 2, <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def workspace_layout(VectorNewton solver):
    if not solver._initialized or solver._busy:
        raise RuntimeError('An idle initialized solver is required.')
    # FD scratch is part of the one reusable owning allocation. The pointers
    # and allocation capacities must stay fixed through resumed/full runs.
    cdef size_t pivots = 0, records = 0
    if solver._pivots.size():
        pivots = <size_t>&solver._pivots[0]
    if solver._records.size():
        records = <size_t>&solver._records[0]
    return (
        (<size_t>solver._storage.data(), <size_t>solver._x, <size_t>solver._f,
         <size_t>solver._jac, <size_t>solver._trial, <size_t>solver._trial_f,
         <size_t>solver._lu, <size_t>solver._delta, <size_t>solver._diff_work,
         <size_t>solver._history_data.data(), records, pivots),
        (solver._storage.size(), solver._history_data.size(),
         solver._records.size(), solver._pivots.size()),
    )


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootJacobianTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(const double* x, size_t n, double* out) noexcept {
            if (n != 2) return 81;
            entered_.store(true);
            auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!released_.load()) {
                if (std::chrono::steady_clock::now() >= deadline) return 83;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            out[0] = x[0] * x[0] + x[1] - 3.;
            out[1] = 2. * x[0] + x[1] * x[1] - 6.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootJacobianTestGate:
        CyrootJacobianTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(const double* x, size_t n, double* out) noexcept nogil


cdef int waiting_residual(const double* x, size_t n, double* out, void* context) noexcept nogil:
    return (<CyrootJacobianTestGate*>context).evaluate(x, n, out)


cdef class WaitingOwner:
    cdef CyrootJacobianTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootJacobianTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_vector_callback(waiting_residual, 2, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()


from cyroot.native.derivatives cimport JacobianPlan


def compute_direct(JacobianPlan plan, point=(1.4, 1.7), base=None,
                   fail_after=-1, failure_code=-19, partial_after=-1):
    """Trusted 2D engine call with canaries and unchanged-input observations."""
    if not plan._initialized or plan._dimension != 2:
        raise ValueError('The direct fixture requires an initialized 2D plan.')
    cdef double x[2]
    cdef double f0[2]
    cdef double output_guard[6]
    cdef double work_guard[12]
    cdef size_t calls = 0
    cdef int callback_code = 0
    cdef int status
    cdef size_t i
    cdef ContextOwner owner = ContextOwner(fail_after, failure_code, partial_after)
    x[0], x[1] = point
    if base is None:
        f0[0], f0[1] = x[0] * x[0] + x[1] - 3, 2 * x[0] + x[1] * x[1] - 6
    else:
        f0[0], f0[1] = base
    for i in range(6):
        output_guard[i] = 1234567.
    for i in range(12):
        work_guard[i] = 7654321.
    with nogil:
        status = plan.compute(x, f0, &output_guard[1], &work_guard[1],
                              coupled, <void*>&owner.parameters, &calls, &callback_code)
    return dict(status=status, calls=calls, callback_code=callback_code,
                jacobian=((output_guard[1], output_guard[3]),
                          (output_guard[2], output_guard[4])),
                point=(x[0], x[1]), base=(f0[0], f0[1]),
                restored_point=(work_guard[1], work_guard[2]),
                guards=(output_guard[0], output_guard[5], work_guard[0], work_guard[11]),
                gil_held=owner.parameters.gil_held)
