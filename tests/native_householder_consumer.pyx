"""Public Householder consumer with runtime derivative-role contexts."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.scalar cimport make_scalar_callback
from cyroot.native.householder_solver cimport Householder, HouseholderRecord, HouseholderRole
from cyroot.native.householder cimport ScaledCoefficient


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t unwritten_after
    int failure_code
    double scale


cdef struct Role:
    size_t order
    Parameters* parameters


cdef int callback(double x, double* out, void* context) noexcept nogil:
    cdef Role* role = <Role*>context
    cdef Parameters* parameters = role.parameters
    parameters.calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.fail_after >= 0 and parameters.calls > <size_t>parameters.fail_after:
        return parameters.failure_code
    if parameters.unwritten_after < 0 or parameters.calls <= <size_t>parameters.unwritten_after:
        if role.order == 0:
            out[0] = parameters.scale*(x*x-2.)
        elif role.order == 1:
            out[0] = parameters.scale*2*x
        elif role.order == 2:
            out[0] = parameters.scale*2
        else:
            out[0] = 0.
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef vector[Role] roles
    cdef object __weakref__

    def __init__(self, size_t order=3, scale=1., fail_after=-1, failure_code=-17,
                 unwritten_after=-1):
        cdef size_t k
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.scale = scale
        self.parameters.fail_after = fail_after
        self.parameters.failure_code = failure_code
        self.parameters.unwritten_after = unwritten_after
        self.roles.resize(order+1)
        for k in range(order+1):
            self.roles[k].order = k
            self.roles[k].parameters = &self.parameters

    def function(self):
        return make_scalar_callback(callback, <void*>&self.roles[0], self)

    def derivatives(self):
        return tuple(make_scalar_callback(callback, <void*>&self.roles[k], self)
                     for k in range(1, self.roles.size()))

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def public_type():
    return Householder


def allocation(Householder solver):
    """Pointers, sizes and capacities for every solver-owned native vector."""
    if not solver._initialized or solver._busy:
        raise RuntimeError('An idle initialized solver is required.')
    return (
        (<size_t>solver._values.data(), solver._values.size(), solver._values.capacity()),
        (<size_t>solver._calls.data(), solver._calls.size(), solver._calls.capacity()),
        (<size_t>solver._roles.data(), solver._roles.size(), solver._roles.capacity()),
        (<size_t>solver._cached.data(), solver._cached.size(), solver._cached.capacity()),
        (<size_t>solver._recurrence.data(), solver._recurrence.size(), solver._recurrence.capacity()),
        (<size_t>solver._difference_workspace.data(), solver._difference_workspace.size(),
         solver._difference_workspace.capacity()),
        (<size_t>solver._records.data(), solver._records.size(), solver._records.capacity()),
        (<size_t>solver._history_derivatives.data(), solver._history_derivatives.size(),
         solver._history_derivatives.capacity()),
        (<size_t>solver._history_calls.data(), solver._history_calls.size(),
         solver._history_calls.capacity()),
    )


def storage_bytes(size_t order):
    """Minimum analytic workspace and per-record bytes from installed C ABI."""
    return (
        (order+1)*(sizeof(double)+sizeof(size_t)+sizeof(HouseholderRole)+
                   sizeof(unsigned char)+2*sizeof(ScaledCoefficient)),
        sizeof(HouseholderRecord)+order*sizeof(double)+(order+1)*sizeof(size_t),
    )


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootHouseholderTestGate {
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
            *out = x-2.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootHouseholderTestGate:
        CyrootHouseholderTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(double x, double* out) noexcept nogil


cdef int waiting_residual(double x, double* out, void* context) noexcept nogil:
    return (<CyrootHouseholderTestGate*>context).evaluate(x, out)


cdef class WaitingOwner:
    cdef CyrootHouseholderTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootHouseholderTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_scalar_callback(waiting_residual, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()
