"""External consumer: column-major native Jacobians, contexts and statuses."""
from libc.stddef cimport size_t
from cpython.pycapsule cimport PyCapsule_New
from cyroot.native.vector cimport make_vector_callback, cyroot_vector_descriptor


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    double scale
    size_t residual_calls
    size_t jacobian_calls
    size_t gil_held
    Py_ssize_t residual_fail_after
    Py_ssize_t jacobian_fail_after
    int residual_code
    int jacobian_code
    bint linear
    bint partial_residual
    bint partial_jacobian
    Py_ssize_t partial_residual_after
    Py_ssize_t partial_jacobian_after


cdef int residual(const double* x, size_t n, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    if n != 2:
        return -101
    parameters.residual_calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.residual_fail_after >= 0 and parameters.residual_calls > <size_t>parameters.residual_fail_after:
        return parameters.residual_code
    if parameters.linear:
        out[0] = parameters.scale * (2 * x[0] + 3 * x[1] + 4)
        if not parameters.partial_residual and not (parameters.partial_residual_after >= 0 and parameters.residual_calls > <size_t>parameters.partial_residual_after):
            out[1] = parameters.scale * (5 * x[0] + 7 * x[1] + 9)
    else:
        out[0] = parameters.scale * (x[0] * x[0] + x[1] - 3)
        if not parameters.partial_residual and not (parameters.partial_residual_after >= 0 and parameters.residual_calls > <size_t>parameters.partial_residual_after):
            out[1] = parameters.scale * (2 * x[0] + x[1] * x[1] - 6)
    return 0


cdef int jacobian(const double* x, size_t n, double* out, void* context) noexcept nogil:
    cdef Parameters* parameters = <Parameters*>context
    if n != 2:
        return -103
    parameters.jacobian_calls += 1
    if PyGILState_Check():
        parameters.gil_held += 1
    if parameters.jacobian_fail_after >= 0 and parameters.jacobian_calls > <size_t>parameters.jacobian_fail_after:
        return parameters.jacobian_code
    # out[row + column*n], not row-major Python nested sequence order.
    if parameters.linear:
        out[0], out[1], out[2] = 2 * parameters.scale, 5 * parameters.scale, 3 * parameters.scale
        if not parameters.partial_jacobian and not (parameters.partial_jacobian_after >= 0 and parameters.jacobian_calls > <size_t>parameters.partial_jacobian_after):
            out[3] = 7 * parameters.scale
    else:
        out[0], out[1], out[2] = 2 * x[0] * parameters.scale, 2 * parameters.scale, parameters.scale
        if not parameters.partial_jacobian and not (parameters.partial_jacobian_after >= 0 and parameters.jacobian_calls > <size_t>parameters.partial_jacobian_after):
            out[3] = 2 * x[1] * parameters.scale
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef cyroot_vector_descriptor residual_descriptor
    cdef cyroot_vector_descriptor jacobian_descriptor
    cdef object __weakref__

    def __init__(self, scale=1., linear=False, residual_fail_after=-1,
                 jacobian_fail_after=-1, residual_code=-17, jacobian_code=29,
                 partial_residual=False, partial_jacobian=False,
                 partial_residual_after=-1, partial_jacobian_after=-1):
        self.parameters.scale = scale
        self.parameters.linear = linear
        self.parameters.residual_calls = 0
        self.parameters.jacobian_calls = 0
        self.parameters.gil_held = 0
        self.parameters.residual_fail_after = residual_fail_after
        self.parameters.jacobian_fail_after = jacobian_fail_after
        self.parameters.residual_code = residual_code
        self.parameters.jacobian_code = jacobian_code
        self.parameters.partial_residual = partial_residual
        self.parameters.partial_jacobian = partial_jacobian
        self.parameters.partial_residual_after = partial_residual_after
        self.parameters.partial_jacobian_after = partial_jacobian_after

    def function(self):
        return make_vector_callback(residual, 2, <void*>&self.parameters, self)

    def jacobian(self):
        return make_vector_callback(jacobian, 2, <void*>&self.parameters, self, jacobian=True)

    def capsule(self, kind=1, version=1, struct_size=None, dimension=2,
                null_function=False, wrong_name=False):
        cdef cyroot_vector_descriptor* descriptor = (
            &self.jacobian_descriptor if kind == 2 else &self.residual_descriptor)
        descriptor.struct_size = sizeof(cyroot_vector_descriptor) if struct_size is None else struct_size
        descriptor.abi_version = version
        descriptor.dimension = dimension
        descriptor.kind = kind
        descriptor.context = <void*>&self.parameters
        if null_function:
            descriptor.function = NULL
        elif kind == 2:
            descriptor.function = jacobian
        else:
            descriptor.function = residual
        if wrong_name:
            return PyCapsule_New(<void*>descriptor, b'wrong.vector.callback', NULL)
        return PyCapsule_New(<void*>descriptor, b'cyroot.vector_callback.v1', NULL)

    @property
    def calls(self):
        return self.parameters.residual_calls, self.parameters.jacobian_calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootVectorTestGate {
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
            out[0] = x[0] - 1.;
            out[1] = x[1] - 2.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootVectorTestGate:
        CyrootVectorTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(const double* x, size_t n, double* out) noexcept nogil


cdef int waiting_residual(const double* x, size_t n, double* out, void* context) noexcept nogil:
    return (<CyrootVectorTestGate*>context).evaluate(x, n, out)


cdef class WaitingOwner:
    cdef CyrootVectorTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootVectorTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_vector_callback(waiting_residual, 2, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()


from cyroot.native.vector cimport VectorNewton


def workspace_layout(VectorNewton solver):
    """Observe allocation addresses and capacities through the installed .pxd."""
    if not solver._initialized or solver._busy:
        raise RuntimeError('An idle initialized solver is required.')
    cdef size_t records = 0, pivots = 0
    if solver._records.size() != 0:
        records = <size_t>&solver._records[0]
    if solver._pivots.size() != 0:
        pivots = <size_t>&solver._pivots[0]
    return (
        (<size_t>solver._storage.data(), <size_t>solver._x, <size_t>solver._f,
         <size_t>solver._jac, <size_t>solver._trial, <size_t>solver._trial_f,
         <size_t>solver._lu, <size_t>solver._delta,
         <size_t>solver._history_data.data(), records, pivots),
        (solver._storage.size(), solver._history_data.size(),
         solver._records.size(), solver._pivots.size()),
    )


def pivot_values(VectorNewton solver):
    """Only read after a nontrivial solve has initialized LAPACK's pivots."""
    if not solver._initialized or solver._busy or solver._state.iters == 0:
        raise RuntimeError('A completed linear solve is required.')
    return tuple(solver._pivots[i] for i in range(solver._pivots.size()))
