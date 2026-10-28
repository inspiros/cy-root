"""Installed Hessian POD consumer: canaries, contexts, allocation and GIL probes."""
from libc.stddef cimport size_t
from libcpp.vector cimport vector
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.hessians cimport HessianPlan, HessianConfig, compute_hessian


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef struct Parameters:
    size_t calls
    size_t gil_held
    Py_ssize_t fail_after
    Py_ssize_t partial_after
    int failure_code


cdef int quadratic(const double* x, size_t n, double* out, void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    if n != 2:
        return -101
    p.calls += 1
    if PyGILState_Check():
        p.gil_held += 1
    if p.fail_after >= 0 and p.calls > <size_t>p.fail_after:
        return p.failure_code
    out[0] = x[0]*x[0]+3*x[0]*x[1]+2*x[1]*x[1]+13.
    if p.partial_after < 0 or p.calls <= <size_t>p.partial_after:
        out[1] = 5*x[0]*x[0]-7*x[0]*x[1]+11*x[1]*x[1]-17.
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef object __weakref__

    def __init__(self, fail_after=-1, failure_code=-17, partial_after=-1):
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.partial_after = partial_after
        self.parameters.failure_code = failure_code

    def callback(self):
        return make_vector_callback(quadratic, 2, <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def public_type():
    return HessianPlan


def plan_allocation(HessianPlan plan):
    cdef HessianConfig config
    plan.copy_config(&config)
    return <size_t>config.steps, config.dimension, config.square, config.cube, config.work


def compute_direct(HessianPlan plan, point=(.75, -1.25), base=None,
                   fail_after=-1, failure_code=-19, partial_after=-1,
                   initial_calls=0, output_capacity=None, capacity=None,
                   use_config=False, config_overrides=None, repeats=1):
    """2D trusted call, with surrounding canaries and untouched-span checks."""
    if plan.dimension != 2:
        raise ValueError('The fixture requires an initialized 2D plan.')
    cdef HessianConfig config
    plan.copy_config(&config)
    cdef size_t needed = config.work
    cdef size_t count = needed if capacity is None else capacity
    cdef size_t output_count = config.cube if output_capacity is None else output_capacity
    cdef vector[double] work
    cdef double output[10]
    cdef double x[2]
    cdef double f0[2]
    cdef size_t calls = initial_calls, i, run, output_address, work_address
    cdef int code = 123, status = -1
    cdef ContextOwner owner = ContextOwner(fail_after, failure_code, partial_after)
    x[0], x[1] = point
    if base is None:
        f0[0] = x[0]*x[0]+3*x[0]*x[1]+2*x[1]*x[1]+13.
        f0[1] = 5*x[0]*x[0]-7*x[0]*x[1]+11*x[1]*x[1]-17.
    else:
        f0[0], f0[1] = base
    work.resize(needed+2)
    for i in range(needed+2):
        work[i] = 9876543.
    for i in range(10):
        output[i] = 1234567.
    output_address, work_address = <size_t>&output[1], <size_t>&work[1]
    if config_overrides is not None:
        for key, value in config_overrides.items():
            if key == 'dimension':
                config.dimension = value
            elif key == 'square':
                config.square = value
            elif key == 'cube':
                config.cube = value
            elif key == 'work':
                config.work = value
            elif key == 'scheme':
                config.scheme = value
            else:
                raise ValueError('Unknown configuration field.')
    for run in range(repeats):
        if use_config:
            with nogil:
                status = compute_hessian(&config, x, f0, &output[1], output_count,
                    &work[1], count, quadratic, <void*>&owner.parameters, &calls, &code)
        else:
            with nogil:
                status = plan.compute(x, f0, &output[1], output_count, &work[1], count,
                    quadratic, <void*>&owner.parameters, &calls, &code)
    # Contract scratch: 2*n*n axis values, then seven n-vectors before trial.
    cdef size_t trial = 2*2*2+7*2
    return dict(status=status, calls=calls, callback_code=code,
                flat=tuple(output[i+1] for i in range(8)),
                hessian=tuple(tuple(tuple(output[1+o+2*(j+2*k)] for k in range(2))
                                    for j in range(2)) for o in range(2)),
                point=(x[0], x[1]), base=(f0[0], f0[1]),
                restored_point=(work[1+trial], work[2+trial]),
                guards=(output[0], output[9], work[0], work[needed+1]),
                untouched_output=all(output[i] == 1234567. for i in range(10)),
                untouched_workspace=all(work[i] == 9876543. for i in range(needed+2)),
                stable=(output_address == <size_t>&output[1] and
                        work_address == <size_t>&work[1] and work.size() == needed+2),
                gil_held=owner.parameters.gil_held)


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootHessianTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(const double* x, size_t n, double* out) noexcept {
            if (n != 2) return 81;
            entered_.store(true);
            auto deadline = std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while (!released_.load()) {
                if (std::chrono::steady_clock::now() >= deadline) return 83;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            out[0] = x[0]*x[0]+3*x[0]*x[1]+2*x[1]*x[1]+13.;
            out[1] = 5*x[0]*x[0]-7*x[0]*x[1]+11*x[1]*x[1]-17.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootHessianTestGate:
        CyrootHessianTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(const double* x, size_t n, double* out) noexcept nogil


cdef int waiting(const double* x, size_t n, double* out, void* context) noexcept nogil:
    return (<CyrootHessianTestGate*>context).evaluate(x, n, out)


cdef class WaitingOwner:
    cdef CyrootHessianTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootHessianTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_vector_callback(waiting, 2, <void*>self.gate, self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()


# Rectangular coverage shares this installed consumer and all square ABI probes.
from cyroot.native.rectangular cimport make_rectangular_callback
from cyroot.native.hessians cimport (RectangularHessianPlan,
    RectangularHessianConfig, compute_rectangular_hessian)


cdef int rectangular_quadratic(const double* x, size_t n, size_t m,
                               double* out, void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    cdef size_t o, j, k, limit = m
    cdef double value
    p.calls += 1
    if PyGILState_Check():
        p.gil_held += 1
    if p.fail_after >= 0 and p.calls > <size_t>p.fail_after:
        return p.failure_code
    if p.partial_after >= 0 and p.calls > <size_t>p.partial_after:
        limit = m-1
    for o in range(limit):
        value = 1.+o
        for j in range(n):
            value += (10.*(o+1)+j+1)*x[j]*x[j]
            for k in range(j+1,n):
                value += (10.*(o+1)+j+k+1)*x[j]*x[k]
        out[o] = value
    return 0


cdef class RectangularContextOwner:
    cdef Parameters parameters
    cdef size_t n, m
    cdef object __weakref__

    def __init__(self, n, m, fail_after=-1, failure_code=-17, partial_after=-1):
        self.n, self.m = n, m
        self.parameters.calls = 0
        self.parameters.gil_held = 0
        self.parameters.fail_after = fail_after
        self.parameters.partial_after = partial_after
        self.parameters.failure_code = failure_code

    def callback(self):
        return make_rectangular_callback(rectangular_quadratic, self.n, self.m,
                                         <void*>&self.parameters, self)

    @property
    def calls(self):
        return self.parameters.calls

    @property
    def gil_held(self):
        return self.parameters.gil_held


def rectangular_allocation(RectangularHessianPlan plan):
    cdef RectangularHessianConfig config
    plan.copy_config(&config)
    return (<size_t>config.steps, config.input_dimension, config.output_dimension,
            config.matrix_size, config.tensor_size, config.work)


def compute_rectangular_direct(RectangularHessianPlan plan, point, base=None,
                               fail_after=-1, failure_code=-19, partial_after=-1,
                               initial_calls=0, output_capacity=None, capacity=None,
                               use_config=False, config_overrides=None, repeats=1):
    """Trusted POD and plan calls on dynamically sized, guarded native spans."""
    cdef RectangularHessianConfig config
    plan.copy_config(&config)
    cdef size_t n = config.input_dimension, m = config.output_dimension
    cdef size_t needed = config.work, tensor = config.tensor_size
    cdef size_t count = needed if capacity is None else capacity
    cdef size_t output_count = tensor if output_capacity is None else output_capacity
    cdef vector[double] work, output, x, f0
    cdef size_t calls = initial_calls, i, j, k, run, output_address, work_address
    cdef size_t trial = 2*n*m+7*n
    cdef int code = 123, status = -1
    cdef double value
    cdef RectangularContextOwner owner = RectangularContextOwner(
        n, m, fail_after, failure_code, partial_after)
    if len(point) != n or (base is not None and len(base) != m):
        raise ValueError('Fixture input/cache dimensions must match the plan.')
    x.resize(n)
    f0.resize(m)
    work.resize(needed+2)
    output.resize(tensor+2)
    for i in range(n):
        x[i] = point[i]
    for i in range(m):
        if base is not None:
            f0[i] = base[i]
        else:
            value = 1.+i
            for j in range(n):
                value += (10.*(i+1)+j+1)*x[j]*x[j]
                for k in range(j+1,n):
                    value += (10.*(i+1)+j+k+1)*x[j]*x[k]
            f0[i] = value
    for i in range(needed+2):
        work[i] = 9876543.
    for i in range(tensor+2):
        output[i] = 1234567.
    output_address, work_address = <size_t>&output[1], <size_t>&work[1]
    if config_overrides is not None:
        for key, override_value in config_overrides.items():
            if key == 'input_dimension':
                config.input_dimension = override_value
            elif key == 'output_dimension':
                config.output_dimension = override_value
            elif key == 'matrix_size':
                config.matrix_size = override_value
            elif key == 'tensor_size':
                config.tensor_size = override_value
            elif key == 'work':
                config.work = override_value
            elif key == 'scheme':
                config.scheme = override_value
            else:
                raise ValueError('Unknown configuration field.')
    for run in range(repeats):
        if use_config:
            with nogil:
                status = compute_rectangular_hessian(&config, &x[0], &f0[0],
                    &output[1], output_count, &work[1], count, rectangular_quadratic,
                    <void*>&owner.parameters, &calls, &code)
        else:
            with nogil:
                status = plan.compute(&x[0], &f0[0], &output[1], output_count,
                    &work[1], count, rectangular_quadratic,
                    <void*>&owner.parameters, &calls, &code)
    return dict(status=status, calls=calls, callback_code=code,
                flat=tuple(output[i+1] for i in range(tensor)),
                hessian=tuple(tuple(tuple(output[1+o+m*(j+n*k)] for k in range(n))
                                    for j in range(n)) for o in range(m)),
                point=tuple(x[i] for i in range(n)),
                base=tuple(f0[i] for i in range(m)),
                restored_point=tuple(work[1+trial+i] for i in range(n)),
                guards=(output[0],output[tensor+1],work[0],work[needed+1]),
                untouched_output=all(output[i] == 1234567. for i in range(tensor+2)),
                untouched_workspace=all(work[i] == 9876543. for i in range(needed+2)),
                stable=(output_address == <size_t>&output[1] and
                        work_address == <size_t>&work[1] and work.size() == needed+2),
                gil_held=owner.parameters.gil_held)
