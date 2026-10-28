"""Installed native vector Chebyshev callbacks, tensor capsule and probes."""
from libc.stddef cimport size_t
from cpython.pycapsule cimport PyCapsule_New
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.tensor_callbacks cimport make_hessian_callback, HessianFunction
from cyroot.native.vector_chebyshev cimport VectorChebyshev, VectorChebyshevRecord
from cyroot.native.tensor_ops cimport contract_hessian


cdef extern from "Python.h":
    int PyGILState_Check() noexcept nogil


cdef extern from "callbacks.h":
    ctypedef struct cyroot_hessian_descriptor:
        size_t struct_size
        unsigned int abi_version
        size_t dimension
        HessianFunction function
        void* context


cdef struct Parameters:
    double scale
    size_t calls[3]
    size_t gil_held
    Py_ssize_t fail_after[3]
    Py_ssize_t partial_after[3]
    int failure_code


cdef bint sample(Parameters* p, size_t role) noexcept nogil:
    p.calls[role] += 1
    if PyGILState_Check():
        p.gil_held += 1
    return p.fail_after[role] < 0 or p.calls[role] <= <size_t>p.fail_after[role]


cdef int residual(const double* x, size_t n, double* out, void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    if n != 2:
        return -101
    if not sample(p, 0):
        return p.failure_code
    out[0] = p.scale*(x[0]*x[0]+x[0]*x[1]+x[1]-5.)
    if p.partial_after[0] < 0 or p.calls[0] <= <size_t>p.partial_after[0]:
        out[1] = p.scale*(2*x[0]+x[1]*x[1]-6.)
    return 0


cdef int jacobian(const double* x, size_t n, double* out, void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    if n != 2:
        return -102
    if not sample(p, 1):
        return p.failure_code
    out[0], out[1], out[2] = p.scale*(2*x[0]+x[1]), 2*p.scale, p.scale*(x[0]+1)
    if p.partial_after[1] < 0 or p.calls[1] <= <size_t>p.partial_after[1]:
        out[3] = 2*x[1]*p.scale
    return 0


cdef int hessian(const double* x, size_t n, double* out, void* context) noexcept nogil:
    cdef Parameters* p = <Parameters*>context
    if n != 2:
        return -103
    if not sample(p, 2):
        return p.failure_code
    out[0], out[1] = 2*p.scale, 0.
    out[2], out[3] = p.scale, 0.
    out[4], out[5], out[6] = p.scale, 0., 0.
    if p.partial_after[2] < 0 or p.calls[2] <= <size_t>p.partial_after[2]:
        out[7] = 2*p.scale
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef cyroot_hessian_descriptor descriptor
    cdef object __weakref__

    def __init__(self, scale=1., fail_after=(-1,-1,-1), partial_after=(-1,-1,-1), failure_code=-37):
        cdef size_t k
        self.parameters.scale = scale
        self.parameters.gil_held = 0
        self.parameters.failure_code = failure_code
        for k in range(3):
            self.parameters.calls[k] = 0
            self.parameters.fail_after[k] = fail_after[k]
            self.parameters.partial_after[k] = partial_after[k]

    def function(self):
        return make_vector_callback(residual, 2, <void*>&self.parameters, self)

    def jacobian(self):
        return make_vector_callback(jacobian, 2, <void*>&self.parameters, self, jacobian=True)

    def hessian(self):
        return make_hessian_callback(hessian, 2, <void*>&self.parameters, self)

    def capsule(self, version=1, struct_size=None, dimension=2, null_function=False, wrong_name=False):
        self.descriptor.struct_size = sizeof(cyroot_hessian_descriptor) if struct_size is None else struct_size
        self.descriptor.abi_version = version
        self.descriptor.dimension = dimension
        if null_function:
            self.descriptor.function = NULL
        else:
            self.descriptor.function = hessian
        self.descriptor.context = <void*>&self.parameters
        return PyCapsule_New(<void*>&self.descriptor,
            b'wrong.hessian.callback' if wrong_name else b'cyroot.hessian_callback.v1', NULL)

    @property
    def calls(self):
        return tuple(self.parameters.calls[k] for k in range(3))

    @property
    def gil_held(self):
        return self.parameters.gil_held


def public_type():
    return VectorChebyshev


def workspace_layout(VectorChebyshev solver):
    if not solver._initialized or solver._busy:
        raise RuntimeError('An idle initialized solver is required.')
    return (
        (<size_t>solver._storage.data(),<size_t>solver._x,<size_t>solver._f,
         <size_t>solver._jac,<size_t>solver._hess,<size_t>solver._trial,
         <size_t>solver._trial_f,<size_t>solver._lu,<size_t>solver._s,
         <size_t>solver._q,<size_t>solver._t,<size_t>solver._delta,
         <size_t>solver._diff_work,<size_t>solver._history_data.data(),
         <size_t>solver._pivots.data(),<size_t>solver._records.data()),
        (solver._storage.size(),solver._history_data.size(),solver._pivots.size(),
         solver._pivots.capacity(),solver._records.size(),solver._records.capacity()),
    )


def storage_bytes(size_t n,size_t scratch=0,size_t plan_steps=0):
    return ((8*n+2*n*n+n*n*n+scratch+plan_steps)*sizeof(double)+n*sizeof(int),
            (2*n+n*n+n*n*n)*sizeof(double)+sizeof(VectorChebyshevRecord))


def contract_direct(tensor,direction,n_override=None):
    """Trusted 2D tensor contraction with guarded input/output observations."""
    cdef double values[8]
    cdef double step[2]
    cdef double output[4]
    cdef size_t n=2 if n_override is None else n_override
    cdef size_t o,j,k
    cdef int status
    for o in range(2):
        step[o]=direction[o]
        for j in range(2):
            for k in range(2):
                values[o+2*(j+2*k)]=tensor[o][j][k]
    output[0],output[1],output[2],output[3]=1234567.,5555555.,5555555.,7654321.
    with nogil:
        status=contract_hessian(values,step,n,&output[1])
    return dict(status=status,value=(output[1],output[2]),guards=(output[0],output[3]),
                direction=(step[0],step[1]),
                flat=tuple(values[o] for o in range(8)))


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootVectorChebyshevTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(const double* x, size_t n, double* out) noexcept {
            if (n != 2) return 81;
            entered_.store(true);
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while (!released_.load()) {
                if (std::chrono::steady_clock::now() >= deadline) return 83;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            out[0]=x[0]*x[0]+x[0]*x[1]+x[1]-5.;
            out[1]=2*x[0]+x[1]*x[1]-6.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootVectorChebyshevTestGate:
        CyrootVectorChebyshevTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(const double* x, size_t n, double* out) noexcept nogil


cdef int waiting(const double* x, size_t n, double* out, void* context) noexcept nogil:
    return (<CyrootVectorChebyshevTestGate*>context).evaluate(x,n,out)


cdef class WaitingOwner:
    cdef CyrootVectorChebyshevTestGate* gate

    def __cinit__(self):
        self.gate = new CyrootVectorChebyshevTestGate()

    def __dealloc__(self):
        del self.gate

    def callback(self):
        return make_vector_callback(waiting,2,<void*>self.gate,self)

    @property
    def entered(self):
        return self.gate.entered()

    def release(self):
        self.gate.release()


# Rectangular Hessian ABI probes share this already compiled higher-order fixture.
from libcpp.vector cimport vector
from cyroot.native.tensor_callbacks cimport (
    RectangularHessianCallback, RectangularHessianFunction,
    make_rectangular_hessian_callback, cyroot_rectangular_hessian_descriptor)
from cyroot.native.tensor_ops cimport (
    contract_rectangular_hessian, contract_rectangular_hessian_direction,
    rectangular_matrix_vector, contract_hessian_direction, matrix_vector)


cdef struct RectangularHessianState:
    size_t calls
    size_t gil_held
    bint partial
    int code


cdef int rectangular_hessian(const double* x, size_t n, size_t m,
                              double* out, void* context) noexcept nogil:
    cdef RectangularHessianState* state = <RectangularHessianState*>context
    cdef size_t o, j, k
    state.calls += 1
    if PyGILState_Check(): state.gil_held += 1
    for k in range(n):
        for j in range(n):
            for o in range(m):
                if not state.partial or o + m*(j+n*k) == 0:
                    out[o+m*(j+n*k)] = 100*(o+1)+10*(j+1)+k+1
    return state.code


cdef class RectangularHessianOwner:
    cdef RectangularHessianState state
    cdef cyroot_rectangular_hessian_descriptor descriptor
    cdef object __weakref__

    def __init__(self, partial=False, code=0):
        self.state.partial = partial
        self.state.code = code

    def callback(self, n, m):
        return make_rectangular_hessian_callback(rectangular_hessian,n,m,
                                                  <void*>&self.state,self)

    def capsule(self, n=2, m=3, version=1, struct_size=None,
                null_function=False, wrong_name=False):
        self.descriptor.struct_size = sizeof(cyroot_rectangular_hessian_descriptor) if struct_size is None else struct_size
        self.descriptor.abi_version = version
        self.descriptor.input_dimension = n
        self.descriptor.output_dimension = m
        if null_function:self.descriptor.function = NULL
        else:self.descriptor.function = rectangular_hessian
        self.descriptor.context = <void*>&self.state
        return PyCapsule_New(<void*>&self.descriptor,
            b'wrong.rectangular.hessian' if wrong_name else b'cyroot.rectangular_hessian_callback.v1',NULL)

    @property
    def counters(self): return self.state.calls,self.state.gil_held


def rectangular_hessian_null_factory():
    return make_rectangular_hessian_callback(NULL,2,3,NULL)


def rectangular_hessian_raw(RectangularHessianCallback callback, point):
    cdef size_t n=callback.input_dimension,m=callback.output_dimension
    cdef size_t count=m*n*n,i
    cdef vector[double] x,out
    cdef int code
    callback._enter()
    try:
        x.resize(n);out.resize(count+2)
        for i in range(n):x[i]=point[i]
        out[0],out[count+1]=1234567.,7654321.
        with nogil:
            code=callback.evaluate_raw(x.data(),out.data()+1)
        if callback._error is not None:
            error=callback._error
            callback._error=None
            raise error
        return dict(status=code,value=tuple(out[i+1] for i in range(count)),
                    guards=(out[0],out[count+1]))
    finally:
        callback._leave()


def rectangular_hessian_reserved(RectangularHessianCallback callback, action):
    callback._enter()
    try:return action()
    finally:callback._leave()


def rectangular_contract(tensor,direction,kind='quadratic',n_override=None,m_override=None):
    """Pack disjoint spans with guards; numerical helpers execute without the GIL."""
    cdef size_t n=len(direction),m=len(tensor),o,j,k,count=m*n*n
    cdef size_t out_count=m*n if kind=='direction' else m
    cdef size_t passed_n=n if n_override is None else n_override
    cdef size_t passed_m=m if m_override is None else m_override
    cdef vector[double] packed,step,out,square
    cdef int code,square_code=-999,mode
    if kind not in ('quadratic','direction','matrix'):raise ValueError('Unknown contraction.')
    mode=1 if kind=='direction' else (2 if kind=='matrix' else 0)
    if kind=='matrix':count=m*n
    packed.resize(count);step.resize(n);out.resize(out_count+2)
    square.resize(out_count)
    for j in range(n):
        step[j]=direction[j]
        for o in range(m):
            if kind=='matrix':packed[o+m*j]=tensor[o][j]
            else:
                for k in range(n):packed[o+m*(j+n*k)]=tensor[o][j][k]
    out[0],out[out_count+1]=1234567.,7654321.
    for o in range(out_count):out[o+1]=5555555.
    with nogil:
        if mode==1:
            code=contract_rectangular_hessian_direction(packed.data(),step.data(),passed_n,passed_m,out.data()+1)
        elif mode==2:
            code=rectangular_matrix_vector(packed.data(),step.data(),passed_n,passed_m,out.data()+1)
        else:
            code=contract_rectangular_hessian(packed.data(),step.data(),passed_n,passed_m,out.data()+1)
    if m==n and n_override is None and m_override is None:
        with nogil:
            if mode==1:square_code=contract_hessian_direction(packed.data(),step.data(),n,square.data())
            elif mode==2:square_code=matrix_vector(packed.data(),step.data(),n,square.data())
            else:square_code=contract_hessian(packed.data(),step.data(),n,square.data())
    return dict(status=code,value=tuple(out[o+1] for o in range(out_count)),
                guards=(out[0],out[out_count+1]),flat=tuple(packed[o] for o in range(count)),
                direction=tuple(step[j] for j in range(n)),square_status=square_code,
                square=tuple(square[o] for o in range(out_count)))


# Nonlinear rectangular Chebyshev probes reuse this compiled fixture.
from cyroot.native.rectangular cimport make_rectangular_callback

cdef struct RectangularChebyshevParameters:
    size_t calls[3]
    size_t gil_held
    int failed_role
    Py_ssize_t fail_after
    int code
    bint partial

cdef bint rectangular_sample(RectangularChebyshevParameters* p, int role) noexcept nogil:
    p.calls[role] += 1
    if PyGILState_Check():p.gil_held += 1
    return role != p.failed_role or p.fail_after < 0 or p.calls[role] <= <size_t>p.fail_after

cdef int rectangular_chebyshev_f(const double* x, size_t n, size_t m,
                                double* out, void* context) noexcept nogil:
    cdef RectangularChebyshevParameters* p = <RectangularChebyshevParameters*>context
    cdef size_t i
    if not rectangular_sample(p, 0):return p.code
    for i in range(m):
        if not p.partial or i+1 < m:out[i] = (i+1)*(x[0]*x[0]+(x[1] if n>1 else 0.)-2.)
    return 0

cdef int rectangular_chebyshev_j(const double* x, size_t n, size_t m,
                                double* out, void* context) noexcept nogil:
    cdef RectangularChebyshevParameters* p = <RectangularChebyshevParameters*>context
    cdef size_t i, j
    if not rectangular_sample(p, 1):return p.code
    for j in range(n):
        for i in range(m):out[i+m*j] = (i+1)*(2*x[0] if j==0 else 1. if j==1 else 0.)
    return 0

cdef int rectangular_chebyshev_h(const double* x, size_t n, size_t m,
                                double* out, void* context) noexcept nogil:
    cdef RectangularChebyshevParameters* p = <RectangularChebyshevParameters*>context
    cdef size_t i, j, k
    if not rectangular_sample(p, 2):return p.code
    for k in range(n):
        for j in range(n):
            for i in range(m):out[i+m*(j+n*k)] = 2*(i+1) if j==0 and k==0 else 0.
    return 0

cdef class RectangularChebyshevOwner:
    cdef RectangularChebyshevParameters parameters
    cdef object __weakref__
    def __init__(self, failed_role=-1, fail_after=-1, code=-51, partial=False):
        self.parameters.failed_role = failed_role
        self.parameters.fail_after = fail_after
        self.parameters.code = code
        self.parameters.partial = partial
    def function(self, n, m):return make_rectangular_callback(rectangular_chebyshev_f,n,m,<void*>&self.parameters,self)
    def jacobian(self, n, m):return make_rectangular_callback(rectangular_chebyshev_j,n,m,<void*>&self.parameters,self,jacobian=True)
    def hessian(self, n, m):return make_rectangular_hessian_callback(rectangular_chebyshev_h,n,m,<void*>&self.parameters,self)
    @property
    def calls(self):return tuple(self.parameters.calls[i] for i in range(3))
    @property
    def gil_held(self):return self.parameters.gil_held


def rectangular_chebyshev_storage(VectorChebyshev solver):
    cdef size_t plan_steps = 0
    if solver._rect_jplan is not None:plan_steps += solver._rect_jplan._axes._step_count
    if solver._rect_hplan is not None:plan_steps += solver._rect_hplan._step_count
    return ((solver._storage.size()+plan_steps)*sizeof(double)+solver._pivots.size()*sizeof(int),
            solver._record_stride*sizeof(double)+sizeof(VectorChebyshevRecord),
            0 if solver._linear_plan is None else solver._linear_plan._work_size)
