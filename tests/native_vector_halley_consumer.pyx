"""Installed matrix Halley family: shared workspace, callbacks and primitives."""
from libc.stddef cimport size_t
from cpython.pycapsule cimport PyCapsule_New
from cyroot.native.vector cimport make_vector_callback
from cyroot.native.tensor_callbacks cimport make_hessian_callback, HessianFunction
from cyroot.native.vector_chebyshev cimport (
    VectorChebyshev, VectorHalley, VectorSuperHalley, VectorChebyshevRecord,
)
from cyroot.native.tensor_ops cimport contract_hessian_direction, negative_half_sum


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


cdef bint sample(Parameters* p,size_t role) noexcept nogil:
    p.calls[role]+=1
    if PyGILState_Check():p.gil_held+=1
    return p.fail_after[role]<0 or p.calls[role]<=<size_t>p.fail_after[role]


cdef int residual(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    if n!=2:return -101
    if not sample(p,0):return p.failure_code
    out[0]=p.scale*(x[0]*x[0]+x[0]*x[1]+x[1]-5.)
    if p.partial_after[0]<0 or p.calls[0]<=<size_t>p.partial_after[0]:
        out[1]=p.scale*(2*x[0]+x[1]*x[1]-6.)
    return 0


cdef int jacobian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    if n!=2:return -102
    if not sample(p,1):return p.failure_code
    out[0],out[1],out[2]=p.scale*(2*x[0]+x[1]),2*p.scale,p.scale*(x[0]+1)
    if p.partial_after[1]<0 or p.calls[1]<=<size_t>p.partial_after[1]:out[3]=2*x[1]*p.scale
    return 0


cdef int hessian(const double* x,size_t n,double* out,void* context) noexcept nogil:
    cdef Parameters* p=<Parameters*>context
    if n!=2:return -103
    if not sample(p,2):return p.failure_code
    out[0],out[1],out[2],out[3]=2*p.scale,0.,p.scale,0.
    out[4],out[5],out[6]=p.scale,0.,0.
    if p.partial_after[2]<0 or p.calls[2]<=<size_t>p.partial_after[2]:out[7]=2*p.scale
    return 0


cdef class ContextOwner:
    cdef Parameters parameters
    cdef cyroot_hessian_descriptor descriptor
    cdef object __weakref__

    def __init__(self,scale=1.,fail_after=(-1,-1,-1),partial_after=(-1,-1,-1),failure_code=-37):
        cdef size_t k
        self.parameters.scale=scale
        self.parameters.gil_held=0
        self.parameters.failure_code=failure_code
        for k in range(3):
            self.parameters.calls[k]=0
            self.parameters.fail_after[k]=fail_after[k]
            self.parameters.partial_after[k]=partial_after[k]

    def function(self):return make_vector_callback(residual,2,<void*>&self.parameters,self)
    def jacobian(self):return make_vector_callback(jacobian,2,<void*>&self.parameters,self,jacobian=True)
    def hessian(self):return make_hessian_callback(hessian,2,<void*>&self.parameters,self)

    def capsule(self):
        self.descriptor.struct_size=sizeof(cyroot_hessian_descriptor)
        self.descriptor.abi_version=1
        self.descriptor.dimension=2
        self.descriptor.function=hessian
        self.descriptor.context=<void*>&self.parameters
        return PyCapsule_New(<void*>&self.descriptor,b'cyroot.hessian_callback.v1',NULL)

    @property
    def calls(self):return tuple(self.parameters.calls[k] for k in range(3))
    @property
    def gil_held(self):return self.parameters.gil_held


def public_types():return VectorHalley,VectorSuperHalley


def allocation(VectorChebyshev solver):
    if not solver._initialized or solver._busy:raise RuntimeError('An idle initialized solver is required.')
    return (
        (<size_t>solver._storage.data(),<size_t>solver._x,<size_t>solver._f,
         <size_t>solver._jac,<size_t>solver._hess,<size_t>solver._trial,<size_t>solver._trial_f,
         <size_t>solver._lu,<size_t>solver._s,<size_t>solver._q,<size_t>solver._t,
         <size_t>solver._delta,<size_t>solver._diff_work,<size_t>solver._secondary,
         <size_t>solver._history_data.data(),<size_t>solver._pivots.data(),<size_t>solver._records.data()),
        (solver._storage.size(),solver._history_data.size(),solver._pivots.size(),solver._pivots.capacity(),
         solver._records.size(),solver._records.capacity()),
    )


def storage_bytes(size_t n,bint secondary=True,size_t scratch=0,size_t plan_steps=0):
    return ((8*n+(3 if secondary else 2)*n*n+n*n*n+scratch+plan_steps)*sizeof(double)+n*sizeof(int),
            (2*n+n*n+n*n*n)*sizeof(double)+sizeof(VectorChebyshevRecord))


def directional(tensor,direction,n_override=None):
    cdef double values[8]
    cdef double step[2]
    cdef double output[6]
    cdef size_t n=2 if n_override is None else n_override
    cdef size_t o,j,k
    cdef int status
    for o in range(2):
        step[o]=direction[o]
        for j in range(2):
            for k in range(2):values[o+2*(j+2*k)]=tensor[o][j][k]
    for o in range(6):output[o]=1234567.
    with nogil:status=contract_hessian_direction(values,step,n,&output[1])
    return dict(status=status,matrix=((output[1],output[3]),(output[2],output[4])),
                flat=(output[1],output[2],output[3],output[4]),guards=(output[0],output[5]),
                direction=(step[0],step[1]))


def negative_mean(double a,double b):
    cdef double result
    with nogil:result=negative_half_sum(a,b)
    return result


cdef extern from *:
    """
    #include <atomic>
    #include <chrono>
    #include <thread>
    class CyrootVectorHalleyTestGate {
        std::atomic<bool> entered_{false};
        std::atomic<bool> released_{false};
    public:
        bool entered() const noexcept { return entered_.load(); }
        void release() noexcept { released_.store(true); }
        int evaluate(const double* x,std::size_t n,double* out) noexcept {
            if(n!=2)return 81;
            entered_.store(true);
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
            while(!released_.load()) {
                if(std::chrono::steady_clock::now()>=deadline)return 83;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            out[0]=x[0]*x[0]+x[0]*x[1]+x[1]-5.;
            out[1]=2*x[0]+x[1]*x[1]-6.;
            return 0;
        }
    };
    """
    cdef cppclass CyrootVectorHalleyTestGate:
        CyrootVectorHalleyTestGate() except +
        bint entered() noexcept nogil
        void release() noexcept nogil
        int evaluate(const double* x,size_t n,double* out) noexcept nogil


cdef int waiting(const double* x,size_t n,double* out,void* context) noexcept nogil:
    return (<CyrootVectorHalleyTestGate*>context).evaluate(x,n,out)


cdef class WaitingOwner:
    cdef CyrootVectorHalleyTestGate* gate
    def __cinit__(self):self.gate=new CyrootVectorHalleyTestGate()
    def __dealloc__(self):del self.gate
    def callback(self):return make_vector_callback(waiting,2,<void*>self.gate,self)
    @property
    def entered(self):return self.gate.entered()
    def release(self):self.gate.release()
